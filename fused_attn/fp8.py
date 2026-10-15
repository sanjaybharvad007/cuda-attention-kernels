"""FP8 (E4M3) KV cache attention.

Decode attention is memory-bound on reading K/V, so storing the cache in FP8
halves HBM traffic versus fp16/bf16.  E4M3 has only 3 mantissa bits, so the
values are scaled per (batch, kv-head, block of ``QBLOCK`` keys): each block is
divided by its absmax/448 before the cast, and the kernel multiplies the
scale back in after the Tensor-Core dot (one FMUL per tile, not per element).

    k8, ks = quantize_kv_fp8(k)          # [B,Hkv,N,D] float8_e4m3fn, [B,Hkv,ceil(N/64)] fp32
    v8, vs = quantize_kv_fp8(v)
    o = attention_fp8_triton(q, k8, ks, v8, vs, causal=True)

Accuracy: with per-block scaling the max-abs error vs. the fp32 reference is
~1e-2 on unit-variance inputs (vs ~1e-3 for fp16) — see tests.  The kernel
needs a GPU with FP8 conversion support (sm_89 Ada / sm_90 Hopper).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

QBLOCK = 64                 # keys per scale
E4M3_MAX = 448.0
FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)


def quantize_kv_fp8(x: torch.Tensor, qblock: int = QBLOCK) -> tuple[torch.Tensor, torch.Tensor]:
    """x: [B, H_kv, N, D] (any float dtype) -> (fp8 tensor same shape, scales [B, H_kv, ceil(N/qblock)])."""
    assert FP8_DTYPE is not None, "this torch build has no float8_e4m3fn"
    b, h, n, d = x.shape
    nblk = (n + qblock - 1) // qblock
    pad = nblk * qblock - n
    xf = x.float()
    if pad:
        xf = torch.nn.functional.pad(xf, (0, 0, 0, pad))
    blocks = xf.view(b, h, nblk, qblock, d)
    amax = blocks.abs().amax(dim=(-1, -2)).clamp_min(1e-12)         # [B, H, nblk]
    scale = amax / E4M3_MAX
    q = (blocks / scale[..., None, None]).clamp(-E4M3_MAX, E4M3_MAX)
    q8 = q.to(FP8_DTYPE).view(b, h, nblk * qblock, d)[:, :, :n].contiguous()
    return q8, scale.contiguous()


def dequantize_kv_fp8(x8: torch.Tensor, scale: torch.Tensor, qblock: int = QBLOCK) -> torch.Tensor:
    b, h, n, d = x8.shape
    nblk = scale.shape[-1]
    xf = x8.float()
    pad = nblk * qblock - n
    if pad:
        xf = torch.nn.functional.pad(xf, (0, 0, 0, pad))
    out = xf.view(b, h, nblk, qblock, d) * scale[..., None, None]
    return out.view(b, h, nblk * qblock, d)[:, :, :n]


@triton.jit
def _attn_fp8_kernel(
    Q, K8, KS, V8, VS, O,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_sb, stride_sh,                      # scales [B, H_kv, nblk]
    stride_ob, stride_oh, stride_om, stride_od,
    H_Q, H_KV, SEQLEN_Q, SEQLEN_K, sm_scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H_Q
    off_h = off_bh % H_Q
    off_hkv = off_h // (H_Q // H_KV)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < SEQLEN_Q, other=0.0)
    k_base = K8 + off_b * stride_kb + off_hkv * stride_kh
    v_base = V8 + off_b * stride_vb + off_hkv * stride_vh
    ks_base = KS + off_b * stride_sb + off_hkv * stride_sh
    vs_base = VS + off_b * stride_sb + off_hkv * stride_sh

    causal_offset = SEQLEN_K - SEQLEN_Q
    hi = SEQLEN_K
    if IS_CAUSAL:
        hi = tl.maximum(0, tl.minimum(SEQLEN_K, (start_m + 1) * BLOCK_M + causal_offset))

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    for start_n in range(0, hi, BLOCK_N):
        cols = start_n + offs_n
        kv_mask = cols[:, None] < SEQLEN_K
        k_scale = tl.load(ks_base + start_n // BLOCK_N)        # one scale per tile (BLOCK_N == QBLOCK)
        v_scale = tl.load(vs_base + start_n // BLOCK_N)
        k8 = tl.load(k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd, mask=kv_mask, other=0.0)
        k = k8.to(q.dtype)                                      # FP8 -> bf16/fp16 for the MMA
        qk = tl.dot(q, tl.trans(k)) * (qk_scale * k_scale)      # scale folded in after the dot
        valid = cols[None, :] < SEQLEN_K
        if IS_CAUSAL:
            valid = valid & (cols[None, :] <= offs_m[:, None] + causal_offset)
        qk = tl.where(valid, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v8 = tl.load(v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd, mask=kv_mask, other=0.0)
        acc += tl.dot(p.to(q.dtype), v8.to(q.dtype)) * v_scale
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe[:, None]
    o_ptrs = O + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=offs_m[:, None] < SEQLEN_Q)


def attention_fp8_triton(
    q: torch.Tensor, k8: torch.Tensor, k_scale: torch.Tensor, v8: torch.Tensor, v_scale: torch.Tensor,
    causal: bool = False, scale: Optional[float] = None,
) -> torch.Tensor:
    assert q.dtype in (torch.float16, torch.bfloat16), "Q must be fp16/bf16 (FP8 is converted to Q's dtype for the MMA)"
    assert k8.dtype == FP8_DTYPE and v8.dtype == FP8_DTYPE
    b, h_q, n_q, d = q.shape
    _, h_kv, n_k, _ = k8.shape
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    o = torch.empty_like(q)
    block_m = 64 if n_q >= 64 else 16
    grid = (triton.cdiv(n_q, block_m), b * h_q)
    _attn_fp8_kernel[grid](
        q, k8, k_scale, v8, v_scale, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k8.stride(0), k8.stride(1), k8.stride(2), k8.stride(3),
        v8.stride(0), v8.stride(1), v8.stride(2), v8.stride(3),
        k_scale.stride(0), k_scale.stride(1),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        h_q, h_kv, n_q, n_k, scale,
        IS_CAUSAL=causal, BLOCK_M=block_m, BLOCK_N=QBLOCK, BLOCK_D=d, num_warps=4,
    )
    return o
