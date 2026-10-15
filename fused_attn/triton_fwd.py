"""Fused multi-head attention forward kernel in Triton.

Design (FlashAttention-2 style, forward / inference only):

* One program per (BLOCK_M query rows, batch*head).  The Q tile is loaded once
  and kept in registers; K/V tiles stream through shared memory.
* Online softmax: running row-max ``m_i`` and row-sum ``l_i`` are updated per
  K tile so the full N x N score matrix never touches HBM.
* ``tl.dot`` lowers to Tensor Core MMA (mma.sync / wgmma) for fp16/bf16 inputs
  with fp32 accumulation.
* exp2 with the log2(e) folded into the softmax scale (one FMUL fewer per
  element than exp).
* Causal masking is bottom-right aligned and the K loop stops at the diagonal,
  so the causal kernel does ~half the work of the dense kernel.
* GQA/MQA: query head h reads KV head h // (H_q // H_kv).
* Variable-length batches: ``IS_VARLEN`` reads per-sequence bounds from
  cu_seqlens and masks the tails; no padding tokens are ever computed.

The kernel also runs under ``TRITON_INTERPRET=1`` (numpy interpreter) which is
how CI exercises it on CPU-only runners.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634

# ----------------------------------------------------------------------------
# Kernel
# ----------------------------------------------------------------------------


@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    k_base, v_base,
    stride_kn, stride_kd, stride_vn, stride_vd,
    seqlen_k, causal_offset,
    offs_m, offs_n, offs_d,
    qk_scale,
    lo, hi,
    IS_CAUSAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        cols = start_n + offs_n
        kv_mask = cols[:, None] < seqlen_k

        # K tile [BLOCK_N, BLOCK_D] -> we need K^T for Q @ K^T.
        k_ptrs = k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)

        # S = Q K^T  (fp32 accumulate on tensor cores), already scaled by log2(e)/sqrt(d)
        qk = tl.dot(q, tl.trans(k)) * qk_scale

        valid = cols[None, :] < seqlen_k
        if IS_CAUSAL:
            valid = valid & (cols[None, :] <= offs_m[:, None] + causal_offset)
        qk = tl.where(valid, qk, float("-inf"))

        # --- online softmax update --------------------------------------
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        # A row with no valid key yet has m_new == -inf; guard the exp2.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        # O += P V
        v_ptrs = v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE,
    cu_seqlens_q, cu_seqlens_k,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh,
    H_Q, H_KV, SEQLEN_Q, SEQLEN_K,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    WRITE_LSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H_Q
    off_h = off_bh % H_Q
    off_hkv = off_h // (H_Q // H_KV)

    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + off_b)
        q_end = tl.load(cu_seqlens_q + off_b + 1)
        k_start = tl.load(cu_seqlens_k + off_b)
        k_end = tl.load(cu_seqlens_k + off_b + 1)
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start
        # Packed layout: batch stride is unused; rows are offset by cu_seqlens.
        q_base = Q + q_start * stride_qm + off_h * stride_qh
        k_base = K + k_start * stride_kn + off_hkv * stride_kh
        v_base = V + k_start * stride_vn + off_hkv * stride_vh
        o_base = O + q_start * stride_om + off_h * stride_oh
    else:
        seqlen_q = SEQLEN_Q
        seqlen_k = SEQLEN_K
        q_base = Q + off_b * stride_qb + off_h * stride_qh
        k_base = K + off_b * stride_kb + off_hkv * stride_kh
        v_base = V + off_b * stride_vb + off_hkv * stride_vh
        o_base = O + off_b * stride_ob + off_h * stride_oh

    if start_m * BLOCK_M >= seqlen_q:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    qk_scale = sm_scale * 1.4426950408889634  # fold log2(e) so we can use exp2
    causal_offset = seqlen_k - seqlen_q       # bottom-right alignment

    if IS_CAUSAL:
        # Last key this block can see: row (start_m+1)*BLOCK_M - 1 + offset.
        hi = tl.minimum(seqlen_k, (start_m + 1) * BLOCK_M + causal_offset)
        hi = tl.maximum(hi, 0)
    else:
        hi = seqlen_k

    acc, l_i, m_i = _attn_fwd_inner(
        acc, l_i, m_i, q,
        k_base, v_base,
        stride_kn, stride_kd, stride_vn, stride_vd,
        seqlen_k, causal_offset,
        offs_m, offs_n, offs_d,
        qk_scale, 0, hi,
        IS_CAUSAL, BLOCK_N,
    )

    # Rows that saw no valid key have l_i == 0 -> emit zeros, not NaN.
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe[:, None]
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)

    if WRITE_LSE:
        lse = (m_i + tl.math.log2(l_safe)) / 1.4426950408889634  # natural-log LSE
        l_ptrs = LSE + off_b * stride_lb + off_h * stride_lh + offs_m
        tl.store(l_ptrs, lse, mask=offs_m < seqlen_q)


# ----------------------------------------------------------------------------
# Autotuning
# ----------------------------------------------------------------------------


def _configs():
    cfgs = []
    for bm, bn in [(128, 64), (128, 128), (64, 64), (64, 128), (128, 32)]:
        for warps in [4, 8]:
            for stages in [2, 3]:
                cfgs.append(
                    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=warps, num_stages=stages)
                )
    return cfgs


_attn_fwd_autotuned = triton.autotune(
    configs=_configs(),
    key=["SEQLEN_Q", "SEQLEN_K", "BLOCK_D", "IS_CAUSAL", "IS_VARLEN"],
)(_attn_fwd_kernel)


def _interpreting() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


def _default_tiles(head_dim: int):
    # Sane fixed tiles when autotuning is unavailable (interpreter / first call).
    if head_dim >= 256:
        return 64, 32, 8
    if head_dim >= 128:
        return 64, 64, 4
    return 128, 64, 4


# ----------------------------------------------------------------------------
# Python entry points
# ----------------------------------------------------------------------------

SUPPORTED_HEAD_DIMS = (16, 32, 64, 128, 256)


def _check(q, k, v):
    assert q.dtype == k.dtype == v.dtype, "q/k/v dtypes must match"
    assert q.dtype in (torch.float16, torch.bfloat16, torch.float32)
    d = q.shape[-1]
    assert d in SUPPORTED_HEAD_DIMS, f"head_dim {d} not in {SUPPORTED_HEAD_DIMS}"
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "head_dim must be contiguous"


def attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
    autotune: Optional[bool] = None,
):
    """Dense layout: q [B, H_q, N_q, D]; k, v [B, H_kv, N_k, D]."""
    _check(q, k, v)
    b, h_q, n_q, d = q.shape
    _, h_kv, n_k, _ = k.shape
    assert h_q % h_kv == 0
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    o = torch.empty_like(q)
    lse = torch.empty((b, h_q, n_q), dtype=torch.float32, device=q.device) if return_lse else o  # dummy ptr
    dummy = torch.zeros(1, dtype=torch.int32, device=q.device)

    use_autotune = (not _interpreting()) if autotune is None else autotune
    grid = lambda META: (triton.cdiv(n_q, META["BLOCK_M"]), b * h_q)
    args = (
        q, k, v, o, lse, dummy, dummy,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        lse.stride(0) if return_lse else 0, lse.stride(1) if return_lse else 0,
        h_q, h_kv, n_q, n_k, scale,
    )
    if use_autotune:
        _attn_fwd_autotuned[grid](
            *args, IS_CAUSAL=causal, IS_VARLEN=False, WRITE_LSE=return_lse, BLOCK_D=d
        )
    else:
        bm, bn, warps = _default_tiles(d)
        _attn_fwd_kernel[grid](
            *args, IS_CAUSAL=causal, IS_VARLEN=False, WRITE_LSE=return_lse,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_D=d, num_warps=warps,
        )
    return (o, lse) if return_lse else o


def attention_varlen_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
    scale: Optional[float] = None,
    autotune: Optional[bool] = None,
) -> torch.Tensor:
    """Packed layout: q [total_q, H_q, D]; k, v [total_k, H_kv, D]."""
    _check(q, k, v)
    assert q.dim() == 3 and cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    total_q, h_q, d = q.shape
    _, h_kv, _ = k.shape
    assert h_q % h_kv == 0
    nb = cu_seqlens_q.numel() - 1
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    o = torch.empty_like(q)

    use_autotune = (not _interpreting()) if autotune is None else autotune
    grid = lambda META: (triton.cdiv(max_seqlen_q, META["BLOCK_M"]), nb * h_q)
    args = (
        q, k, v, o, o, cu_seqlens_q, cu_seqlens_k,
        0, q.stride(1), q.stride(0), q.stride(2),
        0, k.stride(1), k.stride(0), k.stride(2),
        0, v.stride(1), v.stride(0), v.stride(2),
        0, o.stride(1), o.stride(0), o.stride(2),
        0, 0,
        h_q, h_kv, max_seqlen_q, max_seqlen_k, scale,
    )
    if use_autotune:
        _attn_fwd_autotuned[grid](*args, IS_CAUSAL=causal, IS_VARLEN=True, WRITE_LSE=False, BLOCK_D=d)
    else:
        bm, bn, warps = _default_tiles(d)
        _attn_fwd_kernel[grid](
            *args, IS_CAUSAL=causal, IS_VARLEN=True, WRITE_LSE=False,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_D=d, num_warps=warps,
        )
    return o
