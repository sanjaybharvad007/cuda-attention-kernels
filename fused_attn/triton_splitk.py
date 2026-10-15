"""Split-K attention ("flash-decoding") for long-context decode.

During decode every request contributes one (or a few) query rows, so the
standard kernel launches only B·H programs — far too few to fill a GPU when
the KV cache is long.  Split-K partitions the key/value sequence into
``num_splits`` chunks, runs the online-softmax loop on every chunk in
parallel (grid = B·H·num_splits), writes the *un-normalised* partial output
together with its running (max, sum) statistics, and a second tiny kernel
merges the partials:

    m   = max_i m_i
    l   = Σ_i exp2(m_i - m) · l_i
    O   = Σ_i exp2(m_i - m) · O_i / l

This is exact (same math as the online softmax, just associative), so the
result matches the single-pass kernel to rounding.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_splitk_kernel(
    Q, K, V, O_PART, M_PART, L_PART,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_os, stride_om, stride_od,   # O_PART [B, H, S, N_q, D]
    stride_mb, stride_mh, stride_ms,                          # M/L_PART [B, H, S, N_q]
    H_Q, H_KV, SEQLEN_Q, SEQLEN_K, NUM_SPLITS, SPLIT_LEN,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    split = tl.program_id(2)
    off_b = off_bh // H_Q
    off_h = off_bh % H_Q
    off_hkv = off_h // (H_Q // H_KV)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_hkv * stride_kh
    v_base = V + off_b * stride_vb + off_hkv * stride_vh

    q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                mask=offs_m[:, None] < SEQLEN_Q, other=0.0)

    causal_offset = SEQLEN_K - SEQLEN_Q
    lo = split * SPLIT_LEN
    hi = tl.minimum(SEQLEN_K, lo + SPLIT_LEN)
    if IS_CAUSAL:
        hi = tl.minimum(hi, (start_m + 1) * BLOCK_M + causal_offset)
    hi = tl.maximum(hi, lo)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    for start_n in range(lo, hi, BLOCK_N):
        cols = start_n + offs_n
        kv_mask = cols[:, None] < hi
        k = tl.load(k_base + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd, mask=kv_mask, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        valid = cols[None, :] < hi
        if IS_CAUSAL:
            valid = valid & (cols[None, :] <= offs_m[:, None] + causal_offset)
        qk = tl.where(valid, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd, mask=kv_mask, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # un-normalised partial + statistics (log2 domain)
    o_ptrs = (O_PART + off_b * stride_ob + off_h * stride_oh + split * stride_os
              + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < SEQLEN_Q)
    stat_off = off_b * stride_mb + off_h * stride_mh + split * stride_ms + offs_m
    tl.store(M_PART + stat_off, m_i, mask=offs_m < SEQLEN_Q)
    tl.store(L_PART + stat_off, l_i, mask=offs_m < SEQLEN_Q)


@triton.jit
def _splitk_combine_kernel(
    O_PART, M_PART, L_PART, O,
    stride_ob, stride_oh, stride_os, stride_om, stride_od,
    stride_mb, stride_mh, stride_ms,
    stride_fb, stride_fh, stride_fm, stride_fd,
    H_Q, SEQLEN_Q, NUM_SPLITS,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr,
):
    row = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H_Q
    off_h = off_bh % H_Q
    offs_s = tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)
    smask = offs_s < NUM_SPLITS
    stat = off_b * stride_mb + off_h * stride_mh + offs_s * stride_ms + row
    m = tl.load(M_PART + stat, mask=smask, other=float("-inf"))
    l = tl.load(L_PART + stat, mask=smask, other=0.0)
    m_max = tl.max(m, 0)
    m_safe = tl.where(m_max == float("-inf"), 0.0, m_max)
    w = tl.where(smask, tl.math.exp2(m - m_safe), 0.0)          # [BLOCK_S]
    l_tot = tl.sum(w * l, 0)
    o_ptrs = (O_PART + off_b * stride_ob + off_h * stride_oh + offs_s[:, None] * stride_os
              + row * stride_om + offs_d[None, :] * stride_od)
    o_part = tl.load(o_ptrs, mask=smask[:, None], other=0.0)      # [BLOCK_S, BLOCK_D]
    out = tl.sum(o_part * w[:, None], 0) / tl.where(l_tot == 0.0, 1.0, l_tot)
    tl.store(O + off_b * stride_fb + off_h * stride_fh + row * stride_fm + offs_d * stride_fd,
             out.to(O.dtype.element_ty))


def choose_num_splits(batch: int, heads: int, seqlen_k: int, seqlen_q: int, num_sms: int = 108) -> int:
    """Enough splits to put ~2 programs on every SM, each covering >= 256 keys."""
    if seqlen_q > 64:
        return 1  # prefill-sized queries already fill the GPU
    programs = batch * heads
    want = max(1, (2 * num_sms) // max(programs, 1))
    cap = max(1, seqlen_k // 256)
    return int(min(want, cap, 64))


def attention_splitk_triton(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    causal: bool = False, scale: Optional[float] = None, num_splits: Optional[int] = None,
) -> torch.Tensor:
    b, h_q, n_q, d = q.shape
    _, h_kv, n_k, _ = k.shape
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    if num_splits is None:
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count if q.is_cuda else 8
        num_splits = choose_num_splits(b, h_q, n_k, n_q, sms)
    interp = os.environ.get("TRITON_INTERPRET", "0") == "1"
    block_m = 16 if n_q <= 16 else 64
    block_n = 64
    split_len = triton.cdiv(triton.cdiv(n_k, num_splits), block_n) * block_n   # splits aligned to tiles
    num_splits = triton.cdiv(n_k, split_len)

    o_part = torch.empty((b, h_q, num_splits, n_q, d), dtype=torch.float32, device=q.device)
    m_part = torch.empty((b, h_q, num_splits, n_q), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    grid = (triton.cdiv(n_q, block_m), b * h_q, num_splits)
    _attn_splitk_kernel[grid](
        q, k, v, o_part, m_part, l_part,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o_part.stride(0), o_part.stride(1), o_part.stride(2), o_part.stride(3), o_part.stride(4),
        m_part.stride(0), m_part.stride(1), m_part.stride(2),
        h_q, h_kv, n_q, n_k, num_splits, split_len, scale,
        IS_CAUSAL=causal, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=d,
        num_warps=1 if interp else 4,
    )
    o = torch.empty_like(q)
    _splitk_combine_kernel[(n_q, b * h_q)](
        o_part, m_part, l_part, o,
        o_part.stride(0), o_part.stride(1), o_part.stride(2), o_part.stride(3), o_part.stride(4),
        m_part.stride(0), m_part.stride(1), m_part.stride(2),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        h_q, n_q, num_splits,
        BLOCK_D=d, BLOCK_S=max(2, triton.next_power_of_2(num_splits)),
        num_warps=1 if interp else 4,
    )
    return o
