"""Pure-PyTorch reference attention.

Deliberately simple and unfused: this is the oracle every kernel is checked
against.  Supports causal masking (bottom-right aligned, like FlashAttention),
grouped-query attention (H_q is a multiple of H_kv) and variable-length
batches packed with cumulative sequence offsets.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


def _expand_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, H_kv, N, D] -> [B, H_kv * n_rep, N, D] (GQA head sharing)."""
    if n_rep == 1:
        return x
    b, h, n, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, n, d).reshape(b, h * n_rep, n, d)


def causal_mask(seqlen_q: int, seqlen_k: int, device=None) -> torch.Tensor:
    """Boolean [seqlen_q, seqlen_k] mask, True where attention is allowed.

    Bottom-right aligned: the last query row sees every key.  When
    seqlen_q == seqlen_k this is the ordinary lower-triangular mask.
    """
    offset = seqlen_k - seqlen_q
    i = torch.arange(seqlen_q, device=device)[:, None]
    j = torch.arange(seqlen_k, device=device)[None, :]
    return j <= i + offset


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: Optional[float] = None,
    upcast: bool = True,
) -> torch.Tensor:
    """q: [B, H_q, N_q, D]; k, v: [B, H_kv, N_k, D]. Returns [B, H_q, N_q, D]."""
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    b, h_q, n_q, d = q.shape
    _, h_kv, n_k, _ = k.shape
    assert h_q % h_kv == 0, "H_q must be a multiple of H_kv"
    out_dtype = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    k = _expand_kv(k, h_q // h_kv)
    v = _expand_kv(v, h_q // h_kv)
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        mask = causal_mask(n_q, n_k, device=q.device)
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    # Rows with no valid key (possible only when n_q > n_k) produce NaN; zero them.
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.matmul(probs, v).to(out_dtype)


def attention_varlen_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Packed ("ragged") layout.

    q: [total_q, H_q, D]; k, v: [total_k, H_kv, D].
    cu_seqlens_*: int32 [B + 1] cumulative offsets (first element 0).
    """
    assert q.dim() == 3
    out = torch.empty_like(q)
    nb = cu_seqlens_q.numel() - 1
    for i in range(nb):
        qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        ks, ke = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
        if qe == qs:
            continue
        o = attention_reference(
            q[qs:qe].transpose(0, 1)[None],
            k[ks:ke].transpose(0, 1)[None],
            v[ks:ke].transpose(0, 1)[None],
            causal=causal,
            scale=scale,
        )
        out[qs:qe] = o[0].transpose(0, 1)
    return out
