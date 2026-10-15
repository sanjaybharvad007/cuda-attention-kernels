"""Public API and backend dispatch.

``fused_attn.attention`` is registered as a ``torch.library`` custom op so it
composes with ``torch.compile`` (graph breaks are avoided via the fake/meta
implementation) and shows up by name in profiler traces.

Backend priority: CUDA C++ extension (if built, dtype/head_dim supported)
-> Triton -> PyTorch reference.  Override with ``FUSED_ATTN_BACKEND``.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch

from .reference import attention_reference, attention_varlen_reference

try:  # compiled CUDA extension (optional)
    from . import _C as _cuda_ext  # type: ignore
except Exception:  # pragma: no cover - only present after `pip install -e .` with nvcc
    _cuda_ext = None

try:
    from .triton_fwd import attention_triton, attention_varlen_triton

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False

BACKENDS = ("cuda", "triton", "reference")


def available_backends() -> list[str]:
    out = []
    if _cuda_ext is not None:
        out.append("cuda")
    if _HAS_TRITON:
        out.append("triton")
    out.append("reference")
    return out


def _pick_backend(q: torch.Tensor) -> str:
    forced = os.environ.get("FUSED_ATTN_BACKEND")
    if forced:
        assert forced in BACKENDS, f"unknown backend {forced}"
        return forced
    on_gpu = q.is_cuda
    interp = os.environ.get("TRITON_INTERPRET", "0") == "1"
    if on_gpu and _cuda_ext is not None and q.shape[-1] in (64, 128):
        return "cuda"
    if _HAS_TRITON and (on_gpu or interp):
        return "triton"
    return "reference"


@torch.library.custom_op("fused_attn::attention", mutates_args=())
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Fused scaled-dot-product attention.

    q: [B, H_q, N_q, D]; k, v: [B, H_kv, N_k, D] with H_q % H_kv == 0.
    Returns [B, H_q, N_q, D] in q's dtype.
    """
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    backend = _pick_backend(q)
    if backend == "cuda":
        return _cuda_ext.attention_fwd(q, k, v, causal, scale)
    if backend == "triton":
        return attention_triton(q, k, v, causal=causal, scale=scale)
    return attention_reference(q, k, v, causal=causal, scale=scale)


@attention.register_fake
def _attention_fake(q, k, v, causal=False, scale=None):
    return q.new_empty(q.shape)


def attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Packed/ragged attention. q: [total_q, H_q, D]; k, v: [total_k, H_kv, D]."""
    backend = _pick_backend(q)
    if backend == "triton":
        return attention_varlen_triton(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=causal, scale=scale
        )
    return attention_varlen_reference(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=causal, scale=scale)


def pack_varlen(tensors: list[torch.Tensor]):
    """[ [N_i, H, D], ... ] -> packed [sum N_i, H, D], cu_seqlens int32, max_len."""
    lens = [t.shape[0] for t in tensors]
    cu = torch.zeros(len(lens) + 1, dtype=torch.int32, device=tensors[0].device)
    cu[1:] = torch.cumsum(torch.tensor(lens, device=tensors[0].device), 0)
    return torch.cat(tensors, 0), cu, max(lens)
