"""Correctness of every backend against the unfused PyTorch reference.

Runs on CPU-only CI via the Triton interpreter (TRITON_INTERPRET=1) and on GPU
runners for the real kernels.  The reference itself is cross-checked against
torch.nn.functional.scaled_dot_product_attention.
"""
import os

import pytest
import torch
import torch.nn.functional as F

import fused_attn
from fused_attn.reference import attention_reference

CUDA = torch.cuda.is_available()
INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
DEVICE = "cuda" if CUDA else "cpu"


def _tol(dtype):
    return {torch.float32: (1e-4, 1e-4), torch.float16: (2e-2, 2e-2), torch.bfloat16: (5e-2, 5e-2)}[dtype]


def _rand(*shape, dtype, device=DEVICE):
    return torch.randn(*shape, dtype=dtype, device=device)


# --------------------------------------------------------------------------
# Reference vs. PyTorch SDPA (sanity-checks the oracle)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("n", [1, 17, 128])
def test_reference_matches_sdpa(n, causal):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, n, 64) for _ in range(3))
    ref = attention_reference(q, k, v, causal=causal)
    exp = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.testing.assert_close(ref, exp, atol=1e-5, rtol=1e-5)


def test_reference_gqa_equals_expanded():
    torch.manual_seed(0)
    q = torch.randn(1, 8, 32, 64)
    k, v = torch.randn(1, 2, 32, 64), torch.randn(1, 2, 32, 64)
    ref = attention_reference(q, k, v, causal=True)
    k_e = k.repeat_interleave(4, dim=1)
    v_e = v.repeat_interleave(4, dim=1)
    exp = F.scaled_dot_product_attention(q, k_e, v_e, is_causal=True)
    torch.testing.assert_close(ref, exp, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------
# Fused kernels vs. reference
# --------------------------------------------------------------------------
@pytest.mark.skipif(not (CUDA or INTERP), reason="needs a GPU or TRITON_INTERPRET=1")
@pytest.mark.parametrize("dtype", [torch.float32] + ([torch.float16, torch.bfloat16] if CUDA else []))
@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "b,h_q,h_kv,n_q,n_k",
    [
        (1, 1, 1, 64, 64),
        (2, 4, 4, 128, 128),
        (1, 8, 2, 100, 100),    # GQA 4:1, ragged tails
        (1, 4, 1, 37, 90),      # MQA, n_q < n_k (bottom-right causal)
        (1, 2, 2, 90, 40),      # n_q > n_k: leading rows have no keys
        (3, 2, 2, 1, 257),      # single-token decode step
    ],
)
def test_fused_vs_reference(b, h_q, h_kv, n_q, n_k, causal, head_dim, dtype):
    if INTERP and not CUDA and (n_q * n_k > 128 * 128 or head_dim == 128) and dtype == torch.float32:
        # keep the numpy interpreter run short
        if n_q * n_k > 128 * 128:
            pytest.skip("too slow in interpreter")
    torch.manual_seed(0)
    q = _rand(b, h_q, n_q, head_dim, dtype=dtype)
    k = _rand(b, h_kv, n_k, head_dim, dtype=dtype)
    v = _rand(b, h_kv, n_k, head_dim, dtype=dtype)
    out = fused_attn.attention(q, k, v, causal=causal)
    ref = attention_reference(q, k, v, causal=causal)
    assert out.dtype == q.dtype and out.shape == q.shape
    assert not torch.isnan(out).any()
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol)


@pytest.mark.skipif(not (CUDA or INTERP), reason="needs a GPU or TRITON_INTERPRET=1")
def test_custom_scale():
    torch.manual_seed(0)
    q, k, v = (_rand(1, 2, 64, 64, dtype=torch.float32) for _ in range(3))
    out = fused_attn.attention(q, k, v, causal=True, scale=0.05)
    ref = attention_reference(q, k, v, causal=True, scale=0.05)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not (CUDA or INTERP), reason="needs a GPU or TRITON_INTERPRET=1")
def test_noncontiguous_heads_layout():
    """[B, N, H, D] storage (the usual HF layout) viewed as [B, H, N, D]."""
    torch.manual_seed(0)
    x = torch.randn(2, 64, 3, 4, 64, device=DEVICE)  # B, N, (q,k,v), H, D
    q, k, v = (x[:, :, i].transpose(1, 2) for i in range(3))
    assert not q.is_contiguous()
    out = fused_attn.attention(q, k, v, causal=True)
    ref = attention_reference(q, k, v, causal=True)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not CUDA, reason="CUDA-only backend comparison")
def test_all_backends_agree():
    torch.manual_seed(0)
    q = _rand(2, 8, 512, 64, dtype=torch.float16)
    k = _rand(2, 2, 512, 64, dtype=torch.float16)
    v = _rand(2, 2, 512, 64, dtype=torch.float16)
    ref = attention_reference(q, k, v, causal=True)
    for backend in fused_attn.available_backends():
        os.environ["FUSED_ATTN_BACKEND"] = backend
        try:
            out = fused_attn.attention(q, k, v, causal=True)
        finally:
            del os.environ["FUSED_ATTN_BACKEND"]
        torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2, msg=backend)
