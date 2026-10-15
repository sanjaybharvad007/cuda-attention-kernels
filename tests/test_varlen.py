"""Variable-length (packed / ragged) attention."""
import os

import pytest
import torch

import fused_attn
from fused_attn.reference import attention_varlen_reference

CUDA = torch.cuda.is_available()
INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
DEVICE = "cuda" if CUDA else "cpu"


@pytest.mark.skipif(not (CUDA or INTERP), reason="needs a GPU or TRITON_INTERPRET=1")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("lens", [[5, 33, 70], [1, 1, 1], [128], [64, 0, 17]])
@pytest.mark.parametrize("h_q,h_kv", [(4, 4), (8, 2)])
def test_varlen_matches_reference(lens, causal, h_q, h_kv):
    torch.manual_seed(0)
    d = 64
    qs = [torch.randn(n, h_q, d, device=DEVICE) for n in lens]
    ks = [torch.randn(n, h_kv, d, device=DEVICE) for n in lens]
    vs = [torch.randn(n, h_kv, d, device=DEVICE) for n in lens]
    q, cu_q, max_q = fused_attn.pack_varlen(qs)
    k, cu_k, max_k = fused_attn.pack_varlen(ks)
    v, _, _ = fused_attn.pack_varlen(vs)
    out = fused_attn.attention_varlen(q, k, v, cu_q, cu_k, max_q, max_k, causal=causal)
    ref = attention_varlen_reference(q, k, v, cu_q, cu_k, causal=causal)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not (CUDA or INTERP), reason="needs a GPU or TRITON_INTERPRET=1")
def test_varlen_cross_attention_lengths():
    """Different q and k lengths per sequence (e.g. chunked prefill against a KV cache)."""
    torch.manual_seed(0)
    lq, lk = [8, 16], [40, 16]
    q, cu_q, max_q = fused_attn.pack_varlen([torch.randn(n, 4, 64, device=DEVICE) for n in lq])
    k, cu_k, max_k = fused_attn.pack_varlen([torch.randn(n, 4, 64, device=DEVICE) for n in lk])
    v, _, _ = fused_attn.pack_varlen([torch.randn(n, 4, 64, device=DEVICE) for n in lk])
    out = fused_attn.attention_varlen(q, k, v, cu_q, cu_k, max_q, max_k, causal=True)
    ref = attention_varlen_reference(q, k, v, cu_q, cu_k, causal=True)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
