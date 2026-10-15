"""Split-K (flash-decoding) and FP8-KV paths."""
import os

import pytest
import torch

import fused_attn
from fused_attn.reference import attention_reference

CUDA = torch.cuda.is_available()
INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
DEV = "cuda" if CUDA else "cpu"
needs_triton = pytest.mark.skipif(not (CUDA or INTERP), reason="needs a GPU or TRITON_INTERPRET=1")


@needs_triton
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_splits", [1, 3, 8])
@pytest.mark.parametrize("b,h_q,h_kv,n_q,n_k,d", [(1, 2, 1, 1, 1000, 64), (2, 4, 2, 4, 777, 64), (1, 2, 2, 16, 2048, 128)])
def test_splitk_matches_reference(b, h_q, h_kv, n_q, n_k, d, num_splits, causal):
    torch.manual_seed(0)
    dtype = torch.float16 if CUDA else torch.float32
    q = torch.randn(b, h_q, n_q, d, device=DEV, dtype=dtype)
    k = torch.randn(b, h_kv, n_k, d, device=DEV, dtype=dtype)
    v = torch.randn(b, h_kv, n_k, d, device=DEV, dtype=dtype)
    out = fused_attn.attention_splitk_triton(q, k, v, causal=causal, num_splits=num_splits)
    ref = attention_reference(q, k, v, causal=causal)
    tol = 2e-2 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(out.float(), ref.float(), atol=tol, rtol=tol)


@needs_triton
def test_splitk_is_chosen_for_decode_shapes():
    from fused_attn.ops import _pick_backend

    q = torch.randn(1, 8, 1, 64, device=DEV)
    k = torch.randn(1, 2, 4096, 64, device=DEV)
    assert _pick_backend(q, k) == "splitk"
    assert _pick_backend(torch.randn(1, 8, 512, 64, device=DEV), k) != "splitk"
    out = fused_attn.attention(q, k, k, causal=True)
    torch.testing.assert_close(out, attention_reference(q, k, k, causal=True), atol=1e-3, rtol=1e-3)


def test_choose_num_splits_heuristic():
    from fused_attn.triton_splitk import choose_num_splits

    assert choose_num_splits(1, 8, 32768, 1, num_sms=108) == 27      # 2*108 // 8
    assert choose_num_splits(64, 32, 32768, 1, num_sms=108) == 1     # already saturated
    assert choose_num_splits(1, 1, 512, 1, num_sms=108) == 2         # capped by >=256 keys per split
    assert choose_num_splits(1, 1, 100000, 128, num_sms=108) == 1    # prefill: no split


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch build lacks float8")
def test_fp8_quantization_roundtrip_error_bounded():
    torch.manual_seed(0)
    k = torch.randn(2, 2, 200, 64) * 3
    k8, ks = fused_attn.quantize_kv_fp8(k)
    assert k8.dtype == torch.float8_e4m3fn and ks.shape == (2, 2, 4)
    kd = fused_attn.dequantize_kv_fp8(k8, ks)
    rel = ((kd - k).abs() / k.abs().clamp_min(1e-3)).median()
    assert rel < 0.08                                   # 3 mantissa bits -> ~6% typical relative error
    assert (kd - k).abs().max() / k.abs().max() < 0.07  # per-block scaling keeps the absolute error tight


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch build lacks float8")
def test_fp8_kv_attention_accuracy_vs_fp32():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8, 64)
    k, v = torch.randn(2, 2, 300, 64), torch.randn(2, 2, 300, 64)
    k8, ks = fused_attn.quantize_kv_fp8(k)
    v8, vs = fused_attn.quantize_kv_fp8(v)
    ref = attention_reference(q, k, v, causal=True)
    out = attention_reference(q, fused_attn.dequantize_kv_fp8(k8, ks), fused_attn.dequantize_kv_fp8(v8, vs), causal=True)
    assert (out - ref).abs().max() < 5e-2


@pytest.mark.skipif(not CUDA or torch.cuda.get_device_capability()[0] < 8, reason="FP8 kernel needs sm_89+ GPU")
def test_fp8_kernel_matches_dequantized_reference():
    torch.manual_seed(0)
    q = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.float16)
    k, v = torch.randn(2, 2, 1024, 64, device="cuda"), torch.randn(2, 2, 1024, 64, device="cuda")
    k8, ks = fused_attn.quantize_kv_fp8(k)
    v8, vs = fused_attn.quantize_kv_fp8(v)
    out = fused_attn.attention_fp8_triton(q, k8, ks, v8, vs, causal=True)
    ref = attention_reference(q.float(), fused_attn.dequantize_kv_fp8(k8, ks), fused_attn.dequantize_kv_fp8(v8, vs), causal=True)
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not CUDA, reason="Tensor-Core CUDA kernel")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("b,h_q,h_kv,n_q,n_k", [(1, 1, 1, 64, 64), (2, 8, 2, 333, 333), (1, 4, 1, 100, 500), (1, 2, 2, 500, 100)])
def test_cuda_mma_kernel_matches_reference(b, h_q, h_kv, n_q, n_k, causal, head_dim, dtype):
    from fused_attn import ops

    if ops._cuda_ext is None or not ops._has_mma():
        pytest.skip("CUDA extension not built or GPU < sm_80")
    torch.manual_seed(0)
    q = torch.randn(b, h_q, n_q, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(b, h_kv, n_k, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(b, h_kv, n_k, head_dim, device="cuda", dtype=dtype)
    out = ops._cuda_ext.attention_fwd_mma(q, k, v, causal, head_dim ** -0.5)
    ref = attention_reference(q, k, v, causal=causal)
    tol = 2e-2 if dtype == torch.float16 else 5e-2
    torch.testing.assert_close(out.float(), ref.float(), atol=tol, rtol=tol)
