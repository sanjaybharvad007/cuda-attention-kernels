"""Decode-shaped benchmark: 1-16 query tokens against a long KV cache.

Compares the single-pass kernel (B*H programs) with split-K flash-decoding
and the FP8-KV kernel; this is the regime where context length, not FLOPs,
decides latency.

    python benchmarks/bench_decode.py --ctx 4096 16384 65536 --batch 1 --heads 32 --kv-heads 8
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fused_attn  # noqa: E402
from fused_attn.reference import attention_reference  # noqa: E402


def bench(fn, iters=50):
    import triton

    return triton.testing.do_bench(fn, warmup=10, rep=iters)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", type=int, nargs="+", default=[4096, 16384, 65536])
    p.add_argument("--q-len", type=int, default=1)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    a = p.parse_args()
    if not torch.cuda.is_available():
        print("needs a GPU")
        return
    dt = torch.float16
    print(f"{'ctx':>7s} {'single-pass':>12s} {'split-K':>10s} {'splits':>6s} {'fp8-kv':>10s} {'GB/s (splitK)':>14s}")
    for n in a.ctx:
        q = torch.randn(a.batch, a.heads, a.q_len, a.head_dim, device="cuda", dtype=dt)
        k = torch.randn(a.batch, a.kv_heads, n, a.head_dim, device="cuda", dtype=dt)
        v = torch.randn(a.batch, a.kv_heads, n, a.head_dim, device="cuda", dtype=dt)
        ref = attention_reference(q, k, v, causal=True)
        from fused_attn.triton_fwd import attention_triton
        from fused_attn.triton_splitk import attention_splitk_triton, choose_num_splits

        sms = torch.cuda.get_device_properties(0).multi_processor_count
        splits = choose_num_splits(a.batch, a.heads, n, a.q_len, sms)
        t_single = bench(lambda: attention_triton(q, k, v, causal=True))
        o_sk = attention_splitk_triton(q, k, v, causal=True)
        t_sk = bench(lambda: attention_splitk_triton(q, k, v, causal=True))
        assert (o_sk.float() - ref.float()).abs().max() < 5e-2
        t_fp8 = float("nan")
        if torch.cuda.get_device_capability()[0] >= 8 and hasattr(torch, "float8_e4m3fn"):
            k8, ks = fused_attn.quantize_kv_fp8(k)
            v8, vs = fused_attn.quantize_kv_fp8(v)
            try:
                t_fp8 = bench(lambda: fused_attn.attention_fp8_triton(q, k8, ks, v8, vs, causal=True))
            except Exception as e:  # no FP8 conversion on this arch
                print(f"  fp8 kernel unavailable: {type(e).__name__}")
        kv_bytes = 2 * k.numel() * k.element_size()
        print(f"{n:7d} {t_single:9.3f} ms {t_sk:7.3f} ms {splits:6d} {t_fp8:7.3f} ms {kv_bytes / (t_sk * 1e-3) / 1e9:14.0f}")


if __name__ == "__main__":
    main()
