"""Latency / throughput benchmark: fused_attn (CUDA, Triton) vs PyTorch SDPA
(flash / mem-efficient / math backends) vs flash-attn (if installed).

Example:
    python benchmarks/bench.py --seqlens 512 1024 2048 4096 8192 16384 \
        --head-dim 128 --heads 32 --kv-heads 8 --causal --dtype bf16 --out results.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fused_attn  # noqa: E402
from fused_attn.reference import attention_reference  # noqa: E402

try:
    from flash_attn import flash_attn_func  # type: ignore
except Exception:
    flash_attn_func = None


@dataclass
class Result:
    backend: str
    seqlen: int
    batch: int
    heads: int
    kv_heads: int
    head_dim: int
    causal: bool
    dtype: str
    ms: float
    tflops: float
    max_abs_err: float


def attn_flops(b, h, n, d, causal):
    f = 4.0 * b * h * n * n * d  # QK^T and PV, 2 flops per MAC
    return f / 2 if causal else f


def bench_fn(fn, warmup=10, iters=50):
    if torch.cuda.is_available():
        import triton

        return triton.testing.do_bench(fn, warmup=warmup, rep=iters)
    import time

    for _ in range(2):
        fn()
    t = time.perf_counter()
    for _ in range(3):
        fn()
    return (time.perf_counter() - t) / 3 * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seqlens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    p.add_argument("--out", default="bench_results.csv")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    if dev == "cpu":
        print("WARNING: no GPU found; running tiny CPU timings for smoke-test purposes only")
        dtype = torch.float32

    results: list[Result] = []
    for n in args.seqlens:
        b, h, hk, d = args.batch, args.heads, args.kv_heads, args.head_dim
        q = torch.randn(b, h, n, d, device=dev, dtype=dtype)
        k = torch.randn(b, hk, n, d, device=dev, dtype=dtype)
        v = torch.randn(b, hk, n, d, device=dev, dtype=dtype)
        ref = attention_reference(q, k, v, causal=args.causal) if n <= 8192 else None

        cands = {}
        for be in fused_attn.available_backends():
            if be == "reference":
                continue
            cands[f"fused_attn/{be}"] = (be, lambda q=q, k=k, v=v: fused_attn.attention(q, k, v, causal=args.causal))
        k_e = k.repeat_interleave(h // hk, dim=1)
        v_e = v.repeat_interleave(h // hk, dim=1)
        cands["torch.sdpa"] = (None, lambda: F.scaled_dot_product_attention(q, k_e, v_e, is_causal=args.causal))
        if flash_attn_func is not None and dtype != torch.float32:
            qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))
            cands["flash_attn"] = (None, lambda: flash_attn_func(qt, kt, vt, causal=args.causal).transpose(1, 2))

        for name, (be, fn) in cands.items():
            if be:
                os.environ["FUSED_ATTN_BACKEND"] = be
            try:
                out = fn()
                ms = bench_fn(fn)
            except Exception as e:  # unsupported shape/dtype for this backend
                print(f"  {name:>20s} n={n}: skipped ({type(e).__name__}: {e})")
                continue
            finally:
                os.environ.pop("FUSED_ATTN_BACKEND", None)
            err = (out.float() - ref.float()).abs().max().item() if ref is not None else float("nan")
            tflops = attn_flops(b, h, n, d, args.causal) / (ms * 1e-3) / 1e12
            r = Result(name, n, b, h, hk, d, args.causal, args.dtype, ms, tflops, err)
            results.append(r)
            print(f"  {name:>20s} n={n:6d}: {ms:8.3f} ms  {tflops:6.1f} TFLOP/s  max_err={err:.2e}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    print(f"\nwrote {args.out}")

    # Markdown summary
    print("\n| seqlen | " + " | ".join(sorted({r.backend for r in results})) + " |")
    backends = sorted({r.backend for r in results})
    print("|---|" + "---|" * len(backends))
    for n in args.seqlens:
        row = [f"{r.ms:.3f} ms" for be in backends for r in results if r.seqlen == n and r.backend == be]
        print(f"| {n} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
