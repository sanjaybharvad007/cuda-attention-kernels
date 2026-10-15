# cuda-attention-kernels

Fused multi-head attention for LLM inference, implemented three ways — a
**CUDA C++ Tensor-Core kernel** (`mma.sync` + `ldmatrix` + `cp.async`,
FlashAttention-2 structure), a **CUDA C++ SIMT kernel** (every byte explicit,
for profiling), and a **Triton kernel** (autotuned) — plus **split-K
flash-decoding** for long-context decode and an **FP8 KV-cache** path, all
benchmarked against PyTorch SDPA and FlashAttention.

```python
import fused_attn
out = fused_attn.attention(q, k, v, causal=True)              # q [B,Hq,N,D]  k,v [B,Hkv,N,D]
out = fused_attn.attention_varlen(q, k, v, cu_q, cu_k, max_q, max_k, causal=True)   # packed
```

## What's inside

| Feature | CUDA Tensor Core (`csrc/flash_attn_mma.cu`) | CUDA SIMT (`csrc/flash_attn.cu`) | Triton (`fused_attn/`) |
|---|---|---|---|
| Tiling + online softmax (never materialises N×N) | ✓ | ✓ | ✓ |
| Tensor Cores | ✓ `mma.sync.m16n8k16`, P kept in registers | – (fp32 FMA by design) | ✓ `tl.dot` |
| Shared-memory K/V pipeline | ✓ `cp.async` double buffer, `ldmatrix(.trans)` | ✓ 16 B vector loads, double buffer | ✓ `num_stages` |
| Bank-conflict-free smem layout | ✓ (padded rows) | – | n/a |
| Causal (bottom-right aligned, loop stops at diagonal) | ✓ | ✓ | ✓ |
| GQA / MQA | ✓ | ✓ | ✓ |
| Variable-length packed batches | – | – | ✓ `cu_seqlens` |
| Split-K flash-decoding (+ combine kernel) | – | – | ✓ `triton_splitk.py` |
| FP8 (E4M3) K/V with per-block scales | – | – | ✓ `fp8.py` (sm_89+) |
| Head dims | 64, 128 | 64, 128 | 16 – 256 |
| dtypes | fp16, bf16 | fp16, bf16, fp32 | fp16, bf16, fp32 |
| Arch | sm_80+ | any | any |
| Autotuned tiles | – | – | ✓ |

The public op `fused_attn::attention` is a `torch.library` custom op with a
fake kernel, so it traces cleanly through `torch.compile(fullgraph=True)`.
Backend selection: CUDA Tensor-Core → CUDA SIMT → Triton → reference, with
decode-shaped calls (≤16 queries, ≥2K keys) routed to split-K; force with
`FUSED_ATTN_BACKEND=cuda_mma|cuda|triton|splitk|reference`.

The Tensor-Core kernel is documented step by step in
[`docs/tensor_core_kernel.md`](docs/tensor_core_kernel.md).

## Install

```bash
pip install setuptools wheel
pip install -e . --no-build-isolation                        # builds csrc/ if nvcc is found
FUSED_ATTN_SKIP_CUDA=1 pip install -e . --no-build-isolation   # Triton-only
```

## Test

Every kernel is checked against an unfused PyTorch reference (itself verified
against `F.scaled_dot_product_attention`) across batch/head/GQA-ratio/length/
head-dim/causal/dtype combinations, including the awkward ones: ragged tails,
`n_q != n_k`, single-token decode, non-contiguous HF `[B,N,H,D]` layouts, and
sequences of length 0 in a packed batch.

```bash
pytest -q                      # GPU
TRITON_INTERPRET=1 pytest -q   # CPU-only (Triton numpy interpreter) — what CI runs
```

## Benchmark

```bash
python benchmarks/bench.py --seqlens 512 1024 2048 4096 8192 16384 \
    --heads 32 --kv-heads 8 --head-dim 128 --causal --dtype bf16 --out results.csv
python benchmarks/plot.py results.csv
python benchmarks/bench_decode.py --ctx 4096 16384 65536        # single-pass vs split-K vs FP8-KV
```

Reports latency, achieved TFLOP/s and max-abs error vs. the fp32 reference for
`fused_attn/triton`, `fused_attn/cuda`, `torch.sdpa` and `flash_attn` (if
installed). Fill in your numbers here:

| seqlen | fused_attn/cuda_mma | fused_attn/triton | torch.sdpa (flash) | flash_attn 2 |
|---|---|---|---|---|
| 2048 | | | | |
| 8192 | | | | |

| decode ctx | single-pass | split-K | FP8-KV |
|---|---|---|---|
| 16384 | | | |
| 65536 | | | |

## How the kernel works

1. Each program owns `BLOCK_M` query rows of one (batch, head) and loads its Q tile once.
2. It streams K/V in `BLOCK_N` tiles. For each tile: `S = Q·Kᵀ` (fp32 accumulate),
   mask, then the **online softmax** update
   `m_new = max(m, rowmax(S)); α = 2^(m − m_new); l = α·l + Σ 2^(S − m_new); O = α·O + P·V`.
3. After the last tile `O /= l`. Nothing of size N×N is ever written to HBM, so the
   kernel is bound by the N·D reads of K/V rather than the N² score matrix.
4. Causal blocks stop their K loop at the diagonal (≈2× less work), and the Triton
   kernel reads its per-sequence bounds from `cu_seqlens` so no padding is computed.

## Profiling

```bash
bash scripts/profile_ncu.sh 4096 cuda_mma     # writes ncu_cuda_mma_4096.ncu-rep + metric summary
```
Things worth looking at: `smsp__sass_average_data_bytes_per_sector_mem_global` (coalescing),
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared` (bank conflicts), achieved occupancy,
and the Tensor-Core pipe utilisation for the Triton kernel.

## Layout

```
fused_attn/   reference.py  triton_fwd.py  triton_splitk.py  fp8.py  ops.py (torch.library op + dispatch)
csrc/         flash_attn_mma.cu (Tensor Cores)  flash_attn.cu (SIMT)  ptx.cuh  flash_attn.h  bindings.cpp
tests/        correctness, varlen, split-K, FP8, Tensor-Core kernel (GPU), torch.compile / opcheck
benchmarks/   bench.py  bench_decode.py  plot.py
docs/         tensor_core_kernel.md      scripts/  profile_ncu.sh
```

## Status

Triton kernels (dense, varlen, split-K) are verified on CPU via the Triton
interpreter in CI and on GPU by the GPU workflow. The two CUDA C++ kernels
are built by `setup.py` and covered by `tests/test_splitk_fp8.py -k cuda_mma`
and `tests/test_correctness.py::test_all_backends_agree`, which run only on a
GPU runner.
