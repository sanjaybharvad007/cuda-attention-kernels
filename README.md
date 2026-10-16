# cuda-attention-kernels

**Fused multi-head attention for LLM inference: a Tensor-Core CUDA C++ kernel, a SIMT CUDA C++ kernel, autotuned Triton kernels, split-K flash-decoding and an FP8 KV-cache path, behind one `torch.library` op.**

[![CI](https://github.com/sanjaybharvad007/cuda-attention-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/sanjaybharvad007/cuda-attention-kernels/actions/workflows/ci.yml)
[![GPU tests](https://github.com/sanjaybharvad007/cuda-attention-kernels/actions/workflows/gpu.yml/badge.svg)](https://github.com/sanjaybharvad007/cuda-attention-kernels/actions/workflows/gpu.yml)
![CUDA](https://img.shields.io/badge/CUDA-12.x%20%7C%20sm__80%2B-76B900)
![Triton](https://img.shields.io/badge/Triton-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

```python
import fused_attn

out = fused_attn.attention(q, k, v, causal=True)                               # q [B,Hq,N,D]  k,v [B,Hkv,N,D]
out = fused_attn.attention_varlen(q, k, v, cu_q, cu_k, max_q, max_k, causal=True)   # packed, no padding
```

The same attention kernel is written three times at three levels of the stack, so the trade-offs are measurable rather than theoretical: hand-scheduled Tensor-Core PTX (`mma.sync` / `ldmatrix` / `cp.async`), explicit SIMT CUDA where every byte and FMA is visible to the profiler, and Triton where the compiler owns the schedule. All three share the FlashAttention-2 algorithm, the same test-suite and the same benchmark harness, and they are compared against PyTorch SDPA and FlashAttention-2.

---

## Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Feature matrix](#feature-matrix)
- [Installation](#installation)
- [Usage](#usage)
- [Design](#design)
- [Performance](#performance)
- [Testing](#testing)
- [Profiling](#profiling)
- [Repository layout](#repository-layout)
- [Roadmap](#roadmap)
- [References](#references)

---

## Highlights

- **Tensor-Core kernel in CUDA C++** (`csrc/flash_attn_mma.cu`): `mma.sync.m16n8k16` fragments, `ldmatrix` / `ldmatrix.trans` fragment loads, `cp.async` double-buffered K/V tiles, padded bank-conflict-free shared memory, probabilities kept in registers between the QK and PV MMAs.
- **SIMT kernel in CUDA C++** (`csrc/flash_attn.cu`): fp32 FMA, 16-byte vector loads, double-buffered tiles; a readable baseline for profiling and for architectures without Tensor Cores.
- **Autotuned Triton kernels** (`fused_attn/`): dense, variable-length (`cu_seqlens`) and split-K flash-decoding with a log-sum-exp combine kernel; run on CPU under `TRITON_INTERPRET=1`, which is what CI uses.
- **FP8 (E4M3) KV cache** with per-64-key block scales, dequantised inside the kernel (sm_89+).
- **Online softmax in base 2**: `exp2` with `log2(e)` folded into the softmax scale, so the inner loop has no multiply before the exponential.
- **One public op**: `fused_attn::attention` is a `torch.library` custom op with a fake kernel, so it traces through `torch.compile(fullgraph=True)`; dispatch picks the fastest available backend per call.
- **121 tests** against an unfused PyTorch reference, including ragged tails, `n_q != n_k`, single-token decode, GQA/MQA ratios, non-contiguous `[B,N,H,D]` layouts and zero-length sequences in a packed batch.

## Architecture

```
                         fused_attn.attention(q, k, v, causal, scale)
                                          |
                               torch.library custom op
                        (fake kernel -> torch.compile fullgraph)
                                          |
          +-----------------+-------------+--------------+-----------------+
          |                 |                            |                 |
   cuda_mma (sm_80+)    cuda (SIMT)                 triton            reference
   mma.sync m16n8k16    fp32 FMA                    tl.dot, autotune   unfused torch
   ldmatrix(.trans)     16 B vector loads           dense | varlen     (tests only)
   cp.async 2-stage     double buffer               | splitk | fp8-kv
   padded smem                                      TRITON_INTERPRET=1
          |                 |                            |
          +-----------------+-------------+--------------+
                                          |
                     FlashAttention-2 tiling + online softmax (exp2)
                     causal: bottom-right aligned, K loop stops at the diagonal
                     GQA/MQA: h_kv = h_q // (H_q / H_kv), no K/V replication
```

Dispatch order is `cuda_mma -> cuda -> triton -> reference`, with decode-shaped calls (at most 16 queries against at least 2048 keys) routed to split-K. `FUSED_ATTN_BACKEND=cuda_mma|cuda|triton|splitk|reference` forces a backend.

## Feature matrix

| Capability | CUDA Tensor Core<br>`csrc/flash_attn_mma.cu` | CUDA SIMT<br>`csrc/flash_attn.cu` | Triton<br>`fused_attn/` |
|---|:---:|:---:|:---:|
| Tiling + online softmax (no N x N materialisation) | yes | yes | yes |
| Tensor Cores | `mma.sync.m16n8k16`, P in registers | fp32 FMA by design | `tl.dot` |
| Shared-memory K/V pipeline | `cp.async` double buffer, `ldmatrix(.trans)` | 16 B vector loads, double buffer | `num_stages` |
| Bank-conflict-free shared memory | padded rows | – | compiler-managed |
| Causal masking | bottom-right aligned, early exit | same | same |
| GQA / MQA | yes | yes | yes |
| Variable-length packed batches | – | – | `cu_seqlens` |
| Split-K flash-decoding + LSE combine | – | – | `triton_splitk.py` |
| FP8 E4M3 K/V, per-block scales | – | – | `fp8.py` (sm_89+) |
| Head dimensions | 64, 128 | 64, 128 | 16 – 256 |
| Data types | fp16, bf16 | fp16, bf16, fp32 | fp16, bf16, fp32 |
| Architectures | sm_80+ | any | any |
| Autotuned tile configurations | – | – | yes |

## Installation

```bash
git clone https://github.com/sanjaybharvad007/cuda-attention-kernels
cd cuda-attention-kernels
pip install setuptools wheel
pip install -e . --no-build-isolation                           # builds csrc/ when nvcc is on PATH
FUSED_ATTN_SKIP_CUDA=1 pip install -e . --no-build-isolation    # Triton-only install
```

Requirements: PyTorch 2.4+, Triton 3.x, and for the CUDA C++ kernels a CUDA 12.x toolkit. The Tensor-Core kernel targets `sm_80` and newer (`TORCH_CUDA_ARCH_LIST` defaults to `8.0;8.6;8.9;9.0`).

## Usage

```python
import torch, fused_attn

B, Hq, Hkv, N, D = 2, 32, 8, 4096, 128
q = torch.randn(B, Hq, N, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, Hkv, N, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)

out = fused_attn.attention(q, k, v, causal=True)              # GQA handled inside the kernel

# packed variable-length batch (no padding tokens are computed)
cu_q = torch.tensor([0, 1000, 1700, 4096], dtype=torch.int32, device="cuda")
out = fused_attn.attention_varlen(q_packed, k_packed, v_packed, cu_q, cu_q, 2396, 2396, causal=True)

# torch.compile sees a single opaque op with a fake kernel
fn = torch.compile(lambda q, k, v: fused_attn.attention(q, k, v, causal=True), fullgraph=True)

# FP8 KV cache for long-context decode
from fused_attn.fp8 import attention_fp8_triton, quantize_kv_fp8
k8, k_scale = quantize_kv_fp8(k)                               # E4M3 values + per-64-key block scales
v8, v_scale = quantize_kv_fp8(v)
out = attention_fp8_triton(q[:, :, -1:], k8, k_scale, v8, v_scale)

# split-K flash-decoding explicitly (dispatch picks it automatically for decode shapes)
from fused_attn.triton_splitk import attention_splitk_triton
out = attention_splitk_triton(q[:, :, -1:], k, v, num_splits=16)
```

## Design

### The algorithm every kernel shares

1. A program owns `BLOCK_M` query rows of one (batch, head) and loads its Q tile once.
2. K/V stream through in `BLOCK_N` tiles. For each tile: `S = Q K^T` (fp32 accumulate), mask, then the online-softmax update
   `m' = max(m, rowmax(S))`, `alpha = 2^(m - m')`, `l = alpha * l + sum 2^(S - m')`, `O = alpha * O + P V`.
3. After the last tile `O /= l`. Nothing of size N x N is written to HBM, so the kernel is bound by the `N * D` reads of K and V rather than the `N^2` score matrix.
4. Causal blocks stop the K loop at the diagonal (about half the work); the varlen kernel reads per-sequence bounds from `cu_seqlens`, so padding is never computed.

### Tensor-Core kernel

The step-by-step walkthrough is in [`docs/tensor_core_kernel.md`](docs/tensor_core_kernel.md). In short: 4 warps per 64-row Q tile, each warp owning 16 rows; K and V tiles are fetched with `cp.async` into a two-stage shared-memory ring with 8-element row padding so `ldmatrix` is conflict-free; `S = Q K^T` and `O += P V` are `mma.sync.m16n8k16` sequences whose fp32 accumulators are rescaled in registers by `alpha`; P is converted to bf16/fp16 fragments in registers and fed straight into the second MMA without touching shared memory.

### Split-K flash-decoding

A single decode query against a 64k-token context would otherwise occupy one SM. The split-K kernel partitions the keys across `S` programs, each producing a partial `(O_s, m_s, l_s)`; a second kernel combines them with `O = sum_s 2^(m_s - m) O_s / sum_s 2^(m_s - m) l_s`. Dispatch chooses `S` from the context length and the SM count.

### FP8 KV cache

Keys and values are stored as E4M3 with one fp32 scale per 64-key block per head. The Triton kernel loads the 8-bit tile, multiplies by the block scale and proceeds in fp16/bf16; this halves KV-cache bandwidth, which is the binding constraint at decode.

## Performance

```bash
python benchmarks/bench.py --seqlens 512 1024 2048 4096 8192 16384 \
    --heads 32 --kv-heads 8 --head-dim 128 --causal --dtype bf16 --out results.csv
python benchmarks/plot.py results.csv
python benchmarks/bench_decode.py --ctx 4096 16384 65536            # single-pass vs split-K vs FP8-KV
```

The harness reports latency, achieved TFLOP/s and maximum absolute error against an fp32 reference for `fused_attn/cuda_mma`, `fused_attn/cuda`, `fused_attn/triton`, `torch.sdpa` and `flash_attn` when installed. Numbers below are the template the harness fills in; measure on your own hardware and paste them here.

**Prefill, causal, B=2, Hq=32, Hkv=8, D=128, bf16**

| seqlen | cuda_mma | triton | torch.sdpa (flash) | flash_attn 2 |
|---:|---:|---:|---:|---:|
| 2048 | | | | |
| 8192 | | | | |
| 16384 | | | | |

**Decode, 1 query, Hq=32, Hkv=8, D=128**

| context | single-pass | split-K | FP8-KV | speed-up |
|---:|---:|---:|---:|---:|
| 16384 | | | | |
| 65536 | | | | |

## Testing

```bash
pytest -q                        # on a GPU: all backends, torch.compile / opcheck, FP8, split-K, Tensor-Core kernel
TRITON_INTERPRET=1 pytest -q     # CPU-only: Triton kernels through the numpy interpreter (what CI runs)
```

Every kernel is compared with an unfused PyTorch reference that is itself checked against `F.scaled_dot_product_attention`. The matrix covers batch, head count, GQA ratio, sequence length (including ragged tails and `n_q != n_k`), head dimension, causal flag and dtype. `tests/test_correctness.py::test_all_backends_agree` asserts the CUDA kernels and the Triton kernels agree with each other on a GPU runner (`.github/workflows/gpu.yml`, run manually).

## Profiling

```bash
bash scripts/profile_ncu.sh 4096 cuda_mma      # writes ncu_cuda_mma_4096.ncu-rep and a metric summary
```

Metrics worth reading: `smsp__sass_average_data_bytes_per_sector_mem_global` (coalescing), `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` (shared-memory conflicts), achieved occupancy, and Tensor-pipe utilisation (`sm__pipe_tensor_cycles_active`).

## Repository layout

```
fused_attn/        reference.py        unfused PyTorch reference
                   triton_fwd.py       dense + varlen FlashAttention-2 kernel, autotuned
                   triton_splitk.py    split-K flash-decoding + LSE combine kernel
                   fp8.py              E4M3 KV quantisation and FP8-KV attention kernel
                   ops.py              torch.library op, fake kernel, backend dispatch
csrc/              flash_attn_mma.cu   Tensor-Core kernel (mma.sync, ldmatrix, cp.async)
                   flash_attn.cu       SIMT kernel
                   ptx.cuh             PTX wrappers: mma, ldmatrix, cp.async
                   flash_attn.h        bindings.cpp
tests/             test_correctness.py  test_varlen.py  test_splitk_fp8.py  test_torch_compile.py
benchmarks/        bench.py  bench_decode.py  plot.py
docs/              tensor_core_kernel.md
scripts/           profile_ncu.sh
.github/workflows  ci.yml (CPU, Triton interpreter)  gpu.yml (manual, GPU runner)
```

## Roadmap

- Backward pass (recompute-based, FlashAttention-2 style) for the Triton kernel
- Hopper path: `wgmma` + TMA for the Tensor-Core kernel
- Paged (block-table) K/V for the decode kernels
- Sliding-window and ALiBi masks

## References

- Dao, T. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* 2023.
- Dao, T. et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.
- Milakov, M., Gimelshein, N. *Online normalizer calculation for softmax.* 2018.
- Tillet, P. et al. *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations.* MAPL 2019.
- NVIDIA. *PTX ISA: `mma`, `ldmatrix`, `cp.async`.*

## License

MIT. See [LICENSE](LICENSE).
