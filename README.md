# cuda-attention-kernels

Fused multi-head attention for LLM inference, implemented twice — once in
**CUDA C++** (explicit shared-memory tiling, for profiling with Nsight) and once
in **Triton** (Tensor-Core `tl.dot`, autotuned) — and benchmarked against
PyTorch SDPA and FlashAttention.

```python
import fused_attn
out = fused_attn.attention(q, k, v, causal=True)              # q [B,Hq,N,D]  k,v [B,Hkv,N,D]
out = fused_attn.attention_varlen(q, k, v, cu_q, cu_k, max_q, max_k, causal=True)   # packed
```

## What's inside

| Feature | CUDA C++ (`csrc/`) | Triton (`fused_attn/triton_fwd.py`) |
|---|---|---|
| Tiling + online softmax (never materialises N×N) | ✓ | ✓ |
| Shared-memory K/V staging, double buffered | ✓ (16 B vectorised loads) | ✓ (`num_stages` pipelining) |
| Tensor Cores | – (SIMT fp32 FMA, by design) | ✓ (`tl.dot` → `mma.sync`/`wgmma`) |
| Causal masking (bottom-right aligned, loop stops at diagonal) | ✓ | ✓ |
| GQA / MQA head sharing | ✓ | ✓ |
| Variable-length packed batches (`cu_seqlens`) | – | ✓ |
| Head dims | 64, 128 | 16 – 256 |
| dtypes | fp16, bf16, fp32 | fp16, bf16, fp32 |
| exp2 with folded log2(e) scale | ✓ | ✓ |
| Autotuned tile sizes / warps / stages | – | ✓ |
| Returns logsumexp (for backward / split-K merge) | – | ✓ |

The public op `fused_attn::attention` is a `torch.library` custom op with a
fake kernel, so it traces cleanly through `torch.compile(fullgraph=True)`.
Backend selection: CUDA ext → Triton → PyTorch reference, or force with
`FUSED_ATTN_BACKEND=cuda|triton|reference`.

## Install

```bash
pip install -e .                      # builds csrc/ if nvcc is found
FUSED_ATTN_SKIP_CUDA=1 pip install -e .   # Triton-only
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
```

Reports latency, achieved TFLOP/s and max-abs error vs. the fp32 reference for
`fused_attn/triton`, `fused_attn/cuda`, `torch.sdpa` and `flash_attn` (if
installed). Fill in your numbers here:

| seqlen | fused_attn/triton | torch.sdpa (flash) | flash_attn 2 |
|---|---|---|---|
| 2048 | | | |
| 8192 | | | |

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
ncu --set full --kernel-name regex:attn_fwd python benchmarks/bench.py --seqlens 4096
```
Things worth looking at: `smsp__sass_average_data_bytes_per_sector_mem_global` (coalescing),
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared` (bank conflicts), achieved occupancy,
and the Tensor-Core pipe utilisation for the Triton kernel.

## Layout

```
fused_attn/   reference.py  triton_fwd.py  ops.py (torch.library op + dispatch)
csrc/         flash_attn.cu  flash_attn.h  bindings.cpp
tests/        correctness, varlen, torch.compile / opcheck
benchmarks/   bench.py  plot.py
```
