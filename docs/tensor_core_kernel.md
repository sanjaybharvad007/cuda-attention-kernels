# The Tensor-Core kernel (`csrc/flash_attn_mma.cu`)

This note explains the kernel at the level an interviewer will ask about.

## Work decomposition

```
grid  = (ceil(N_q / 64), B * H_q)        one CTA per 64 query rows of one head
block = 128 threads = 4 warps            warp w owns query rows 16w .. 16w+15
```

Every warp keeps its 16 Q rows as `mma` A-fragments in registers for the whole
kernel (`D/16 * 4` registers). K and V stream through shared memory in
64-key tiles, double-buffered: while tile *t* is being multiplied, tile *t+1*
is in flight via `cp.async` (16-byte async copies issued by all 128 threads,
zero-filled for rows past the sequence end so no branch is needed later).

## One tile, one warp

1. **S = Q·Kᵀ** — 8 n-tiles × (D/16) k-steps of `mma.sync.m16n8k16`.
   K is row-major `[key][d]` in shared memory, which is exactly the "col"
   layout the B operand wants, so a plain `ldmatrix.x4` yields fragments for
   two 8-key tiles at once (`{r0,r2}` and `{r1,r3}`).
2. **Mask + scale** happen in the C-fragment layout: lane `l` holds columns
   `2(l%4), 2(l%4)+1` of rows `l/4` and `l/4+8`, so the causal test is a
   compare on those indices. The softmax scale is pre-multiplied by log2(e)
   so the exponentials below are `exp2f`.
3. **Online softmax** — the row max is a max over the lane's 16 values then
   two `shfl_xor` (lanes 1 and 2 apart share the row). `alpha = exp2(m_old −
   m_new)` rescales both the running sum and the O accumulator.
4. **P → A operand** — the fp32 C fragment of two adjacent 8-wide S tiles has
   the same element-to-lane mapping as a 16-wide b16 A fragment, so P is
   packed with `cvt` into registers and fed straight into the next MMA. No
   shared-memory round trip for P: this is the key FlashAttention-2 trick.
5. **O += P·V** — V is `[key][d]` but the B operand needs keys along k with
   d along n, so `ldmatrix.x4.trans` transposes on the way from shared
   memory. 4 k-steps (64 keys) × (D/8) n-tiles.
6. `cp.async.wait_group 0; __syncthreads()` — next tile has landed and
   everyone is done reading this buffer.

## Memory layout details

* Shared rows are padded to `D + 8` elements. 8 consecutive rows of an
  `ldmatrix` then start 144 B (D=64) or 272 B (D=128) apart, i.e. 4 banks
  apart, so the 8 × 16 B reads touch all 32 banks exactly once: **no bank
  conflicts** (check `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` in ncu).
* Shared memory per CTA: `(64 + 4·64)·(D+8)·2 B` = 46 KB (D=64) / 87 KB
  (D=128); the launcher opts in with `cudaFuncSetAttribute`.
* Registers per thread ≈ 16/32 (Q) + 32 (S/P) + 32/64 (O) + addressing;
  `__launch_bounds__(128)` lets the compiler use up to 255.

## What to measure

`scripts/profile_ncu.sh 4096 cuda_mma` reports Tensor-pipe utilisation
(`sm__pipe_tensor_cycles_active`), DRAM throughput, shared-memory bank
conflicts and the occupancy limiter. A healthy run shows the tensor pipe
>60 % active for long sequences, zero bank conflicts, and occupancy limited by
shared memory (not registers) at D=128.

## Known limitations / next steps

* Forward only, head_dim 64 and 128, no dropout, no sliding window.
* Hopper `wgmma` + TMA (warp-group MMA with register-free B operands) would be
  the next step; this kernel is the Ampere-style `mma.sync` design.
* The kernel has been written and reviewed against the PTX ISA fragment
  layouts but must be validated on an sm_80+ GPU with
  `pytest tests/test_splitk_fp8.py -k cuda_mma` before use.
