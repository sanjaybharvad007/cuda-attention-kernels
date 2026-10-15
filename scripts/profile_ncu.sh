#!/usr/bin/env bash
# Nsight Compute profile of the attention kernels at one shape.
#   bash scripts/profile_ncu.sh [seqlen] [backend]      (backend: cuda_mma | cuda | triton | splitk)
# Produces ncu_<backend>_<seqlen>.ncu-rep (open in the Nsight Compute GUI) and a text summary.
set -euo pipefail
SEQ="${1:-4096}"
BACKEND="${2:-cuda_mma}"
OUT="ncu_${BACKEND}_${SEQ}"
FUSED_ATTN_BACKEND="$BACKEND" ncu \
  --set full \
  --import-source yes \
  --kernel-name regex:"attn_fwd|_attn_fwd|splitk" \
  --launch-count 3 --launch-skip 5 \
  -o "$OUT" \
  python benchmarks/bench.py --seqlens "$SEQ" --heads 32 --kv-heads 8 --head-dim 128 --causal --dtype bf16 --out /tmp/bench.csv
ncu --import "$OUT.ncu-rep" --page details \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem \
  | tee "$OUT.txt"
