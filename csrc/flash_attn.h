#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace fused_attn {

enum class DType { kFloat16, kBFloat16, kFloat32 };

struct AttnParams {
  const void* q;
  const void* k;
  const void* v;
  void* o;
  int64_t q_batch_stride, q_head_stride, q_row_stride;
  int64_t k_batch_stride, k_head_stride, k_row_stride;
  int64_t v_batch_stride, v_head_stride, v_row_stride;
  int64_t o_batch_stride, o_head_stride, o_row_stride;
  int batch, num_heads_q, num_heads_kv, seqlen_q, seqlen_k, head_dim;
  float scale;
  bool causal;
  DType dtype;
};

void run_attention_fwd(const AttnParams& p, cudaStream_t stream);

}  // namespace fused_attn
