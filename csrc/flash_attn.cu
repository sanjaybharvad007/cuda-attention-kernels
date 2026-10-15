// Fused multi-head attention forward kernel (CUDA C++).
//
// A FlashAttention-style SIMT implementation written for readability and to
// make the memory/compute structure explicit:
//
//   * grid  = (ceil(N_q / BLOCK_M), B * H_q)
//   * block = BLOCK_M * THREADS_PER_ROW threads.  Each query row is owned by
//     THREADS_PER_ROW (=2) adjacent lanes that split the head dimension; the
//     partial Q.K dot products are combined with one __shfl_xor_sync.
//   * K and V tiles (BLOCK_N x HEAD_DIM) are staged through shared memory with
//     16-byte vectorised, coalesced loads and double-buffered across iterations.
//   * Online softmax in fp32 (running max + running sum), exp2 with log2(e)
//     folded into the scale.
//   * Causal: bottom-right aligned; the K loop stops at the diagonal.
//   * GQA/MQA: query head h reads KV head h / (H_q / H_kv).
//
// The Triton kernel in this repo is the Tensor-Core path (tl.dot -> mma);
// this kernel is the "every byte explicit" reference for profiling with
// Nsight Compute (bank conflicts, coalescing, occupancy).
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "flash_attn.h"

namespace fused_attn {

namespace {

constexpr int WARP_SIZE = 32;
constexpr float LOG2E = 1.4426950408889634f;

// ---- dtype helpers --------------------------------------------------------
template <typename T> struct Cvt;
template <> struct Cvt<__half> {
  __device__ static float to_f32(__half x) { return __half2float(x); }
  __device__ static __half from_f32(float x) { return __float2half(x); }
};
template <> struct Cvt<__nv_bfloat16> {
  __device__ static float to_f32(__nv_bfloat16 x) { return __bfloat162float(x); }
  __device__ static __nv_bfloat16 from_f32(float x) { return __float2bfloat16(x); }
};
template <> struct Cvt<float> {
  __device__ static float to_f32(float x) { return x; }
  __device__ static float from_f32(float x) { return x; }
};

// Number of elements of T in a 16-byte vector.
template <typename T> constexpr int vec_elems() { return 16 / sizeof(T); }

// Cooperative, coalesced copy of a [rows x HEAD_DIM] tile from global to smem.
// Rows >= valid_rows are zero-filled.  Every thread issues 16B loads.
template <typename T, int ROWS, int HEAD_DIM, int NTHREADS>
__device__ __forceinline__ void load_tile(T* __restrict__ smem,
                                          const T* __restrict__ gmem,
                                          int64_t row_stride, int valid_rows,
                                          int tid) {
  constexpr int VE = vec_elems<T>();
  constexpr int VEC_PER_ROW = HEAD_DIM / VE;
  constexpr int TOTAL_VEC = ROWS * VEC_PER_ROW;
  static_assert(TOTAL_VEC % NTHREADS == 0, "tile must divide evenly");
#pragma unroll
  for (int i = tid; i < TOTAL_VEC; i += NTHREADS) {
    const int r = i / VEC_PER_ROW;
    const int c = (i % VEC_PER_ROW) * VE;
    uint4 val = make_uint4(0, 0, 0, 0);
    if (r < valid_rows) {
      val = *reinterpret_cast<const uint4*>(gmem + (int64_t)r * row_stride + c);
    }
    *reinterpret_cast<uint4*>(smem + r * HEAD_DIM + c) = val;
  }
}

template <typename T, int HEAD_DIM, int BLOCK_M, int BLOCK_N, bool IS_CAUSAL>
__global__ void __launch_bounds__(BLOCK_M * 2)
attn_fwd_kernel(const T* __restrict__ Q, const T* __restrict__ K,
                const T* __restrict__ V, T* __restrict__ O,
                int64_t sqb, int64_t sqh, int64_t sqm,
                int64_t skb, int64_t skh, int64_t skn,
                int64_t svb, int64_t svh, int64_t svn,
                int64_t sob, int64_t soh, int64_t som,
                int H_q, int H_kv, int N_q, int N_k, float sm_scale) {
  constexpr int THREADS_PER_ROW = 2;
  constexpr int NTHREADS = BLOCK_M * THREADS_PER_ROW;
  constexpr int D_HALF = HEAD_DIM / THREADS_PER_ROW;

  // Double-buffered K and V tiles.
  __shared__ __align__(16) T Ks[2][BLOCK_N * HEAD_DIM];
  __shared__ __align__(16) T Vs[2][BLOCK_N * HEAD_DIM];

  const int tid = threadIdx.x;
  const int row_local = tid / THREADS_PER_ROW;   // which query row in the block
  const int half = tid % THREADS_PER_ROW;        // which half of head_dim
  const int start_m = blockIdx.x;
  const int bh = blockIdx.y;
  const int b = bh / H_q;
  const int h = bh % H_q;
  const int h_kv = h / (H_q / H_kv);

  const int row = start_m * BLOCK_M + row_local;
  const bool row_valid = row < N_q;
  const int causal_offset = N_k - N_q;

  const T* q_ptr = Q + b * sqb + h * sqh + (int64_t)(row_valid ? row : 0) * sqm + half * D_HALF;
  const T* k_base = K + b * skb + h_kv * skh;
  const T* v_base = V + b * svb + h_kv * svh;

  // Q fragment in registers, pre-scaled by sm_scale * log2(e).
  float q[D_HALF];
  const float qk_scale = sm_scale * LOG2E;
#pragma unroll
  for (int i = 0; i < D_HALF; ++i) q[i] = Cvt<T>::to_f32(q_ptr[i]) * qk_scale;

  float acc[D_HALF];
#pragma unroll
  for (int i = 0; i < D_HALF; ++i) acc[i] = 0.f;
  float m_i = -CUDART_INF_F;
  float l_i = 0.f;

  int hi = N_k;
  if (IS_CAUSAL) {
    hi = min(N_k, (start_m + 1) * BLOCK_M + causal_offset);
    hi = max(hi, 0);
  }
  const int n_tiles = (hi + BLOCK_N - 1) / BLOCK_N;
  if (n_tiles > 0) {
    load_tile<T, BLOCK_N, HEAD_DIM, NTHREADS>(Ks[0], k_base, skn, min(BLOCK_N, hi), tid);
    load_tile<T, BLOCK_N, HEAD_DIM, NTHREADS>(Vs[0], v_base, svn, min(BLOCK_N, hi), tid);
  }
  __syncthreads();

  for (int t = 0; t < n_tiles; ++t) {
    const int buf = t & 1;
    const int start_n = t * BLOCK_N;
    // Prefetch next tile into the other buffer (overlaps with compute below).
    if (t + 1 < n_tiles) {
      const int nxt = start_n + BLOCK_N;
      load_tile<T, BLOCK_N, HEAD_DIM, NTHREADS>(Ks[buf ^ 1], k_base + (int64_t)nxt * skn, skn, min(BLOCK_N, hi - nxt), tid);
      load_tile<T, BLOCK_N, HEAD_DIM, NTHREADS>(Vs[buf ^ 1], v_base + (int64_t)nxt * svn, svn, min(BLOCK_N, hi - nxt), tid);
    }

    // S[j] = q . k_j  for the BLOCK_N keys in this tile (partial over our half).
    float s[BLOCK_N];
    float tile_max = -CUDART_INF_F;
#pragma unroll
    for (int j = 0; j < BLOCK_N; ++j) {
      const T* kr = &Ks[buf][j * HEAD_DIM + half * D_HALF];
      float dot = 0.f;
#pragma unroll
      for (int i = 0; i < D_HALF; ++i) dot = fmaf(q[i], Cvt<T>::to_f32(kr[i]), dot);
      // combine the two halves of the row (lanes 2r and 2r+1 are adjacent)
      dot += __shfl_xor_sync(0xffffffff, dot, 1);
      const int col = start_n + j;
      bool valid = col < hi;
      if (IS_CAUSAL) valid = valid && (col <= row + causal_offset);
      s[j] = valid ? dot : -CUDART_INF_F;
      tile_max = fmaxf(tile_max, s[j]);
    }

    // Online softmax rescale.
    const float m_new = fmaxf(m_i, tile_max);
    const float m_safe = (m_new == -CUDART_INF_F) ? 0.f : m_new;
    const float alpha = exp2f(m_i - m_safe);
    l_i *= alpha;
#pragma unroll
    for (int i = 0; i < D_HALF; ++i) acc[i] *= alpha;

    // P = exp2(S - m);  O += P V
#pragma unroll
    for (int j = 0; j < BLOCK_N; ++j) {
      const float p = exp2f(s[j] - m_safe);
      l_i += p;
      const T* vr = &Vs[buf][j * HEAD_DIM + half * D_HALF];
#pragma unroll
      for (int i = 0; i < D_HALF; ++i) acc[i] = fmaf(p, Cvt<T>::to_f32(vr[i]), acc[i]);
    }
    m_i = m_new;
    __syncthreads();  // tile buf is free to be overwritten by the next prefetch
  }

  if (!row_valid) return;
  const float inv_l = (l_i == 0.f) ? 0.f : 1.f / l_i;
  T* o_ptr = O + b * sob + h * soh + (int64_t)row * som + half * D_HALF;
#pragma unroll
  for (int i = 0; i < D_HALF; ++i) o_ptr[i] = Cvt<T>::from_f32(acc[i] * inv_l);
}

template <typename T, int HEAD_DIM, bool IS_CAUSAL>
void launch(const AttnParams& p, cudaStream_t stream) {
  constexpr int BLOCK_M = 64;
  constexpr int BLOCK_N = 32;
  dim3 grid((p.seqlen_q + BLOCK_M - 1) / BLOCK_M, p.batch * p.num_heads_q);
  dim3 block(BLOCK_M * 2);
  attn_fwd_kernel<T, HEAD_DIM, BLOCK_M, BLOCK_N, IS_CAUSAL><<<grid, block, 0, stream>>>(
      static_cast<const T*>(p.q), static_cast<const T*>(p.k), static_cast<const T*>(p.v),
      static_cast<T*>(p.o),
      p.q_batch_stride, p.q_head_stride, p.q_row_stride,
      p.k_batch_stride, p.k_head_stride, p.k_row_stride,
      p.v_batch_stride, p.v_head_stride, p.v_row_stride,
      p.o_batch_stride, p.o_head_stride, p.o_row_stride,
      p.num_heads_q, p.num_heads_kv, p.seqlen_q, p.seqlen_k, p.scale);
}

template <typename T, int HEAD_DIM>
void dispatch_causal(const AttnParams& p, cudaStream_t s) {
  if (p.causal) launch<T, HEAD_DIM, true>(p, s);
  else          launch<T, HEAD_DIM, false>(p, s);
}

template <typename T>
void dispatch_headdim(const AttnParams& p, cudaStream_t s) {
  switch (p.head_dim) {
    case 64:  dispatch_causal<T, 64>(p, s);  break;
    case 128: dispatch_causal<T, 128>(p, s); break;
    default:  break;  // validated by the caller
  }
}

}  // namespace

void run_attention_fwd(const AttnParams& p, cudaStream_t stream) {
  switch (p.dtype) {
    case DType::kFloat16:  dispatch_headdim<__half>(p, stream); break;
    case DType::kBFloat16: dispatch_headdim<__nv_bfloat16>(p, stream); break;
    case DType::kFloat32:  dispatch_headdim<float>(p, stream); break;
  }
}

}  // namespace fused_attn
