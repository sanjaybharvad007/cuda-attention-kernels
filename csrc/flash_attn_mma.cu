// Tensor-Core fused attention forward (FlashAttention-2 structure, sm_80+).
//
//   grid  = (ceil(N_q / 64), B * H_q), block = 128 threads = 4 warps
//   each warp owns 16 query rows of the 64-row Q tile; K/V stream through
//   shared memory in 64-row tiles, double-buffered with cp.async so the
//   next tile's global loads overlap this tile's MMAs.
//
// Per K/V tile, per warp:
//   S  = Q·Kᵀ        8 x (D/16) mma.m16n8k16, Q fragments resident in registers,
//                     K fragments via ldmatrix (row-major K == "col" B operand)
//   online softmax   entirely in the mma C-fragment layout: the 4 lanes that
//                     share a row reduce with two shfl_xor; exp2 with the
//                     log2(e)·scale folded in
//   P  -> A operand  the fp32 C fragment of S repacks into the b16 A fragment
//                     of the next MMA with no shared-memory round trip
//   O += P·V         (D/8) x 4 mma, V fragments via ldmatrix.trans
//
// Shared memory rows are padded by 8 elements (16 B) so the 8 row addresses
// of every ldmatrix hit 8 distinct bank groups (conflict-free).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "flash_attn.h"
#include "ptx.cuh"

namespace fused_attn {
namespace {

using namespace ptx;

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int NTHREADS = 128;

template <typename T, int HEAD_DIM>
struct Smem {
  static constexpr int LDS = HEAD_DIM + 8;  // padded row stride (elements)
  static constexpr int Q_ELEMS = BLOCK_M * LDS;
  static constexpr int KV_ELEMS = BLOCK_N * LDS;
  static constexpr size_t BYTES = (Q_ELEMS + 4 * KV_ELEMS) * sizeof(T);  // Q + 2xK + 2xV
};

// Issue cp.async for a [rows x HEAD_DIM] tile; rows >= valid_rows are zero-filled.
template <typename T, int HEAD_DIM>
__device__ __forceinline__ void load_tile_async(T* dst, const T* src, int64_t row_stride, int first_row,
                                                int valid_rows, int tid) {
  constexpr int LDS = Smem<T, HEAD_DIM>::LDS;
  constexpr int CHUNKS_PER_ROW = HEAD_DIM / 8;  // 8 b16 = 16 bytes
  constexpr int CHUNKS = BLOCK_N * CHUNKS_PER_ROW;
  static_assert(CHUNKS % NTHREADS == 0, "tile must split evenly across threads");
#pragma unroll
  for (int c = tid; c < CHUNKS; c += NTHREADS) {
    const int r = c / CHUNKS_PER_ROW;
    const int col = (c % CHUNKS_PER_ROW) * 8;
    const bool ok = r < valid_rows;
    const T* s = ok ? src + (int64_t)(first_row + r) * row_stride + col : src;
    cp_async_16(dst + r * LDS + col, s, ok ? 16 : 0);
  }
}

template <typename T, int HEAD_DIM, bool IS_CAUSAL>
__global__ void __launch_bounds__(NTHREADS)
attn_fwd_mma_kernel(const T* __restrict__ Q, const T* __restrict__ K, const T* __restrict__ V, T* __restrict__ O,
                    int64_t sqb, int64_t sqh, int64_t sqm, int64_t skb, int64_t skh, int64_t skn,
                    int64_t svb, int64_t svh, int64_t svn, int64_t sob, int64_t soh, int64_t som,
                    int H_q, int H_kv, int N_q, int N_k, float sm_scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  __trap();  // cp.async / ldmatrix / bf16 mma need Ampere or newer
#else
  using S = Smem<T, HEAD_DIM>;
  constexpr int LDS = S::LDS;
  constexpr int KSTEPS = HEAD_DIM / 16;  // k-steps of Q·Kᵀ
  constexpr int DTILES = HEAD_DIM / 8;   // 8-wide n-tiles of O
  static_assert(HEAD_DIM % 16 == 0 && HEAD_DIM >= 16, "head_dim must be a multiple of 16");

  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* Qs = reinterpret_cast<T*>(smem_raw);
  T* Ks = Qs + S::Q_ELEMS;        // [2][BLOCK_N][LDS]
  T* Vs = Ks + 2 * S::KV_ELEMS;   // [2][BLOCK_N][LDS]

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;   // row within the 16-row warp slab (and +8)
  const int t4 = lane & 3;   // column pair within an 8-wide n-tile

  const int start_m = blockIdx.x;
  const int bh = blockIdx.y;
  const int b = bh / H_q;
  const int h = bh % H_q;
  const int h_kv = h / (H_q / H_kv);
  const int m_base = start_m * BLOCK_M;
  const int causal_offset = N_k - N_q;

  const T* q_base = Q + b * sqb + h * sqh;
  const T* k_base = K + b * skb + h_kv * skh;
  const T* v_base = V + b * svb + h_kv * svh;
  T* o_base = O + b * sob + h * soh;

  int hi = N_k;
  if (IS_CAUSAL) hi = max(0, min(N_k, m_base + BLOCK_M + causal_offset));
  const int n_tiles = (hi + BLOCK_N - 1) / BLOCK_N;

  // ---- prologue: Q tile + first K/V tile ------------------------------------
  load_tile_async<T, HEAD_DIM>(Qs, q_base, sqm, m_base, min(BLOCK_M, N_q - m_base), tid);
  if (n_tiles > 0) {
    load_tile_async<T, HEAD_DIM>(Ks, k_base, skn, 0, min(BLOCK_N, hi), tid);
    load_tile_async<T, HEAD_DIM>(Vs, v_base, svn, 0, min(BLOCK_N, hi), tid);
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  // Q fragments stay in registers for the whole kernel.
  uint32_t qf[KSTEPS][4];
#pragma unroll
  for (int kk = 0; kk < KSTEPS; ++kk)
    ldmatrix_x4(qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3],
                Qs + (warp * 16 + (lane & 15)) * LDS + kk * 16 + (lane >> 4) * 8);

  float o[DTILES][4];
#pragma unroll
  for (int d = 0; d < DTILES; ++d) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.f;
  float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;  // running max, rows g and g+8
  float l0 = 0.f, l1 = 0.f;                       // running sum
  const float qk_scale = sm_scale * 1.4426950408889634f;
  const int row0 = m_base + warp * 16 + g;
  const int row1 = row0 + 8;

  // ---- main loop over K/V tiles ------------------------------------------------
  for (int t = 0; t < n_tiles; ++t) {
    const int buf = t & 1;
    if (t + 1 < n_tiles) {  // prefetch next tile into the other buffer
      const int nxt = (t + 1) * BLOCK_N;
      load_tile_async<T, HEAD_DIM>(Ks + (buf ^ 1) * S::KV_ELEMS, k_base, skn, nxt, min(BLOCK_N, hi - nxt), tid);
      load_tile_async<T, HEAD_DIM>(Vs + (buf ^ 1) * S::KV_ELEMS, v_base, svn, nxt, min(BLOCK_N, hi - nxt), tid);
    }
    cp_async_commit();
    const T* Kt = Ks + buf * S::KV_ELEMS;
    const T* Vt = Vs + buf * S::KV_ELEMS;

    // S = Q Kᵀ : 8 n-tiles of 8 keys, accumulated over KSTEPS
    float s[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < KSTEPS; ++kk) {
#pragma unroll
      for (int np = 0; np < 4; ++np) {  // pairs of n-tiles (16 keys)
        uint32_t r0, r1, r2, r3;
        ldmatrix_x4(r0, r1, r2, r3, Kt + (np * 16 + (lane & 15)) * LDS + kk * 16 + (lane >> 4) * 8);
        const uint32_t b0[2] = {r0, r2};  // keys np*16 + 0..7
        const uint32_t b1[2] = {r1, r3};  // keys np*16 + 8..15
        Mma<T>::m16n8k16(s[2 * np], qf[kk], b0);
        Mma<T>::m16n8k16(s[2 * np + 1], qf[kk], b1);
      }
    }

    // scale + mask (in the C-fragment layout)
    const int col_base = t * BLOCK_N + 2 * t4;
    float mloc0 = -CUDART_INF_F, mloc1 = -CUDART_INF_F;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int col = col_base + n * 8 + (i & 1);
        const int row = (i < 2) ? row0 : row1;
        bool valid = col < hi;
        if (IS_CAUSAL) valid = valid && (col <= row + causal_offset);
        s[n][i] = valid ? s[n][i] * qk_scale : -CUDART_INF_F;
      }
      mloc0 = fmaxf(mloc0, fmaxf(s[n][0], s[n][1]));
      mloc1 = fmaxf(mloc1, fmaxf(s[n][2], s[n][3]));
    }
    mloc0 = quad_max(mloc0);
    mloc1 = quad_max(mloc1);

    // online softmax rescale
    const float mn0 = fmaxf(m0, mloc0), mn1 = fmaxf(m1, mloc1);
    const float ms0 = (mn0 == -CUDART_INF_F) ? 0.f : mn0;
    const float ms1 = (mn1 == -CUDART_INF_F) ? 0.f : mn1;
    const float alpha0 = exp2f(m0 - ms0), alpha1 = exp2f(m1 - ms1);
    float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      s[n][0] = exp2f(s[n][0] - ms0);
      s[n][1] = exp2f(s[n][1] - ms0);
      s[n][2] = exp2f(s[n][2] - ms1);
      s[n][3] = exp2f(s[n][3] - ms1);
      rs0 += s[n][0] + s[n][1];
      rs1 += s[n][2] + s[n][3];
    }
    l0 = l0 * alpha0 + quad_sum(rs0);
    l1 = l1 * alpha1 + quad_sum(rs1);
    m0 = mn0;
    m1 = mn1;
#pragma unroll
    for (int d = 0; d < DTILES; ++d) {
      o[d][0] *= alpha0; o[d][1] *= alpha0;
      o[d][2] *= alpha1; o[d][3] *= alpha1;
    }

    // O += P V : 4 k-steps of 16 keys; P comes straight from the S fragments
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t pa[4] = {
          Mma<T>::pack(s[2 * j][0], s[2 * j][1]),         // row g,   keys 16j + 2t..
          Mma<T>::pack(s[2 * j][2], s[2 * j][3]),         // row g+8
          Mma<T>::pack(s[2 * j + 1][0], s[2 * j + 1][1]), // row g,   keys 16j + 8 + 2t..
          Mma<T>::pack(s[2 * j + 1][2], s[2 * j + 1][3]), // row g+8
      };
#pragma unroll
      for (int dp = 0; dp < DTILES / 2; ++dp) {  // pairs of 8-wide d tiles
        uint32_t v0, v1, v2, v3;
        ldmatrix_x4_trans(v0, v1, v2, v3,
                          Vt + (j * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LDS + dp * 16 + (lane >> 4) * 8);
        const uint32_t vb0[2] = {v0, v1};
        const uint32_t vb1[2] = {v2, v3};
        Mma<T>::m16n8k16(o[2 * dp], pa, vb0);
        Mma<T>::m16n8k16(o[2 * dp + 1], pa, vb1);
      }
    }

    cp_async_wait<0>();
    __syncthreads();  // next tile landed; this buffer is free for the following prefetch
  }

  // ---- epilogue: normalise and store --------------------------------------------
  const float inv0 = (l0 == 0.f) ? 0.f : 1.f / l0;
  const float inv1 = (l1 == 0.f) ? 0.f : 1.f / l1;
#pragma unroll
  for (int d = 0; d < DTILES; ++d) {
    const int col = d * 8 + 2 * t4;
    if (row0 < N_q)
      *reinterpret_cast<uint32_t*>(o_base + (int64_t)row0 * som + col) = Mma<T>::pack(o[d][0] * inv0, o[d][1] * inv0);
    if (row1 < N_q)
      *reinterpret_cast<uint32_t*>(o_base + (int64_t)row1 * som + col) = Mma<T>::pack(o[d][2] * inv1, o[d][3] * inv1);
  }
#endif
}

template <typename T, int HEAD_DIM, bool IS_CAUSAL>
void launch_mma(const AttnParams& p, cudaStream_t stream) {
  auto* kern = attn_fwd_mma_kernel<T, HEAD_DIM, IS_CAUSAL>;
  const size_t smem = Smem<T, HEAD_DIM>::BYTES;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  dim3 grid((p.seqlen_q + BLOCK_M - 1) / BLOCK_M, p.batch * p.num_heads_q);
  kern<<<grid, NTHREADS, smem, stream>>>(
      static_cast<const T*>(p.q), static_cast<const T*>(p.k), static_cast<const T*>(p.v), static_cast<T*>(p.o),
      p.q_batch_stride, p.q_head_stride, p.q_row_stride, p.k_batch_stride, p.k_head_stride, p.k_row_stride,
      p.v_batch_stride, p.v_head_stride, p.v_row_stride, p.o_batch_stride, p.o_head_stride, p.o_row_stride,
      p.num_heads_q, p.num_heads_kv, p.seqlen_q, p.seqlen_k, p.scale);
}

template <typename T>
void dispatch_mma(const AttnParams& p, cudaStream_t s) {
  switch (p.head_dim) {
    case 64:  p.causal ? launch_mma<T, 64, true>(p, s)  : launch_mma<T, 64, false>(p, s);  break;
    case 128: p.causal ? launch_mma<T, 128, true>(p, s) : launch_mma<T, 128, false>(p, s); break;
    default: break;
  }
}

}  // namespace

void run_attention_fwd_mma(const AttnParams& p, cudaStream_t stream) {
  switch (p.dtype) {
    case DType::kFloat16:  dispatch_mma<__half>(p, stream); break;
    case DType::kBFloat16: dispatch_mma<__nv_bfloat16>(p, stream); break;
    default: break;  // fp32 is not a Tensor-Core input type here; caller routes to the SIMT kernel
  }
}

size_t attention_fwd_mma_smem_bytes(int head_dim, int elem_size) {
  return (BLOCK_M + 4 * BLOCK_N) * (head_dim + 8) * elem_size;
}

}  // namespace fused_attn
