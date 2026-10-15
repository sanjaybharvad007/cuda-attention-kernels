// Thin wrappers over the PTX instructions used by the Tensor-Core kernel.
//
//   * mma.sync.m16n8k16   - one Tensor-Core MMA per warp: D[16x8] += A[16x16] * B[16x8]
//   * ldmatrix            - loads 8x8 b16 matrices from shared memory straight into
//                           the register layout mma expects (optionally transposed)
//   * cp.async            - asynchronous global->shared copies (Ampere+), used to
//                           double-buffer K/V tiles behind the compute
//
// Fragment layouts (PTX ISA, "Matrix Fragments for mma.m16n8k16"), with
// g = lane / 4 and t = lane % 4:
//   A (16x16, row):  a0 = A[g][2t..2t+1]      a1 = A[g+8][2t..2t+1]
//                    a2 = A[g][2t+8..2t+9]    a3 = A[g+8][2t+8..2t+9]
//   B (16x8,  col):  b0 = B[2t..2t+1][g]      b1 = B[2t+8..2t+9][g]
//   C/D (16x8, f32): c0,c1 = C[g][2t..2t+1]   c2,c3 = C[g+8][2t..2t+1]
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace fused_attn {
namespace ptx {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---- cp.async ---------------------------------------------------------------
// 16-byte async copy; src_bytes < 16 zero-fills the remainder (used for OOB rows).
__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src, int src_bytes = 16) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(smem_dst)), "l"(gmem_src),
               "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// ---- ldmatrix ---------------------------------------------------------------
// Four 8x8 b16 matrices; lane i supplies the row address for matrix i/8, row i%8.
__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(smem_u32(smem)));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(smem_u32(smem)));
}

// ---- mma.sync ---------------------------------------------------------------
template <typename T> struct Mma;

template <> struct Mma<__half> {
  __device__ __forceinline__ static void m16n8k16(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  // pack two fp32 into one b16x2 register (low half = first element)
  __device__ __forceinline__ static uint32_t pack(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
  }
};

template <> struct Mma<__nv_bfloat16> {
  __device__ __forceinline__ static void m16n8k16(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  __device__ __forceinline__ static uint32_t pack(float lo, float hi) {
    __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
  }
};

// ---- warp reductions over the 4 lanes that share a row in the C fragment ---------
__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
  v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
  return v;
}
__device__ __forceinline__ float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffff, v, 1);
  v += __shfl_xor_sync(0xffffffff, v, 2);
  return v;
}

}  // namespace ptx
}  // namespace fused_attn
