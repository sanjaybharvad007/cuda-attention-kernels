// PyTorch bindings for the CUDA attention kernel.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "flash_attn.h"

namespace {

fused_attn::DType to_dtype(at::ScalarType t) {
  switch (t) {
    case at::kHalf: return fused_attn::DType::kFloat16;
    case at::kBFloat16: return fused_attn::DType::kBFloat16;
    case at::kFloat: return fused_attn::DType::kFloat32;
    default: TORCH_CHECK(false, "unsupported dtype ", t);
  }
}

fused_attn::AttnParams make_params(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                                   torch::Tensor& o, bool causal, double scale) {
  fused_attn::AttnParams p{};
  p.q = q.data_ptr(); p.k = k.data_ptr(); p.v = v.data_ptr(); p.o = o.data_ptr();
  p.q_batch_stride = q.stride(0); p.q_head_stride = q.stride(1); p.q_row_stride = q.stride(2);
  p.k_batch_stride = k.stride(0); p.k_head_stride = k.stride(1); p.k_row_stride = k.stride(2);
  p.v_batch_stride = v.stride(0); p.v_head_stride = v.stride(1); p.v_row_stride = v.stride(2);
  p.o_batch_stride = o.stride(0); p.o_head_stride = o.stride(1); p.o_row_stride = o.stride(2);
  p.batch = q.size(0); p.num_heads_q = q.size(1); p.num_heads_kv = k.size(1);
  p.seqlen_q = q.size(2); p.seqlen_k = k.size(2); p.head_dim = q.size(3);
  p.scale = static_cast<float>(scale);
  p.causal = causal;
  p.dtype = to_dtype(q.scalar_type());
  return p;
}

void check_inputs(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "expected [B, H, N, D]");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type());
  TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1, "head_dim must be contiguous");
  const int64_t d = q.size(3);
  TORCH_CHECK(d == 64 || d == 128, "CUDA kernel supports head_dim 64/128 (use the Triton path otherwise)");
  TORCH_CHECK(q.size(1) % k.size(1) == 0, "H_q must be a multiple of H_kv");
  TORCH_CHECK(k.sizes() == v.sizes());
  TORCH_CHECK(reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0, "K/V must be 16B aligned");
  const int64_t vec = 16 / k.element_size();  // elements per 16B vector load
  TORCH_CHECK(k.stride(2) % vec == 0 && v.stride(2) % vec == 0 && k.stride(1) % vec == 0 && v.stride(1) % vec == 0,
              "K/V row and head strides must be multiples of ", vec, " elements for vectorised loads");

}

torch::Tensor attention_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                            bool causal, double scale) {
  check_inputs(q, k, v);
  const at::cuda::OptionalCUDAGuard guard(q.device());
  auto o = torch::empty_like(q);
  auto p = make_params(q, k, v, o, causal, scale);
  fused_attn::run_attention_fwd(p, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return o;
}

torch::Tensor attention_fwd_mma(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                                bool causal, double scale) {
  check_inputs(q, k, v);
  TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16, "Tensor-Core kernel needs fp16/bf16");
  TORCH_CHECK(q.stride(2) % 8 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0,
              "Q rows must be 16B aligned for cp.async");
  TORCH_CHECK(q.stride(2) % 2 == 0, "row strides must be even for 32-bit vector stores");
  const auto prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major >= 8, "Tensor-Core kernel requires sm_80+ (got sm_", prop->major, prop->minor, ")");
  const size_t smem = fused_attn::attention_fwd_mma_smem_bytes(q.size(3), q.element_size());
  TORCH_CHECK(smem <= static_cast<size_t>(prop->sharedMemPerBlockOptin), "not enough shared memory per block: need ",
              smem, " bytes");
  const at::cuda::OptionalCUDAGuard guard(q.device());
  auto o = torch::empty_like(q);
  auto p = make_params(q, k, v, o, causal, scale);
  fused_attn::run_attention_fwd_mma(p, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return o;
}

bool has_tensor_core_kernel() {
  if (!at::cuda::is_available()) return false;
  return at::cuda::getCurrentDeviceProperties()->major >= 8;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attention_fwd", &attention_fwd, "Fused attention forward (CUDA, SIMT)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false, py::arg("scale") = 0.0);
  m.def("attention_fwd_mma", &attention_fwd_mma, "Fused attention forward (CUDA, Tensor Cores: mma.sync)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false, py::arg("scale") = 0.0);
  m.def("has_tensor_core_kernel", &has_tensor_core_kernel, "True on sm_80+ devices");
}
