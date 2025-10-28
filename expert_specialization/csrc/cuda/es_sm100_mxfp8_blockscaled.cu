#include <torch/all.h>

#include "es_sm100_mxfp8_blockscaled_launcher.cuh"

void es_sm100_mxfp8_blockscaled_grouped_mm(
    torch::Tensor& a,
    torch::Tensor& b,
    torch::Tensor& sfa,
    torch::Tensor& sfb,
    torch::Tensor& d,
    torch::Tensor& problem_sizes,
    torch::Tensor& expert_offsets) {
  auto stream = at::cuda::getCurrentCUDAStream();
  if (d.dtype() == torch::kBFloat16) {
    expert_specialization::es_sm100_mxfp8_blockscaled_group_mm_distpatch_out_dtype<cutlass::bfloat16_t>(
      a, b, sfa, sfb, d, problem_sizes, expert_offsets, stream);
  } else if (d.dtype() == torch::kFloat16) {
    expert_specialization::es_sm100_mxfp8_blockscaled_group_mm_distpatch_out_dtype<cutlass::half_t>(
      a, b, sfa, sfb, d, problem_sizes, expert_offsets, stream);
  } else {
    TORCH_CHECK(false, "dtype must be kFloat16 or kBFloat16");
  }
}
