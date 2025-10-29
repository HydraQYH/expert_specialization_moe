#include <torch/all.h>

#include "es_sm100_mxfp8_blockscaled_launcher.cuh"

void es_sm100_mxfp8_blockscaled_grouped_mm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& sfa,
    const torch::Tensor& sfb,
    torch::Tensor& d,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets) {
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
  TORCH_CHECK(problem_sizes.size(1) == 3, "problem_sizes must have shape (num_experts, 3)");
  TORCH_CHECK(problem_sizes.size(0) == expert_offsets.size(0), "Number of experts in problem_sizes must match expert_offsets");
  TORCH_CHECK(problem_sizes.dtype() == torch::kInt32, "problem_sizes must be int32");
  TORCH_CHECK(a.size(1) % 128 == 0, "k should align 128");

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
