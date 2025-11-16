#include <torch/all.h>

#include "mxfp8_quant.cuh"

void es_sm100_mxfp8_blockscaled_grouped_quant(
    const torch::Tensor& input,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& blockscale_offsets,
    torch::Tensor& quant_output,
    torch::Tensor& scale_factor) {
  auto stream = at::cuda::getCurrentCUDAStream();
  if (input.dtype() == torch::kBFloat16) {
    expert_specialization::launch_es_sm100_mxfp8_blockscaled_grouped_quant<__nv_bfloat16>(input, problem_sizes, expert_offsets, blockscale_offsets, quant_output, scale_factor);
  } else if (input.dtype() == torch::kFloat16) {
    expert_specialization::launch_es_sm100_mxfp8_blockscaled_grouped_quant<__half>(input, problem_sizes, expert_offsets, blockscale_offsets, quant_output, scale_factor);
  } else {
    TORCH_CHECK(false, "dtype must be kFloat16 or kBFloat16");
  }
}
