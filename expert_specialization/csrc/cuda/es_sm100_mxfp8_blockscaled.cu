#include <torch/all.h>

#include "es_sm100_mxfp8_blockscaled_launcher.cuh"

template void expert_specialization::launch_sm100_mxfp8_blockscaled_group_mm<expert_specialization::ExpertSpecializationSm100MXFP8BlockscaledGroupedGemmTraits<expert_specialization::MMA1SMConfig>>(
    torch::Tensor& d_ptrs,
    const torch::Tensor& a_ptrs,
    const torch::Tensor& b_ptrs,
    const torch::Tensor& sfa_ptrs,
    const torch::Tensor& sfb_ptrs,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_d,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes,
    cudaStream_t stream
);

void es_sm100_mxfp8_blockscaled_grouped_mm(torch::Tensor& output) {
  using namespace expert_specialization;
  using GEMMTraits = ExpertSpecializationSm100MXFP8BlockscaledGroupedGemmTraits<MMA1SMConfig>;
  using Gemm = GEMMTraits::Gemm;
  using SFFunctor = expert_specialization::Sm100Mxfp8BlockScaledLayoutFunctor<GEMMTraits>;

  auto gemm = Gemm();
  auto sf_functor = SFFunctor();
}
