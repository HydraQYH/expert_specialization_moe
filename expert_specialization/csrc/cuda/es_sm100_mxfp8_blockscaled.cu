#include <torch/all.h>

#include "es_sm100_mxfp8_blockscaled_launcher.cuh"

template void expert_specialization::launch_sm100_mxfp8_blockscaled_group_mm<expert_specialization::ExpertSpecializationSm100MXFP8BlockscaledGroupedGemmTraits<expert_specialization::MMA1SMConfig, cutlass::bfloat16_t>>(
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
  using GEMMTraits = ExpertSpecializationSm100MXFP8BlockscaledGroupedGemmTraits<MMA1SMConfig, cutlass::bfloat16_t>;
  using Gemm = GEMMTraits::Gemm;
  using SFFunctor = expert_specialization::Sm100Mxfp8BlockScaledLayoutFunctor<GEMMTraits>;
  using StrideFunctor = expert_specialization::Sm100Mxfp8BlockScaledStrideFunctor<GEMMTraits>;

  using Sm1xxBlkScaledConfig = typename GEMMTraits::Sm1xxBlkScaledConfig;
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(512, 1024, 768, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(512, 1024, 768, 1));
  cute::print(layout_SFA);
  printf("\n");
  cute::print(cute::size(layout_SFA));
  printf("\n");
  cute::print(cute::cosize(layout_SFA));
  printf("\n");
  cute::print(cute::size<0>(layout_SFA));
  printf("\n");
  cute::print(cute::size<1>(layout_SFA));
  printf("\n");
  cute::print(layout_SFB);
  printf("\n");
  cute::print(cute::size(layout_SFB));
  printf("\n");
  cute::print(cute::cosize(layout_SFB));
  printf("\n");
  cute::print(cute::size<0>(layout_SFB));
  printf("\n");
  cute::print(cute::size<1>(layout_SFB));
  printf("\n");

  auto gemm = Gemm();
  auto sf_functor = SFFunctor();
}
