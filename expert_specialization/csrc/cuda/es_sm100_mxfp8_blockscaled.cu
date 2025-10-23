#include <torch/all.h>

#include "es_sm100_mxfp8_blockscaled_launcher.cuh"

void es_sm100_mxfp8_blockscaled_grouped_mm(torch::Tensor& output) {
  using namespace expert_specialization;
  using GEMMTraits = ExpertSpecializationSm100MXFP8BlockscaledGroupedGemmTraits<MMA1SMConfig>;
  using Gemm = GEMMTraits::Gemm;

  auto gemm = Gemm();
}
