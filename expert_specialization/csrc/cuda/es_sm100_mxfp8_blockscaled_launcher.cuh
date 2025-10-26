#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cassert>
#include <iostream>
#include <string>

#include "cute/tensor.hpp"

#include "es_sm100_mxfp8_blockscaled_traits.cuh"
#include "es_sm100_mxfp8_blockscaled_functor.cuh"

namespace expert_specialization {

template <typename GemmTraits>
void launch_sm100_mxfp8_blockscaled_group_mm(
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
    cudaStream_t stream) {
  using Gemm = typename GemmTraits::Gemm;
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementSF = typename GemmTraits::ElementSF;
  using ElementD = typename GemmTraits::ElementOutput;
  using StrideA = typename GemmTraits::StrideA;
  using StrideB = typename GemmTraits::StrideB;
  using StrideD = typename GemmTraits::StrideD;
  using LayoutSFA = typename GemmTraits::LayoutSFA;
  using LayoutSFB = typename GemmTraits::LayoutSFB;
  using UnderlyingProblemShape = typename GemmTraits::ProblemShape::UnderlyingProblemShape;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = c10::cuda::current_device();
  hw_info.sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  hw_info.cluster_shape = GemmTraits::MMAConfig::preferred_cluster;
  hw_info.cluster_shape_fallback = GemmTraits::MMAConfig::fallback_cluster;
  
  int num_experts = (int)problem_sizes.size(0);

  UnderlyingProblemShape* underlying_problem_shape = \
    reinterpret_cast<UnderlyingProblemShape*>(problem_sizes.data_ptr());

  typename Gemm::Arguments arguments = {
    cutlass::gemm::GemmUniversalMode::kGrouped,
    {num_experts, underlying_problem_shape, nullptr},
    {
      reinterpret_cast<const ElementA**>(a_ptrs.data_ptr()),
      reinterpret_cast<StrideA*>(stride_a.data_ptr()),
      reinterpret_cast<const ElementB**>(b_ptrs.data_ptr()),
      reinterpret_cast<StrideB*>(stride_b.data_ptr()),
      reinterpret_cast<const ElementSF**>(sfa_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),
      reinterpret_cast<const ElementSF**>(sfb_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr())
    },
    {
      {},
      nullptr,
      nullptr,
      reinterpret_cast<ElementD**>(a_ptrs.data_ptr()),
      reinterpret_cast<StrideD*>(stride_d.data_ptr())
    },
    hw_info,
    {}  // Scheduler
  };

  Gemm gemm;

  auto can_implement_status = gemm.can_implement(arguments);
  TORCH_CHECK(can_implement_status == cutlass::Status::kSuccess, "Failed to implement GEMM");

  torch::TensorOptions options_uint8 = torch::TensorOptions().dtype(torch::kUInt8).device(d_ptrs.device());
  size_t workspace_size = gemm.get_workspace_size(arguments);
  torch::Tensor workspace = torch::empty(workspace_size, options_uint8);

  auto status = gemm.initialize(arguments, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm.run(stream, nullptr, true);  // Enable PDL
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}

template <typename OutType>
void es_sm100_mxfp8_blockscaled_group_mm_distpatch_out_dtype() {

}

} // namespace expert_specialization