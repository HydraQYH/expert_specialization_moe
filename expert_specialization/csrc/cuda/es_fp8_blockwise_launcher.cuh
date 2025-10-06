#pragma once
#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

#include "es_fp8_blockwise_functor.cuh"

namespace expert_specialization {

using namespace cute;

void es_sm90_fp8_blockwise_scaled_group_mm_pre_compute(
    // Output
    torch::Tensor& out_ptrs,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& a_scales_ptrs,
    torch::Tensor& b_scales_ptrs,
    torch::Tensor& layout_sfa,
    torch::Tensor& layout_sfb,
    torch::Tensor& lm_problem_sizes,
    torch::Tensor& mm_problem_sizes,
    torch::Tensor& hm_problem_sizes,
    // Input
    torch::Tensor& out_tensors,
    torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& expert_offsets
) {
  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b_tensors.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  using LayoutSFA = typename Fp8BlockwiseGroupedGemmSFLayoutFunctor<PerfConfigMiddleM>::LayoutSFA;
  using LayoutSFB = typename Fp8BlockwiseGroupedGemmSFLayoutFunctor<PerfConfigMiddleM>::LayoutSFB;
  struct Fp8BlockwiseGroupedGemmSFLayoutFunctor<PerfConfigMiddleM> sf_layout(
    reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()), reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr()));

  struct Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor<PerfConfigLowM> lm_psf(static_cast<int*>(lm_problem_sizes.data_ptr()));
  struct Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor<PerfConfigMiddleM> mm_psf(static_cast<int*>(mm_problem_sizes.data_ptr()));
  struct Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor<PerfConfigHighM> hm_psf(static_cast<int*>(hm_problem_sizes.data_ptr()));

  int num_experts = (int)expert_offsets.size(0);
  auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());
  if (out_tensors.dtype() == torch::kBFloat16) {
    struct Fp8BlockwiseGroupedGemmOffsetFunctor<cutlass::float_e4m3_t, float, cutlass::bfloat16_t> of(
      static_cast<int*>(expert_offsets.data_ptr()),
      static_cast<cutlass::float_e4m3_t*>(a_tensors.data_ptr()),
      static_cast<cutlass::float_e4m3_t*>(b_tensors.data_ptr()),
      static_cast<cutlass::bfloat16_t*>(out_tensors.data_ptr()),
      static_cast<float*>(a_scales.data_ptr()),
      static_cast<float*>(b_scales.data_ptr()),
      static_cast<cutlass::float_e4m3_t**>(a_ptrs.data_ptr()),
      static_cast<cutlass::float_e4m3_t**>(b_ptrs.data_ptr()),
      static_cast<float**>(a_scales_ptrs.data_ptr()),
      static_cast<float**>(b_scales_ptrs.data_ptr()),
      static_cast<cutlass::bfloat16_t**>(out_ptrs.data_ptr())
    );
    groupedGemmPreComputeKernel<<<1, num_experts, 0, stream>>>(
      static_cast<int*>(problem_sizes.data_ptr()), 
      of, sf_layout, lm_psf, mm_psf, hm_psf
    );
  } else if (out_tensors.dtype() == torch::kFloat16) {
    struct Fp8BlockwiseGroupedGemmOffsetFunctor<cutlass::float_e4m3_t, float, cutlass::half_t> of(
      static_cast<int*>(expert_offsets.data_ptr()),
      static_cast<cutlass::float_e4m3_t*>(a_tensors.data_ptr()),
      static_cast<cutlass::float_e4m3_t*>(b_tensors.data_ptr()),
      static_cast<cutlass::half_t*>(out_tensors.data_ptr()),
      static_cast<float*>(a_scales.data_ptr()),
      static_cast<float*>(b_scales.data_ptr()),
      static_cast<cutlass::float_e4m3_t**>(a_ptrs.data_ptr()),
      static_cast<cutlass::float_e4m3_t**>(b_ptrs.data_ptr()),
      static_cast<float**>(a_scales_ptrs.data_ptr()),
      static_cast<float**>(b_scales_ptrs.data_ptr()),
      static_cast<cutlass::half_t**>(out_ptrs.data_ptr())
    );
    groupedGemmPreComputeKernel<<<1, num_experts, 0, stream>>>(
      static_cast<int*>(problem_sizes.data_ptr()), 
      of, sf_layout, lm_psf, mm_psf, hm_psf
    );
  } else {
    TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
  }
}

template<typename GemmTraits>
void launch_sm90_fp8_blockwise_scaled_group_mm(
    torch::Tensor& out_ptrs,
    const torch::Tensor& a_ptrs,
    const torch::Tensor& b_ptrs,
    const torch::Tensor& a_scales_ptrs,
    const torch::Tensor& b_scales_ptrs,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_d,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes) {
  using ElementA = typename GemmTraits::ElementA;
  using StrideA = typename GemmTraits::StrideA;
  using ElementB = typename GemmTraits::ElementB;
  using StrideB = typename GemmTraits::StrideB;
  using ElementAccumulator = typename GemmTraits::ElementAccumulator;
  using LayoutSFA = typename GemmTraits::LayoutSFA;
  using LayoutSFB = typename GemmTraits::LayoutSFB;
  using ElementD = typename GemmTraits::ElementD;
  using StrideD = typename GemmTraits::StrideD;
  using UnderlyingProblemShape = typename GemmTraits::ProblemShape::UnderlyingProblemShape;
  using Gemm = typename GemmTraits::Gemm;
  using GemmKernel = typename GemmTraits::GemmKernel;

  int num_experts = (int)problem_sizes.size(0);
  Gemm gemm_op;

  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementA**>(a_ptrs.data_ptr()),
      static_cast<StrideA*>(stride_a.data_ptr()),
      static_cast<const ElementB**>(b_ptrs.data_ptr()),
      static_cast<StrideB*>(stride_b.data_ptr()),
      static_cast<const ElementAccumulator**>(a_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),
      static_cast<const ElementAccumulator**>(b_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr())};

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = c10::cuda::current_device();
  hw_info.sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  typename GemmKernel::EpilogueArguments epilogue_args{
      {},
      nullptr,
      nullptr,
      static_cast<ElementD**>(out_ptrs.data_ptr()),
      static_cast<StrideD*>(stride_d.data_ptr())};

  UnderlyingProblemShape* problem_sizes_as_shapes = static_cast<UnderlyingProblemShape*>(problem_sizes.data_ptr());
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info};

  at::cuda::CUDAGuard device_guard{(char)a_ptrs.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(a_ptrs.get_device());

  auto can_implement_status = gemm_op.can_implement(args);
  TORCH_CHECK(can_implement_status == cutlass::Status::kSuccess, "Failed to implement GEMM");

  torch::TensorOptions options_uint8 = torch::TensorOptions().dtype(torch::kUInt8).device(out_ptrs.device());
  size_t workspace_size = gemm_op.get_workspace_size(args);
  torch::Tensor workspace = torch::empty(workspace_size, options_uint8);

  auto status = gemm_op.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm_op.run(stream, nullptr, true);  // Enable PDL
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}

template<typename OutType>
void es_sm90_fp8_blockwise_scaled_group_mm_distpatch_out_dtype(
    torch::Tensor& out_ptrs,
    const torch::Tensor& a_ptrs,
    const torch::Tensor& b_ptrs,
    const torch::Tensor& a_scales_ptrs,
    const torch::Tensor& b_scales_ptrs,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_d,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& lm_problem_sizes,
    const torch::Tensor& mm_problem_sizes,
    const torch::Tensor& hm_problem_sizes) {
  using LowMGemmTraits = ExpertSpecializationSm90FP8BlockwiseGroupedGemmTraits<OutType, cutlass::layout::ColumnMajor, PerfConfigLowM>;
  using MiddleMGemmTraits = ExpertSpecializationSm90FP8BlockwiseGroupedGemmTraits<OutType, cutlass::layout::RowMajor, PerfConfigMiddleM>;
  using HighMGemmTraits = ExpertSpecializationSm90FP8BlockwiseGroupedGemmTraits<OutType, cutlass::layout::RowMajor, PerfConfigHighM>;
  launch_sm90_fp8_blockwise_scaled_group_mm<LowMGemmTraits>(
    out_ptrs, b_ptrs, a_ptrs, b_scales_ptrs, a_scales_ptrs,
    stride_b, stride_a, stride_d, layout_sfb, layout_sfa,
    lm_problem_sizes
  );
  launch_sm90_fp8_blockwise_scaled_group_mm<MiddleMGemmTraits>(
    out_ptrs, a_ptrs, b_ptrs, a_scales_ptrs, b_scales_ptrs,
    stride_a, stride_b, stride_d, layout_sfa, layout_sfb,
    mm_problem_sizes
  );
  launch_sm90_fp8_blockwise_scaled_group_mm<HighMGemmTraits>(
    out_ptrs, a_ptrs, b_ptrs, a_scales_ptrs, b_scales_ptrs,
    stride_a, stride_b, stride_d, layout_sfa, layout_sfb,
    hm_problem_sizes
  );
}

} // namespace expert_specialization
