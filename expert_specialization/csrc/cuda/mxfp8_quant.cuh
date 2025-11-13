#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <cuda.h>
#include "cute/tensor.hpp"

namespace expert_specialization {

using namespace cute;
constexpr int BLOCK_M = 128;
constexpr int BLOCK_K = 128;

template <typename T_IN>
__global__ void mxfp8_group_quant(
    const T_IN* input,
    const int* problem_sizes,
    const int* expert_offsets,
    cutlass::float_e4m3_t* quant_output,
    cutlass::float_ue8m0_t* scale_factor,
    int groups) {
  for (int g = 0; g < groups; g++) {
    int m = problem_sizes[g * 3 + 0];
    int k = problem_sizes[g * 3 + 2];
    int expert_offset = expert_offsets[g];

    auto input_tensor = make_tensor(
      make_gmem_ptr(input + expert_offset * k),
      make_layout(make_shape(m, k), LayoutRight{})
    );

    auto quant_output_tensor = make_tensor(
      make_gmem_ptr(quant_output + expert_offset * k),
      make_layout(make_shape(m, k), LayoutRight{})
    );
  }
}

template <typename T_IN>
void launch_es_sm100_mxfp8_blockscaled_grouped_quant(
    const torch::Tensor& input,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& blockscale_offsets,
    torch::Tensor& quant_output,
    torch::Tensor& scale_factor) {

  auto thr_layout = make_layout(
    make_shape(_8{}, _16{}),
    make_stride(_16{}, _1{})
  );
  auto val_layout = make_layout(
    make_shape(_1{}, _8{})
  );
  using CopyOpG2R = UniversalCopy<cutlass::AlignedArray<T_IN, size(val_layout)>>;
  using CopyAtomG2R = cute::Copy_Atom<CopyOpG2R, T_IN>;
  auto tiled_copy_g2s = cute::make_tiled_copy(CopyAtomG2R{}, thr_layout, val_layout);

  using CopyOpR2G = UniversalCopy<cutlass::AlignedArray<cutlass::float_e4m3_t, size(val_layout)>>;
  using CopyAtomR2G = cute::Copy_Atom<CopyOpR2G, cutlass::float_e4m3_t>;
  auto tiled_copy_s2g = cute::make_tiled_copy(CopyAtomR2G{}, thr_layout, val_layout);
  print(tiled_copy_g2s);
  print(tiled_copy_s2g);
}

} // namespace expert_specialization