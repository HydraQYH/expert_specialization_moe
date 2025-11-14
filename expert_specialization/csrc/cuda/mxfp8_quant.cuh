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
    const int* blockscale_offsets,
    cutlass::float_e4m3_t* quant_output,
    cutlass::float_ue8m0_t* scale_factor,
    int groups) {
  for (int g = 0; g < groups; g++) {
    int m = problem_sizes[g * 3 + 0];
    int k = problem_sizes[g * 3 + 2];
    int expert_offset = expert_offsets[g];
    int blockscale_offset = blockscale_offsets[g];

    auto input_tensor = make_tensor(
      make_gmem_ptr(input + expert_offset * k),
      make_layout(make_shape(m, k), LayoutRight{})
    );  // (M, K):(K, 1) half_t/bfloat16_t

    auto quant_output_tensor = make_tensor(
      make_gmem_ptr(quant_output + expert_offset * k),
      make_layout(make_shape(m, k), LayoutRight{})
    );  // (M, K):(K, 1) cutlass::float_e4m3_t

    auto scale_factor_tile_layout = make_layout(
      make_shape(make_shape(_32{}, _4{}), _4{}),
      make_stride(make_stride(_16{}, _4{}), _1{})
    );  // Scale Factor Tile: ((_32,_4), _4):((_16,_4), _1)
    auto scale_factor_shape = make_shape(ceil_div(m, 128) * 128, k / 32);
    auto scale_factor_layout = tile_to_shape(scale_factor_tile_layout, scale_factor_shape, LayoutRight{});
    // layout<0>(layout<0>(scale_factor_layout))  (_32,_4):(_16,_4) -- static
    // layout<1>(layout<0>(scale_factor_layout))  M_align_128 / 128 -- dynamic
    // layout<0>(layout<1>(scale_factor_layout))  _4:_1 -- static
    // layout<1>(layout<1>(scale_factor_layout))  K / 32 -- dynamic

    auto zipped_scale_factor_layout = make_layout(
      make_layout(layout<0>(layout<0>(scale_factor_layout)), layout<0>(layout<1>(scale_factor_layout))),
      make_layout(layout<1>(layout<0>(scale_factor_layout)), layout<1>(layout<1>(scale_factor_layout)))
    );

    auto scale_factor_tensor = make_tensor(
      make_gmem_ptr(scale_factor + blockscale_offset * (k / 32)),
      zipped_scale_factor_layout
    );

    auto input_shape = shape(input_tensor);  // (M, K):(K, 1)
    auto identity_tensor = make_identity_tensor(input_shape);
    auto predict_tensor = cute::lazy::transform(identity_tensor, [&](auto c) { return elem_less(c, input_shape); });

    auto tiler = make_shape(Int<BLOCK_M>{}, Int<BLOCK_K>{});

    auto tiled_input_tensor = zipped_divide(input_tensor, tiler);  // ((128, 128), (cdiv(M, 128), cdiv(K, 128)))
    auto tiled_quant_output_tensor = zipped_divide(quant_output_tensor, tiler);  // ((128, 128), (cdiv(M, 128), cdiv(K, 128)))
    auto tiled_predict_tensor = zipped_divide(predict_tensor, tiler);  // ((128, 128), (cdiv(M, 128), cdiv(K, 128)))

    auto total_tiles = size<1>(tiled_input_tensor); // cdiv(M, 128) * cdiv(K, 128)
    decltype(total_tiles) blk_offset = blockIdx.x;
    while (blk_offset < total_tiles) {
      auto current_input_tile = tiled_input_tensor(_, blk_offset);
      auto current_quant_output_tile = tiled_quant_output_tensor(_, blk_offset);
      auto current_predict_tile = tiled_predict_tensor(_, blk_offset);
      auto current_scale_factor_tile = scale_factor_tensor(_, blk_offset);

      if (thread0()) {
        print(tiled_input_tensor);
        printf("\n");
        print(tiled_quant_output_tensor);
        printf("\n");
        print(tiled_predict_tensor);
        printf("\n");
        print(scale_factor_tensor);
        printf("\n");
        print(current_scale_factor_tile);
        printf("\n");
      }

      blk_offset += gridDim.x;
    }
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
  // Specialize for Row Major Matrix Coalesced Access
  auto thr_layout = make_layout(
    make_shape(_8{}, _16{}),
    make_stride(_16{}, _1{})
  );
  auto val_layout = make_layout(
    make_shape(_1{}, _8{})
  );
  using CopyOpG2R = UniversalCopy<cutlass::AlignedArray<T_IN, size(val_layout)>>;
  using CopyAtomG2R = cute::Copy_Atom<CopyOpG2R, T_IN>;
  auto tiled_copy_g2s = cute::make_tiled_copy(CopyAtomG2R{}, thr_layout, val_layout); // Tiler_MN: (8, 128)

  using CopyOpR2G = UniversalCopy<cutlass::AlignedArray<cutlass::float_e4m3_t, size(val_layout)>>;
  using CopyAtomR2G = cute::Copy_Atom<CopyOpR2G, cutlass::float_e4m3_t>;
  auto tiled_copy_s2g = cute::make_tiled_copy(CopyAtomR2G{}, thr_layout, val_layout); // Tiler_MN: (8, 128)
#ifndef NDEBUG
  print(tiled_copy_g2s);
  print(tiled_copy_s2g);
#endif
  int max_active_blocks_per_sm = -1;
  AT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm,
    mxfp8_group_quant<T_IN>, size(tiled_copy_g2s), 0));
  dim3 grid(at::cuda::getCurrentDeviceProperties()->multiProcessorCount * max_active_blocks_per_sm, 1, 1);
  dim3 block(size(tiled_copy_g2s), 1, 1);
  int num_experts = (int)problem_sizes.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  mxfp8_group_quant<T_IN><<<grid, block, 0, stream>>>(
    reinterpret_cast<const T_IN*>(input.data_ptr()),
    reinterpret_cast<const int*>(problem_sizes.data_ptr()),
    reinterpret_cast<const int*>(expert_offsets.data_ptr()),
    reinterpret_cast<const int*>(blockscale_offsets.data_ptr()),
    reinterpret_cast<cutlass::float_e4m3_t*>(quant_output.data_ptr()),
    reinterpret_cast<cutlass::float_ue8m0_t*>(scale_factor.data_ptr()),
    num_experts
  );
}

} // namespace expert_specialization