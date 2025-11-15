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

template <
    typename TensorS,
    typename TensorP,
    typename TensorD,
    typename TensorSF,
    typename TiledCopyG2R,
    typename TiledCopyR2G>
__device__ void mxfp8_group_quant_tile(
    TensorS tensor_s,
    TensorP tensor_p,
    TensorD tensor_d,
    TensorSF tensor_sf,
    int m,
    TiledCopyG2R tiled_copy_g2r,
    TiledCopyR2G tiled_copy_r2g) {
  static_assert(size<0>(tensor_s) == 128 && size<1>(tensor_s) == 128 && stride<1>(tensor_s) == 1);
  static_assert(size<0>(tensor_d) == 128 && size<1>(tensor_d) == 128 && stride<1>(tensor_d) == 1);
  static_assert(size<0>(tensor_p) == 128 && size<1>(tensor_p) == 128);

  using Tiler_MN = typename TiledCopyG2R::Tiler_MN;

  auto tiled_tensor_s = tiled_divide(tensor_s, Tiler_MN{});
  auto tiled_tensor_p = tiled_divide(tensor_p, Tiler_MN{});
  auto tiled_tensor_d = tiled_divide(tensor_d, Tiler_MN{});
  static_assert(size<2>(tiled_tensor_s) == 1);
  static_assert(size<2>(tiled_tensor_p) == 1);
  static_assert(size<2>(tiled_tensor_d) == 1);
  auto squeeze_tiled_tensor_s = take<0, 2>(tiled_tensor_s);
  auto squeeze_tiled_tensor_d = take<0, 2>(tiled_tensor_p);
  auto squeeze_tiled_tensor_p = take<0, 2>(tiled_tensor_d);

  constexpr int tile_loop_count = size<1>(tiled_tensor_s);
// #ifndef NDEBUG
#if 0
  if (thread0()) {
    print(tensor_s);
    printf("\n");
    print(tiled_tensor_s);
    printf("\n");
  }
#endif

  constexpr int rows_in_tile = size<0>(Tiler_MN{});
  for (int t = 0; t < tile_loop_count; t++) {
    if (t * rows_in_tile >= m) {
      break;
    }
    auto current_copy_tile_s = tensor<0>(squeeze_tiled_tensor_s(_, t));
    auto current_copy_tile_p = tensor<0>(squeeze_tiled_tensor_p(_, t));
    auto current_copy_tile_d = tensor<0>(squeeze_tiled_tensor_d(_, t));

    // Global to Register copy
    auto thr_copy = tiled_copy_g2r.get_thread_slice(threadIdx.x);
    auto thr_tile_s = thr_copy.partition_S(current_copy_tile_s);
    auto thr_tile_p = thr_copy.partition_S(current_copy_tile_p);
    auto fragment = make_fragment_like(thr_tile_s);
    copy_if(tiled_copy_g2r, thr_tile_p, thr_tile_s, fragment);
#ifndef NDEBUG
    if (thread0()) {
      print(fragment);
      printf("\n");
    }
    break;
#endif
  }
}

template <typename T_IN, typename TiledCopyG2R, typename TiledCopyR2G>
__global__ void mxfp8_group_quant(
    const T_IN* input,
    const int* problem_sizes,
    const int* expert_offsets,
    const int* blockscale_offsets,
    cutlass::float_e4m3_t* quant_output,
    cutlass::float_ue8m0_t* scale_factor,
    int groups,
    TiledCopyG2R tiled_copy_g2r,
    TiledCopyR2G tiled_copy_r2g) {
  // Group-wise Schedule
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
    // layout<1>(layout<0>(scale_factor_layout))  M_align_128 / 128 -- dynamic shape dynamic stride
    // layout<0>(layout<1>(scale_factor_layout))  _4:_1 -- static
    // layout<1>(layout<1>(scale_factor_layout))  (K / 32) / 4 : _512 -- dynamic shape static stride

    // Reshape to zipped layout for 1D indexing
    auto zipped_scale_factor_layout = make_layout(
      make_layout(layout<0>(layout<0>(scale_factor_layout)), layout<0>(layout<1>(scale_factor_layout))),
      make_layout(layout<1>(layout<0>(scale_factor_layout)), layout<1>(layout<1>(scale_factor_layout)))
    );  // (((_32,_4),_4),(M_align_128 / 128,(K / 32) / 4)):(((_16,_4),_1),(?,_512))

    auto scale_factor_tensor = make_tensor(
      make_gmem_ptr(scale_factor + blockscale_offset * (k / 32)),
      zipped_scale_factor_layout
    );

    // Used for cases where M is not divisible by 128 (most scenarios).
    auto input_shape = shape(input_tensor);  // (M, K):(K, 1)
    auto identity_tensor = make_identity_tensor(input_shape);
    auto predict_tensor = cute::lazy::transform(identity_tensor, [&](auto c) { return elem_less(c, input_shape); });

    // (128, 128)
    auto tiler = make_shape(Int<BLOCK_M>{}, Int<BLOCK_K>{});

    auto tiled_input_tensor = zipped_divide(input_tensor, tiler);  // ((128, 128), (cdiv(M, 128), cdiv(K, 128)))
    auto tiled_quant_output_tensor = zipped_divide(quant_output_tensor, tiler);  // ((128, 128), (cdiv(M, 128), cdiv(K, 128)))
    auto tiled_predict_tensor = zipped_divide(predict_tensor, tiler);  // ((128, 128), (cdiv(M, 128), cdiv(K, 128)))

    auto total_tiles = size<1>(tiled_input_tensor); // cdiv(M, 128) * cdiv(K, 128)
    decltype(total_tiles) blk_offset = blockIdx.x;
    while (blk_offset < total_tiles) {
      auto current_input_tile = tensor<0>(tiled_input_tensor(_, blk_offset));
      auto current_quant_output_tile = tensor<0>(tiled_quant_output_tensor(_, blk_offset));
      auto current_predict_tile = tensor<0>(tiled_predict_tensor(_, blk_offset));
      auto current_scale_factor_tile = tensor<0>(scale_factor_tensor(_, blk_offset));
// #ifndef NDEBUG
#if 0
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
#endif
      mxfp8_group_quant_tile<
          decltype(current_input_tile),
          decltype(current_predict_tile),
          decltype(current_quant_output_tile),
          decltype(current_scale_factor_tile),
          TiledCopyG2R,
          TiledCopyR2G>(
        current_input_tile,
        current_predict_tile,
        current_quant_output_tile,
        current_scale_factor_tile,
        m,
        tiled_copy_g2r,
        tiled_copy_r2g);
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
  // TODO: Check Row Major
  auto thr_layout = make_layout(
    make_shape(_8{}, _16{}),
    make_stride(_16{}, _1{})
  );
  auto val_layout = make_layout(
    make_shape(_1{}, _8{})
  );
  using CopyOpG2R = UniversalCopy<cutlass::AlignedArray<T_IN, size(val_layout)>>;
  using CopyAtomG2R = cute::Copy_Atom<CopyOpG2R, T_IN>;
  auto tiled_copy_g2r = cute::make_tiled_copy(CopyAtomG2R{}, thr_layout, val_layout); // Tiler_MN: (8, 128)

  using CopyOpR2G = UniversalCopy<cutlass::AlignedArray<cutlass::float_e4m3_t, size(val_layout)>>;
  using CopyAtomR2G = cute::Copy_Atom<CopyOpR2G, cutlass::float_e4m3_t>;
  auto tiled_copy_r2g = cute::make_tiled_copy(CopyAtomR2G{}, thr_layout, val_layout); // Tiler_MN: (8, 128)
// #ifndef NDEBUG
#if 0
  print(tiled_copy_g2r);
  print(tiled_copy_r2g);
#endif
  int max_active_blocks_per_sm = -1;
  AT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm,
    mxfp8_group_quant<T_IN, decltype(tiled_copy_g2r), decltype(tiled_copy_r2g)>, 128, 0));
  dim3 grid(at::cuda::getCurrentDeviceProperties()->multiProcessorCount * max_active_blocks_per_sm, 1, 1);
  dim3 block(128, 1, 1);
  int num_experts = (int)problem_sizes.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  mxfp8_group_quant<T_IN, decltype(tiled_copy_g2r), decltype(tiled_copy_r2g)><<<grid, block, 0, stream>>>(
    reinterpret_cast<const T_IN*>(input.data_ptr()),
    reinterpret_cast<const int*>(problem_sizes.data_ptr()),
    reinterpret_cast<const int*>(expert_offsets.data_ptr()),
    reinterpret_cast<const int*>(blockscale_offsets.data_ptr()),
    reinterpret_cast<cutlass::float_e4m3_t*>(quant_output.data_ptr()),
    reinterpret_cast<cutlass::float_ue8m0_t*>(scale_factor.data_ptr()),
    num_experts,
    tiled_copy_g2r,
    tiled_copy_r2g
  );
}

} // namespace expert_specialization