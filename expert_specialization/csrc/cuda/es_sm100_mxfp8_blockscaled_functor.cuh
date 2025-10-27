#pragma once
#include <cuda.h>

#include <iostream>

#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"
#include "es_sm100_mxfp8_blockscaled_traits.cuh"

namespace expert_specialization {

using namespace cute;

template <typename GemmTraits>
struct Sm100Mxfp8BlockScaledLayoutFunctor {
  using Sm1xxBlkScaledConfig =  typename GemmTraits::Sm1xxBlkScaledConfig;
  using LayoutSFA = typename GemmTraits::LayoutSFA;
  using LayoutSFB = typename GemmTraits::LayoutSFB;
  LayoutSFA* layout_sfa_base{nullptr};
  LayoutSFB* layout_sfb_base{nullptr};

  Sm100Mxfp8BlockScaledLayoutFunctor() = default;
  Sm100Mxfp8BlockScaledLayoutFunctor(LayoutSFA* _layout_sfa_base, LayoutSFB* _layout_sfb_base)
      : layout_sfa_base(_layout_sfa_base), layout_sfb_base(_layout_sfb_base) {}

  void CUTE_DEVICE operator()(int64_t expert_id, int m, int n, int k) {
    LayoutSFA* layout_sfa_ptr = layout_sfa_base + expert_id;
    LayoutSFB* layout_sfb_ptr = layout_sfb_base + expert_id;
    *layout_sfa_ptr = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    *layout_sfb_ptr = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  }
};

template <typename GemmTraits>
struct Sm100Mxfp8BlockScaledStrideFunctor {
  using StrideA = typename GemmTraits::StrideA;
  using StrideB = typename GemmTraits::StrideB;
  using StrideD = typename GemmTraits::StrideD;
  StrideA* stride_A_base{nullptr};
  StrideB* stride_B_base{nullptr};
  StrideD* stride_D_base{nullptr};

  Sm100Mxfp8BlockScaledStrideFunctor() = default;
  Sm100Mxfp8BlockScaledStrideFunctor(StrideA* _stride_A_base, StrideB* _stride_B_base, StrideD* _stride_D_base)
      : stride_A_base(_stride_A_base), stride_B_base(_stride_B_base), stride_D_base(_stride_D_base) {}

  void CUTE_DEVICE operator()(int64_t expert_id, int m, int n, int k) {
    StrideA* stride_A = stride_A_base + expert_id;
    StrideB* stride_B = stride_B_base + expert_id;
    StrideD* stride_D = stride_D_base + expert_id;
    *stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    *stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    *stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  }
};

} // namespace expert_specialization
