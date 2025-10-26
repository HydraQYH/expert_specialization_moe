#pragma once
#include <cuda.h>

#include <iostream>

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

} // namespace expert_specialization
