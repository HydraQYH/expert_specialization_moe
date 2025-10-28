import numpy as np
import torch

__all__ = ["es_fp8_blockwise_scaled_grouped_mm"]

def es_fp8_blockwise_scaled_grouped_mm(
    output,
    a,
    b,
    scales_a,
    scales_b,
    stride_a,
    stride_b,
    stride_d,
    problem_sizes,
    expert_offsets
):
    torch.ops.expert_specialization.es_fp8_blockwise_scaled_grouped_mm.default(
        output,
        a,
        b,
        scales_a,
        scales_b,
        stride_a,
        stride_b,
        stride_d,
        problem_sizes,
        expert_offsets
    )

def es_sm100_mxfp8_blockscaled_grouped_mm(
    output,
    a,
    b,
    sfa,
    sfb,
    problem_sizes,
    expert_offsets
):
    torch.ops.expert_specialization.es_sm100_mxfp8_blockscaled_grouped_mm.default(
        a, b, sfa, sfb, output, problem_sizes, expert_offsets
    )