import argparse
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

import torch
import tensorrt_llm
from expert_specialization.ops import es_sm100_mxfp8_blockscaled_grouped_mm

random.seed(28)

def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def create_unbalanced_expert_token_distribution(max_num_experts):
    ratios = [random.random() for _ in range(max_num_experts)]
    def convert_to_tokens(ratio: float):
        if ratio <= 0.7:
            return random.randint(1, 32)
        elif ratio > 0.7 and ratio <= 0.85:
            return random.randint(32, 64)
        elif ratio > 0.85 and ratio <= 0.95:
            return random.randint(64, 128)
        elif ratio > 0.95:
            return random.randint(128, 1024)
        else:
            return 128
    group_ms = [convert_to_tokens(ratio) for ratio in ratios]
    return group_ms

group_ms = create_unbalanced_expert_token_distribution(8192)
# group_ms = [128 for _ in range(8192)]
# group_ms = [128 if i % 2 == 0 else 64 for i in range(8192)]

def align(val: int, alignment: int = 128) -> int:
    return int((val + alignment - 1) // alignment * alignment)


def bench_es(
    n: int,
    k: int,
    num_groups: int,
    num_warmup: int,
    num_run: int,
) -> Tuple[float, int]:
    device = "cuda"
    alignment = 128
    n_g = ceil_div(n, alignment) * alignment
    k_g = ceil_div(k, alignment) * alignment
    out_dtype = torch.bfloat16

    expert_offset = 0
    expert_offsets = []
    blockscale_offset = 0
    blockscale_offsets = []
    problem_sizes = []
    a_list = []
    b_list = []
    sfa_list = []
    sfb_list = []

    for g in range(num_groups):
        m_g = group_ms[g]
        expert_offsets.append(expert_offset)
        expert_offset += m_g
        blockscale_offsets.append(blockscale_offset)
        blockscale_offset += align(m_g, 128)
        problem_sizes.append([m_g, n_g, k_g])

        a = torch.randn((m_g, k_g), device=device, dtype=out_dtype)  # (M, K):(K, 1)
        a_q, sfa = torch.ops.trtllm.mxfp8_quantize(a, True, alignment=128)
        b = torch.randn((n_g, k_g), device=device, dtype=out_dtype)  # (N, K):(K, 1)
        b_q, sfb = torch.ops.trtllm.mxfp8_quantize(b, True, alignment=128)
        a_list.append(a_q)
        b_list.append(b_q)
        sfa_list.append(sfa)
        sfb_list.append(sfb)

    a = torch.concat(a_list, dim=0)
    b = torch.stack(b_list, dim=0).transpose(1, 2)
    sfa = torch.concat(sfa_list, dim=0)
    sfb = torch.concat(sfb_list, dim=0)
    d = torch.empty((expert_offset, n_g), device=device, dtype=out_dtype)

    _problem_sizes = torch.tensor(problem_sizes).to(device=device, dtype=torch.int32)
    _expert_offsets = torch.tensor(expert_offsets).to(device=device, dtype=torch.int32)
    _blockscale_offsets = torch.tensor(blockscale_offsets).to(device=device, dtype=torch.int32)
    
    def run_cutlass():
        es_sm100_mxfp8_blockscaled_grouped_mm(
            d, a, b, sfa, sfb, _problem_sizes, _expert_offsets, _blockscale_offsets
        )

    # warmup
    for _ in range(num_warmup):
        run_cutlass()
    torch.cuda.synchronize()

    # run
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_run):
        run_cutlass()
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    avg = start_event.elapsed_time(end_event) / num_run * 1000  # us

    torch.cuda.nvtx.range_push("es")
    run_cutlass()
    torch.cuda.nvtx.range_pop()

    return avg, expert_offsets[-1]

benchmark_kernels = {
    "es": bench_es
}

@dataclass
class ShapeArg:
    n: int
    k: int
    num_groups: int


def benchmark_one_shape(
    shape_args: List[ShapeArg],
    num_warmup: int,
    num_run: int,
):
    for shape in shape_args:
        print(
            f"\nBenchmark: n={shape.n}, k={shape.k}, num_groups={shape.num_groups}"
        )
        for kernel_name, kernel_func in benchmark_kernels.items():
            average_time, m = kernel_func(
                shape.n,
                shape.k,
                shape.num_groups,
                num_warmup,
                num_run,
            )
            print(f"{kernel_name}: {average_time} us")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-run", type=int, default=20)
    shape_args = [
        # Prefill, DeepSeek-R1, gateup, chunk_size = 4096, TP = 8
        ShapeArg(n=512, k=7168, num_groups=256),
        # Prefill, DeepSeek-R1, down, chunk_size = 4096, TP = 8
        ShapeArg(n=7168, k=256, num_groups=256),
        # Prefill, Qwen3-235B-A22B-FP8, gateup, TP = 4
        ShapeArg(n=768, k=4096, num_groups=128),
        # Prefill, Qwen3-235B-A22B-FP8, down, TP = 4
        ShapeArg(n=4096, k=384, num_groups=128),
        # Decode, DeepSeek-R1, gateup, bs = 128, EP = 8
        ShapeArg(n=4096, k=7168, num_groups=32),
        # Decode, DeepSeek-R1, gateup, bs = 256, EP = 16
        ShapeArg(n=4096, k=7168, num_groups=16),
    ]
    args = parser.parse_args()
    benchmark_one_shape(shape_args, args.num_warmup, args.num_run)

if __name__ == "__main__":
    main()
