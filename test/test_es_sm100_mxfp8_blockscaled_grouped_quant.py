import random
from typing import Tuple
import pytest
import torch
import tensorrt_llm
from expert_specialization.ops import es_sm100_mxfp8_blockscaled_grouped_quant

seed = 77
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def align(val: int, alignment: int = 128) -> int:
  return int((val + alignment - 1) // alignment * alignment)

# Copy from: https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/utils.py
def calc_diff(x, y):
  x, y = x.double(), y.double()
  denominator = (x * x + y * y).sum()
  sim = 2 * (x * y).sum() / denominator
  return 1 - sim

def test_es_sm100_mxfp8_blockscaled_grouped_quant(num_experts, out_dtype):
  device = "cuda"
  alignment = 128
  n_g = random.randint(1, 64) * alignment
  k_g = random.randint(1, 64) * alignment
  # k_g = alignment

  expert_offset = 0
  expert_offsets = []
  blockscale_offset = 0
  blockscale_offsets = []
  problem_sizes = []
  a_list = []
  b_list = []
  a_q_list = []
  b_q_list = []
  sfa_list = []
  sfb_list = []
  ref_d_list = []

  for g in range(num_experts):
    m_g = random.randint(1, 4096)
    # m_g = 128
    expert_offsets.append(expert_offset)
    expert_offset += m_g
    blockscale_offsets.append(blockscale_offset)
    blockscale_offset += align(m_g, 128)
    problem_sizes.append([m_g, n_g, k_g])

    a = torch.randn((m_g, k_g), device=device, dtype=out_dtype)  # (M, K):(K, 1)
    a_q, sfa = torch.ops.trtllm.mxfp8_quantize(a, True, alignment=128)
    a_list.append(a)
    a_q_list.append(a_q)
    sfa_list.append(sfa)
  a = torch.concat(a_list, dim=0)
  a_q = torch.concat(a_q_list, dim=0)
  sfa = torch.concat(sfa_list, dim=0)

  _problem_sizes = torch.tensor(problem_sizes).to(device=device, dtype=torch.int32)
  _expert_offsets = torch.tensor(expert_offsets).to(device=device, dtype=torch.int32)
  _blockscale_offsets = torch.tensor(blockscale_offsets).to(device=device, dtype=torch.int32)
  a_quant = torch.zeros_like(a_q, device=device)
  scale_factor = torch.zeros_like(sfa, device=device)
  es_sm100_mxfp8_blockscaled_grouped_quant(
    a,
    _problem_sizes,
    _expert_offsets,
    _blockscale_offsets,
    a_quant,
    scale_factor
  )
  print(a_q)
  print(a_quant)
  print(torch.sum(scale_factor - sfa))
  print(scale_factor)
  print(sfa)

if __name__ == '__main__':
  test_es_sm100_mxfp8_blockscaled_grouped_quant(2, torch.bfloat16)
