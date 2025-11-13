import random
from typing import Tuple
import pytest
import torch
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

  expert_offset = 0
  expert_offsets = []
  blockscale_offset = 0
  blockscale_offsets = []
  problem_sizes = []
  a_list = []
  b_list = []
  sfa_list = []
  sfb_list = []
  ref_d_list = []

  for g in range(num_experts):
    m_g = random.randint(1, 4096)
    expert_offsets.append(expert_offset)
    expert_offset += m_g
    blockscale_offsets.append(blockscale_offset)
    blockscale_offset += align(m_g, 128)
    problem_sizes.append([m_g, n_g, k_g])

    a = torch.randn((m_g, k_g), device=device, dtype=out_dtype)  # (M, K):(K, 1)
    a_list.append(a)
  a = torch.concat(a_list, dim=0)

  _problem_sizes = torch.tensor(problem_sizes).to(device=device, dtype=torch.int32)
  _expert_offsets = torch.tensor(expert_offsets).to(device=device, dtype=torch.int32)
  _blockscale_offsets = torch.tensor(blockscale_offsets).to(device=device, dtype=torch.int32)
  a_quant = torch.empty_like(a, dtype=torch.float8_e4m3fn)
  scale_factor = torch.empty((blockscale_offset, k_g // 32), dtype=torch.float8_e8m0fnu, device='cuda')
  es_sm100_mxfp8_blockscaled_grouped_quant(
    a,
    _problem_sizes,
    _expert_offsets,
    _blockscale_offsets,
    a_quant,
    scale_factor
  )

if __name__ == '__main__':
  test_es_sm100_mxfp8_blockscaled_grouped_quant(8, torch.bfloat16)
