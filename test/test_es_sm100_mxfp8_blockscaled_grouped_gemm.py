import random
from typing import Tuple
import pytest
import torch
import tensorrt_llm
from expert_specialization.ops import es_sm100_mxfp8_blockscaled_grouped_mm

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

def test_es_sm100_mxfp8_blockscaled_grouped_mm(num_experts, out_dtype):
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
    m_g = random.randint(1, 256)
    # print("Problem Size:", m_g, n_g, k_g)
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
    
    ref_d = a @ b.T
    ref_d_list.append(ref_d)

  a = torch.concat(a_list, dim=0)
  b = torch.stack(b_list, dim=0).transpose(1, 2)
  sfa = torch.concat(sfa_list, dim=0)
  sfb = torch.concat(sfb_list, dim=0)
  d = torch.empty((expert_offset, n_g), device=device, dtype=out_dtype)

  _problem_sizes = torch.tensor(problem_sizes).to(device=device, dtype=torch.int32)
  _expert_offsets = torch.tensor(expert_offsets).to(device=device, dtype=torch.int32)
  _blockscale_offsets = torch.tensor(blockscale_offsets).to(device=device, dtype=torch.int32)
  es_sm100_mxfp8_blockscaled_grouped_mm(
    d, a, b, sfa, sfb, _problem_sizes, _expert_offsets, _blockscale_offsets
  )

  for g in range(num_experts):
    baseline = ref_d_list[g]
    actual = d[expert_offsets[g] : (expert_offsets[g] + problem_sizes[g][0])]
    diff = calc_diff(actual, baseline)
    assert diff < 0.001
    print(f"m_g={baseline.shape[0]} n_g={n_g} k_g={k_g} num_experts={num_experts}, out_dtype={out_dtype}, diff={diff:.5f}: OK")
    print(baseline)
    print(actual)

if __name__ == '__main__':
  test_es_sm100_mxfp8_blockscaled_grouped_mm(8, torch.bfloat16)
