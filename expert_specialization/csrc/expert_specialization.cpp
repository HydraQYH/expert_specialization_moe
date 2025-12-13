#include <Python.h>
#include <torch/all.h>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

// Declaration
void es_fp8_blockwise_scaled_grouped_mm(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_d,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace);

void es_sm100_mxfp8_blockscaled_grouped_mm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& sfa,
    const torch::Tensor& sfb,
    torch::Tensor& d,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& blockscale_offsets);

void es_sm100_mxfp8_blockscaled_grouped_quant(
    const torch::Tensor& input,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& blockscale_offsets,
    torch::Tensor& quant_output,
    torch::Tensor& scale_factor);

// Defines the operators
TORCH_LIBRARY(expert_specialization, m) {
  m.def(
      "es_fp8_blockwise_scaled_grouped_mm(Tensor output, Tensor a, Tensor b, Tensor scales_a, Tensor scales_b, Tensor "
      "stride_a, Tensor stride_b, Tensor stride_d, Tensor problem_sizes, Tensor expert_offsets, Tensor workspace) -> "
      "()");
  m.def(
      "es_sm100_mxfp8_blockscaled_grouped_mm(Tensor a, Tensor b, Tensor sfa, Tensor sfb, Tensor d, Tensor "
      "problem_sizes, Tensor expert_offsets, Tensor blockscale_offsets) -> ()");
  m.def(
      "es_sm100_mxfp8_blockscaled_grouped_quant(Tensor input, Tensor problem_sizes, Tensor expert_offsets, Tensor "
      "blockscale_offsets, Tensor quant_output, Tensor scale_factor) -> () ");
}

TORCH_LIBRARY_IMPL(expert_specialization, CUDA, m) {
  m.impl("es_fp8_blockwise_scaled_grouped_mm", &es_fp8_blockwise_scaled_grouped_mm);
  m.impl("es_sm100_mxfp8_blockscaled_grouped_mm", &es_sm100_mxfp8_blockscaled_grouped_mm);
  m.impl("es_sm100_mxfp8_blockscaled_grouped_quant", &es_sm100_mxfp8_blockscaled_grouped_quant);
}
