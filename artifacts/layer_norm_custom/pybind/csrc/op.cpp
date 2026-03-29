
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor layer_norm_impl_npu(const at::Tensor& x, const at::Tensor& gamma, const at::Tensor& beta, double epsilon) {
    // float argument not supported now, so use double negative_slope
    at::Tensor result = at::empty_like(x);
    EXEC_NPU_CMD(aclnnLayerNormCustom, x, gamma, beta, epsilon, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("layer_norm_custom", &layer_norm_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layer_norm_custom", &layer_norm_impl_npu, "layer norm");
}
