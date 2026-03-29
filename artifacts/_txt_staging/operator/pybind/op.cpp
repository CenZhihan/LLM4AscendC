
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor matmul_with_transposed_b_custom_impl_npu(const at::Tensor& A, const at::Tensor& B) {
    // A: [M, K], B: [N, K], output: [M, N] with accumulation to float
    auto out_sizes = std::vector<int64_t>{A.size(0), B.size(0)};
    at::Tensor result = at::empty(out_sizes, A.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnMatmulWithTransposedBCustom, A, B, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matmul_with_transposed_b_custom", &matmul_with_transposed_b_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_with_transposed_b_custom", &matmul_with_transposed_b_custom_impl_npu, "C = A @ (B^T)");
}
