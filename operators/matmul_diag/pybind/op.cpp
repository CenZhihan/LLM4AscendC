
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor matmul_with_diagonal_matrices_custom_impl_npu(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.dim() == 1 && B.dim() == 2, "A must be 1D and B must be 2D");
    TORCH_CHECK(A.size(0) == B.size(0), "A length must equal B rows");
    at::Tensor result = at::empty_like(B);
    EXEC_NPU_CMD(aclnnMatmulWithDiagonalMatricesCustom, A, B, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matmul_with_diagonal_matrices_custom", &matmul_with_diagonal_matrices_custom_impl_npu);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_with_diagonal_matrices_custom", &matmul_with_diagonal_matrices_custom_impl_npu, "diag(A) @ B");
}

