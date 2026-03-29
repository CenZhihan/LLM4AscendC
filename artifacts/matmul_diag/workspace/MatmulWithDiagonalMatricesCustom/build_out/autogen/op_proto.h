#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(MatmulWithDiagonalMatricesCustom)
    .INPUT(A, ge::TensorType::ALL())
    .INPUT(B, ge::TensorType::ALL())
    .OUTPUT(C, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(MatmulWithDiagonalMatricesCustom);

}

#endif
