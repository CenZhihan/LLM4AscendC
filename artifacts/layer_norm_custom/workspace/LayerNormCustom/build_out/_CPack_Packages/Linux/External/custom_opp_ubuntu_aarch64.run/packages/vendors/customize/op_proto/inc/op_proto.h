#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(LayerNormCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(gamma, ge::TensorType::ALL())
    .INPUT(beta, ge::TensorType::ALL())
    .OUTPUT(res_out, ge::TensorType::ALL())
    .ATTR(epsilon, Float, 1e-05)
    .OP_END_FACTORY_REG(LayerNormCustom);

}

#endif
