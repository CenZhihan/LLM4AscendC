#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(Conv2dGroupNormScaleMaxPoolClampCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(weight, ge::TensorType::ALL())
    .INPUT(bias, ge::TensorType::ALL())
    .INPUT(gn_gamma, ge::TensorType::ALL())
    .INPUT(gn_beta, ge::TensorType::ALL())
    .INPUT(scale, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(Conv2dGroupNormScaleMaxPoolClampCustom);

}

#endif
