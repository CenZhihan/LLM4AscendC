#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(GruCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(h0, ge::TensorType::ALL())
    .INPUT(w_ih, ge::TensorType::ALL())
    .INPUT(w_hh, ge::TensorType::ALL())
    .INPUT(b_ih, ge::TensorType::ALL())
    .INPUT(b_hh, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(GruCustom);

}

#endif
