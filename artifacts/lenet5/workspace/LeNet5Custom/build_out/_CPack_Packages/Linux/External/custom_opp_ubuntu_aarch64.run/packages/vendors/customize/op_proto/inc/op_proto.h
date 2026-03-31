#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(LeNet5Custom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(conv1_w, ge::TensorType::ALL())
    .INPUT(conv1_b, ge::TensorType::ALL())
    .INPUT(conv2_w, ge::TensorType::ALL())
    .INPUT(conv2_b, ge::TensorType::ALL())
    .INPUT(fc1_w, ge::TensorType::ALL())
    .INPUT(fc1_b, ge::TensorType::ALL())
    .INPUT(fc2_w, ge::TensorType::ALL())
    .INPUT(fc2_b, ge::TensorType::ALL())
    .INPUT(fc3_w, ge::TensorType::ALL())
    .INPUT(fc3_b, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(LeNet5Custom);

}

#endif
