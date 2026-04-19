#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""
dynamic apply_adam_v2
"""

# 'pylint: disable=unused-argument,too-many-arguments,too-many-locals
def apply_adam_v2(var, m, v, lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay,
                  step_size, var_out, m_out, v_out, adam_mode, kernel_name="ApplyAdamV2"):
    """
    algorithm: assign positive bboxes
        default:
            if max_grad_norm > 0 and global_grad_norm > 1: combined_grad /= global_grad_norm
            m = m * beta1 + combined_grad * (1 - beta1)
            v = v * beta2 + combined_grad * combined_grad * (1 - beta2)
            update = m / (v.sqrt() + epsilon)
            if weight_decay > 0: update += weight_decay * var
            update_with_lr = lr * update
            var -= update_with_lr
        if adam_mode == "mbart_adam":
            exp_avg = exp_avg * beta1 + combined_grad * (1-beta1)
            exp_avg_sq = exp_avg_sq * beta2 + combined_grad * combined_grad * (1 - beta2)
            update = exp_avg / (exp_avg_sq.sqrt() + epsilon)
            update_with_st = update * step_size
            if compute_mode // 2 == 1: update_with_st += weight_decay * lr * combined_param
            combined_param -= update_with_st

    Parameters
    ----------
    var:
        A Tensor. Support float16/float32.
    m :
        A Tensor. Datatype and shape are same as var.
    v:
        A Tensor. Datatype and shape are same as var.
    lr:
        A Tensor. Datatype is same as var. Shape (1, )
    beta1 :
        A Tensor. Datatype is same as var. Shape (1, )
    beta2 :
        A Tensor. Datatype is same as var. Shape (1, )
    epsilon :
        A Tensor. Datatype is same as var. Shape (1, )
    grad :
        A Tensor. Datatype and shape are same as var.
    max_grad_norm:
        A Tensor. Datatype is same as var. Shape (1, )
    global_grad_norm :
        A Tensor. Datatype is same as var. Shape (1, )
    weight_decay :
        A Tensor. Datatype is same as var. Shape (1, )
    var_out:
        A Tensor. Datatype and shape are same as var.
    m_out:
        A Tensor. Datatype and shape are same as var.
    v_out:
        A Tensor. Datatype and shape are same as var.
    kernel_name : str
        cce kernel name, default value is ApplyAdamV2
    Returns
    -------
    None
    """
    return