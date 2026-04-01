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
tbe register API:
To make it easy to manage operator registration information, TBE provides a set of register APIs.
"""


class op_compute:

    """
    define a option.
    """
    def __init__(self, op_type):
        self.__op_type = op_type

    def get_func(self):

        def conv2d_compute(*args, **kwargs):
            tensor = Tensor()
            op = Op()
            op.attrs = {"attr_type": 0}
            tensor.op = op
            return [tensor]

        return conv2d_compute
        

class Op:
    attrs = {}

class Tensor:
    op = Op()


class op_operator:
    """
    define a option.
    """
    def __init__(self, op_type):
        self.__op_type = op_type

    def get_func(self):
        return conv2d


def conv2d(*args, **kwargs):
    """
    conv2d func
    """
    return


def get_op_compute(op_type, op_mode="dynamic"):
    """
    get op compute func info
    """
    if op_type == "GNTrainingReduce":
        return None
    else:
        return op_compute(op_type)


def get_operator(op_type):
    """
    get op realization func info
    """
    if op_type == "asdsddd":
        return None

    return op_operator(op_type)

case_switch = 0
def get_fusion_buildcfg():
    if case_switch == 0:
        return {"Add":{
            "read_write_bank_conflict": 1,
            "InjectSync": {"sync_mode": 3}}}
    elif case_switch == 1:
        return {"Add":{
            "read_write_bank_conflict": 1,
            "InjectSync": {"sync_mode": 3}},
            "Sub":{
            "read_write_bank_conflict": 1,
            "InjectSync": {"sync_mode": 2}}}
    return {}

def get_all_fusion_pass():
    return []