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

from tbe.dsl.unify_schedule.constants import Pattern


class _op():
    def __init__(self):
        self.value = []
        self.idx = []


class CubeClassifyFusion():
    def __init__(self, op_list, pattern, mode="cube"):
        self.op_list = op_list
        self.pattern = pattern
        self.ins_list = None

        self.placeholder_op = _op()
        self.init()


    def init(self):
        cube_op_type = ""
        cube_input_name_vec = []
        for key, node in enumerate(self.op_list):
            if node.get("pattern") in [Pattern.CONV2D, Pattern.CONV2D_BACKPROP_INPUT, Pattern.CONV2D_BACKPROP_FILTER, \
                                       Pattern.MAT_MUL, Pattern.BATCH_MATMUL, Pattern.CONV3D, \
                                       Pattern.CONV3D_BACKPROP_INPUT, Pattern.CONV3D_BACKPROP_FILTER]:
                for input_desc in node["input_desc"]:
                    cube_input_name_vec.append(input_desc["name"])
                cube_op_type = node.get("type", "")

        input_desc_vec = []
        for key, node in enumerate(self.op_list):
            if node.get("type") == "Data":
                self.placeholder_op.value.append(node)
                self.placeholder_op.idx.append(key)
                output_desc = node.get("output_desc")[0]
                if output_desc["name"] in cube_input_name_vec:
                    output_desc["input_pattern"] = "cube"
                    output_desc["input_op_type"] = cube_op_type
                input_desc_vec.append(output_desc)

        self.ins_list = [input_desc_vec]

