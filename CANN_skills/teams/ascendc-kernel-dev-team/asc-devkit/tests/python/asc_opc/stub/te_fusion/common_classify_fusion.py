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

from tbe.dsl import classify

class _op():
    def __init__(self):
        self.value = []
        self.idx = []


class CommonClassifyFusion():
    def __init__(self, op_list, pattern, mode):
        self.op_list = op_list
        self.pattern = pattern
        self.ins_list = None
        self.mode = mode
        self.placeholder_op = _op()
        self.init()


    @staticmethod
    def _handle_input_range(inputs):
        for input in inputs:
            shape_range = input.get("range")
            if shape_range is None:
                return
            for range in shape_range:
                if len(range) == 2:
                    if range[1] == -1:
                        range[1] = None


    def init(self):
        for key, node in enumerate(self.op_list):
            if node.get("type") == "Data":
                self.placeholder_op.value.append(node)
                self.placeholder_op.idx.append(key)

        # define dynamic inputs
        inputs_desc = [x.get("output_desc")[0] for x in self.placeholder_op.value]
        CommonClassifyFusion._handle_input_range(inputs_desc)
        if self.pattern == "cube":
            self.ins_list = [inputs_desc]
        else:
            self.ins_list = classify(inputs_desc, self.pattern)
