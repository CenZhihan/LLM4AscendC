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
class api_config():

    _bit_width = 32
    _singleton = None

    def __init__(self):
        pass

    def __enter__(self):
        return self

    @classmethod
    def get_instance(cls):
        if cls._singleton is None:
            cls._singleton = api_config()
        return cls._singleton

    @classmethod
    def bit_width_64(cls):
        cls._bit_width = 64
        return cls.get_instance()

    @classmethod
    def bit_width_32(cls):
        cls._bit_width = 32
        return cls.get_instance()

    @classmethod
    def query_bit_width(cls):
        return cls._bit_width

    def __exit__(self, ptype, value, trace):
        # recover to default value
        api_config._bit_width = 32

