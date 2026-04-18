#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

class CmdCompilerConfig:
    def __init__(self, compiler_path_o_level, self_defined_options, enable_sanitizer, enable_jump_expand, \
        self_defined_debug_config, meta_info_options, gen_assembel):
        self.compiler_path_o_level = compiler_path_o_level
        self.self_defined_options = self_defined_options
        self.enable_sanitizer = enable_sanitizer
        self.enable_jump_expand = enable_jump_expand
        self.self_defined_debug_config = self_defined_debug_config
        self.meta_info_options = meta_info_options
        self.gen_assembel = gen_assembel


class CmdCompilerParams:
    def __init__(self):
        pass


class CmdCompiler:
    def __init__(self, params: CmdCompilerParams, config: CmdCompilerConfig):
        pass

