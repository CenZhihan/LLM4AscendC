# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


cmake_minimum_required(VERSION 3.16)

if(DEFINED RUN_MODE)
    set(CMAKE_ASC_RUN_MODE ${RUN_MODE} CACHE STRING "Alias of RUN_MODE")
endif()

if(DEFINED NPU_ARCH)
    set(CMAKE_ASC_ARCHITECTURES ${NPU_ARCH} CACHE STRING "Alias of NPU_ARCH")
endif()

find_package(ASC REQUIRED)

project(kernel_samples LANGUAGES ASC CXX)

# 场景参数（通过命令行传入）
# 场景1：CastDequant - 反量化与类型转换融合
# 场景2：AddRelu - 向量加法与ReLU激活融合
# 场景3：Axpy - 标量乘法与向量加法融合
set(SCENARIO_NUM "2" CACHE STRING "Scene number (1: CastDequant, 2: AddRelu, 3: Axpy)")

add_definitions(-DSCENARIO_NUM=${SCENARIO_NUM})

add_executable(demo
    element_wise_compound_compute.asc
)

set(NPU_ARCH "dav-2201" CACHE STRING "NPU ARCH, e.g. dav-2201, dav-3510")

target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>
)

# ======================================================================================
# 链接库配置
# ======================================================================================
target_link_libraries(demo PRIVATE
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)