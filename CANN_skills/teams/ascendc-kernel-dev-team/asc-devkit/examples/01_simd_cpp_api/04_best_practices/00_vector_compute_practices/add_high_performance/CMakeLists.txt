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

find_package(ASC REQUIRED HINTS $ENV{ASCEND_INSTALL_PATH}/compiler/tikcpp/ascendc_kernel_cmake)

project(kernel_samples LANGUAGES ASC CXX)

set(CASE_TYPE 0 CACHE STRING "Case type to compile (0-7)")

add_executable(demo
    add.asc
)

set(NPU_ARCH "" CACHE STRING "NPU architecture: 2201(A3) or 3510(A5)")

if(NOT NPU_ARCH)
    message(FATAL_ERROR "NPU_ARCH is required. Usage: cmake -DNPU_ARCH=2201(A3) or -DNPU_ARCH=3510(A5)")
endif()

if(NOT NPU_ARCH EQUAL 2201 AND NOT NPU_ARCH EQUAL 3510)
    message(FATAL_ERROR "Unsupported NPU_ARCH=${NPU_ARCH}, only 2201(A3) and 3510(A5) are supported")
endif()

target_compile_definitions(demo PRIVATE
    CASE_TYPE=${CASE_TYPE}
    NPU_ARCH=${NPU_ARCH}
)

target_link_libraries(demo PRIVATE
    tiling_api
    register
    platform
    m
    dl
)

# ======================================================================================
# NPU 编译选项配置
#
# 说明：
#   - 需根据实际部署的 NPU 硬件架构选择对应的 `npu-arch` 参数。
# ======================================================================================

target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-${NPU_ARCH}>
)