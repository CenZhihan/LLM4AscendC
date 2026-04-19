# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


cmake_minimum_required(VERSION 3.16.0)
project(parallel_ops_package_proj)
include(ExternalProject)

set(ASCEND_COMPUTE_UNIT ascend910b ascend910_93 ascend950)
set(PACKAGE_TYPE "RUN")
set(INSTALL_TARGET "package")

ExternalProject_Add(add_custom 
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/add_custom 
    CMAKE_CACHE_ARGS
        -DASCEND_COMPUTE_UNIT:STRING=${ASCEND_COMPUTE_UNIT}
    CMAKE_ARGS
        -Dvendor_name=add_custom 
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/add_custom 
        -DPACKAGE_TYPE=${PACKAGE_TYPE} 
    BUILD_COMMAND 
        ${CMAKE_COMMAND} --build <BINARY_DIR> --target binary 
    COMMAND 
        ${CMAKE_COMMAND} --build <BINARY_DIR> --target ${INSTALL_TARGET} 
    INSTALL_COMMAND "" 
)

ExternalProject_Add(leaky_relu_custom 
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/leaky_relu_custom
    CMAKE_CACHE_ARGS
        -DASCEND_COMPUTE_UNIT:STRING=${ASCEND_COMPUTE_UNIT}
    CMAKE_ARGS
        -Dvendor_name=leaky_relu_custom 
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/leaky_relu_custom
        -DPACKAGE_TYPE=${PACKAGE_TYPE} 
    BUILD_COMMAND 
        ${CMAKE_COMMAND} --build <BINARY_DIR> --target binary 
    COMMAND 
        ${CMAKE_COMMAND} --build <BINARY_DIR> --target ${INSTALL_TARGET} 
    INSTALL_COMMAND "" 
)
