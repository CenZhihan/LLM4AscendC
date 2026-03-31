# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


cmake_minimum_required(VERSION 3.16.0)
project(static_aclnn_invocation_superbuild)
set(CUSTOMIZE_INSTALL_DIR ${CMAKE_BINARY_DIR}/customize-install)
set(CUSTOMIZE_OPS_NAME my_ops)

include(ExternalProject)
ExternalProject_Add(customize_op
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/op
    BINARY_DIR ${CMAKE_BINARY_DIR}/op-build
    INSTALL_DIR ${CUSTOMIZE_INSTALL_DIR}
    CMAKE_ARGS
        -Dpackage_name=${CUSTOMIZE_OPS_NAME}
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
)

ExternalProject_Add(execute_add_app
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/app
    BINARY_DIR ${CMAKE_BINARY_DIR}/app-build
    CMAKE_ARGS
        -Dpackage_name=${CUSTOMIZE_OPS_NAME}
        -DCMAKE_PREFIX_PATH=${CUSTOMIZE_INSTALL_DIR}
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}
    DEPENDS customize_op
)
