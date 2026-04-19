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
project(static_aclnn_execute_add_app)
find_package(${package_name} REQUIRED)

add_executable(execute_add_op
    main.cpp
)

target_link_libraries(execute_add_op PRIVATE
    nnopbase
    rt2_registry
    acl_rt
    ${package_name}::${package_name}
)

set_target_properties(execute_add_op PROPERTIES
    BUILD_RPATH ${CMAKE_PREFIX_PATH}/lib
    INSTALL_RPATH ${CMAKE_PREFIX_PATH}/lib
)

install(TARGETS execute_add_op
    RUNTIME DESTINATION .
)
