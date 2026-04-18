# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

add_library(intf_llt_pub_basic INTERFACE)

target_include_directories(intf_llt_pub_basic INTERFACE)

target_compile_definitions(intf_llt_pub_basic INTERFACE
    CFG_BUILD_DEBUG
    _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(intf_llt_pub_basic INTERFACE
    -g
    -w
    $<$<BOOL:${ENABLE_GCOV}>:-fprofile-arcs -ftest-coverage>
    $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -fno-omit-frame-pointer -static-libasan -fsanitize=undefined -static-libubsan -fsanitize=leak -static-libtsan>
    -fPIC
    -pipe
    -Werror
)

target_link_options(intf_llt_pub_basic INTERFACE
    $<$<BOOL:${ENABLE_GCOV}>:-fprofile-arcs -ftest-coverage>
    $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -static-libasan -fsanitize=undefined  -static-libubsan -fsanitize=leak -static-libtsan>
)

target_link_directories(intf_llt_pub_basic INTERFACE)

target_link_libraries(intf_llt_pub_basic INTERFACE
    gtest
    mockcpp_static
    -lpthread
    $<$<BOOL:${ENABLE_GCOV}>:-lgcov>
)

add_library(kernel_tiling INTERFACE)
target_include_directories(kernel_tiling INTERFACE
    ${CMAKE_BINARY_DIR}
)

# gen tiling header file
set(GEN_TILING_SCRIPT ${ASCENDC_DIR}/impl/adv_api/cmake/scripts/gen_kernel_tiling_data_def.py)
set(TILING_DEF_DIR ${ASCENDC_DIR}/include/adv_api)
set(KERNEL_TILING_FILE ${CMAKE_BINARY_DIR}/kernel_tiling/kernel_tiling.h)

if (NOT EXISTS "${KERNEL_TILING_FILE}")
    add_custom_command(
        OUTPUT ${KERNEL_TILING_FILE}
        COMMAND ${Python3_EXECUTABLE} ${GEN_TILING_SCRIPT} ${TILING_DEF_DIR} ${KERNEL_TILING_FILE}
        DEPENDS ${GEN_TILING_SCRIPT} ${TILING_DEF_DIR}
        COMMENT "Generating kernel tiling configuration header"
    )
    add_custom_target(generate_kernel_headers ALL	 
        DEPENDS ${KERNEL_TILING_FILE}	 
        COMMENT "Ensuring kernel headers are generated"	 
    )
    set_source_files_properties(
        ${KERNEL_TILING_FILE}
        PROPERTIES GENERATED TRUE
    )
    add_dependencies(kernel_tiling generate_kernel_headers)
endif()