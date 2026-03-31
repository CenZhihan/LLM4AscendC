# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
project(ascendc_python_tests)
cmake_minimum_required(VERSION 3.16.0)
add_subdirectory(ascendc_hidden)

set(INSTALL_PATH ${CMAKE_BINARY_DIR})
set(DST_PATH adapter_ut)

add_custom_target(generate_hidden_dat_file_ut ALL
    COMMAND code_hidden ${ASCENDC_DIR}/tests/python/adapter/stub_kernels ${ASCENDC_DIR}/tests/python/adapter/stub_kernels/ascendc_impl.dat
    DEPENDS code_hidden
)

add_custom_target(install_adapter_api_ut ALL
    COMMAND bash ${ASCENDC_DIR}/tests/python/ci/install_files.sh ${ASCENDC_DIR} ${INSTALL_PATH} ${DST_PATH}
)

run_python_llt_test(
    TARGET ascendc_pyut_aclrt_launch_kernel
    SRC_FILES_DIR ${ASCENDC_DIR}/cmake/
    TEST_FILES_DIR ${ASCENDC_DIR}/tests/python/aclrt_launch_kernel/
    PRE_TEST_COMMAND
    COVERAGERC_DIR
)

run_python_llt_test(
    TARGET ascendc_pyut_asc_compile_common
    SRC_FILES_DIR ${ASCENDC_DIR}/tools/build/asc_op_compile_base/common/
    TEST_FILES_DIR ${ASCENDC_DIR}/tests/python/asc_compile_common/
    PRE_TEST_COMMAND
    COVERAGERC_DIR
)

add_dependencies(ascendc_pyut_asc_compile_common
    asc_platform_test
)

run_python_llt_test(
    TARGET ascendc_pyut_asc_op_compiler
    SRC_FILES_DIR ${ASCENDC_DIR}/tools/build/asc_op_compile_base/asc_op_compiler/
    TEST_FILES_DIR ${ASCENDC_DIR}/tests/python/asc_op_compiler/
    ENV_FILE ${ASCENDC_DIR}/tests/python/ci/set_python_env.sh
    ENV_PARAMS ${ASCENDC_DIR}\ py_ut \ 910
    PRE_TEST_COMMAND
    COVERAGERC_DIR
)

add_dependencies(ascendc_pyut_asc_op_compiler
    register
    kernel_tiling
    install_adapter_api_ut
    generate_hidden_dat_file_ut
    asc_platform_test
)

run_python_llt_test(
    TARGET ascendc_pyut_compile_trace_log
    SRC_FILES_DIR ${ASCENDC_DIR}/tools/scripts/compile_trace_log
    TEST_FILES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/compile_trace_log
    EXPORT_PYTHONPATH ${CMAKE_CURRENT_SOURCE_DIR}/stub/
    PRE_TEST_COMMAND
    COVERAGERC_DIR
)

set(OPC_BASE "${CMAKE_CURRENT_SOURCE_DIR}")
file(MAKE_DIRECTORY ${OPC_BASE}/asc_opc/output)
file(WRITE ${OPC_BASE}/asc_opc/stub/files/kernel_meta/MatMul_build_res.o "")

run_python_llt_test(
    TARGET asc_opc_unittest
    SRC_FILES_DIR ${ASCENDC_DIR}/tools/build/asc_opc/python
    TEST_FILES_DIR ${ASCENDC_DIR}/tests/python/asc_opc
    EXPORT_PYTHONPATH  ${ASCENDC_DIR}/tools/build/asc_opc
    ENV_FILE ${ASCENDC_DIR}/tests/python/asc_opc/common/ci/set_python_env.sh
    ENV_PARAMS ${ASCENDC_DIR}\ py_ut\ 310
    COVERAGERC_DIR
)

add_dependencies(asc_opc_unittest
    register
    kernel_tiling
    install_adapter_api_ut
    generate_hidden_dat_file_ut
    asc_platform_test
)

# clear generated files
add_custom_target(run_all_tests_and_clean ALL
    # dependency asc_platform_test
    COMMAND rm -f ${ASCENDC_DIR}/tools/build/asc_op_compile_base/c_api/libasc_platform.so
    # dependency generate_hidden_dat_file_ut
    COMMAND rm -f ${ASCENDC_DIR}/tests/python/adapter/stub_kernels/ascendc_impl.dat
    # created by tests/python/CMakeLists.txt
    COMMAND rm -f ${OPC_BASE}/asc_opc/stub/files/kernel_meta/MatMul_build_res.o
    # created by tests/python/asc_opc/common/ci/set_python_env.sh
    COMMAND rm -rf ${ASCENDC_DIR}/out/
    # created by run_python_llt_test
    COMMAND rm -f ${ASCENDC_DIR}/null
    DEPENDS ascendc_pyut_asc_compile_common ascendc_pyut_asc_op_compiler asc_opc_unittest ascendc_pyut_compile_trace_log ascendc_pyut_aclrt_launch_kernel 
    COMMENT "clean generated test files"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

