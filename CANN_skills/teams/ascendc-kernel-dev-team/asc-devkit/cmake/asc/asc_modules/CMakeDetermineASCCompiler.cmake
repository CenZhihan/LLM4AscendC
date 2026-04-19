# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# CMakeDetermineASCCompiler.cmake is used to initialize ASC-related variables.
# And this file will not be triggered again during incremental compilation.
# 1. Find compiler for ASC extension
message(STATUS "System processer: ${CMAKE_SYSTEM_PROCESSOR}")
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ASCEND_CANN_PACKAGE_LINUX_PATH $ENV{ASCEND_HOME_PATH}/x86_64-linux)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
    set(ASCEND_CANN_PACKAGE_LINUX_PATH $ENV{ASCEND_HOME_PATH}/aarch64-linux)
else ()
    message(FATAL_ERROR "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

find_program(CMAKE_ASC_COMPILER NAMES "bisheng" PATHS "${ASCEND_CANN_PACKAGE_LINUX_PATH}/ccec_compiler/bin/" "$ENV{PATH}" "$ENV{ASCEND_HOME_PATH}" DOC "ASC Compiler")

mark_as_advanced(CMAKE_ASC_COMPILER)
message(STATUS "CMAKE_ASC_COMPILER: " ${CMAKE_ASC_COMPILER})

set(CMAKE_ASC_SOURCE_FILE_EXTENSIONS asc)    # Specify .asc as the supported file extension
set(CMAKE_ASC_COMPILER_ENV_VAR "ASC")        # Name the language ASC

# Sequence for the first compilation: CMakeDetermineASCCompiler.cmake -> CMakeASCInformation.cmake
# Incremental compilation: CMakeASCInformation.cmake
find_program(CMAKE_ASC_LLD_LINKER NAMES "ld.lld" PATHS "${ASCEND_CANN_PACKAGE_LINUX_PATH}/ccec_compiler/bin/" DOC "ASC ld.lld Linker" NO_DEFAULT_PATH)

message(STATUS "ASCEND_CANN_PACKAGE_LINUX_PATH: " ${ASCEND_CANN_PACKAGE_LINUX_PATH})
message(STATUS "CMAKE_ASC_LLD_LINKER: ${CMAKE_ASC_LLD_LINKER}")

# configure all variables set in this file
configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeASCCompiler.cmake.in
    ${CMAKE_PLATFORM_INFO_DIR}/CMakeASCCompiler.cmake
    @ONLY
)