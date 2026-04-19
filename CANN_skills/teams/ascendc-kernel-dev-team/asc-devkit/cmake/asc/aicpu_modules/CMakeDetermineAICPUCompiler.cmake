# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" SYSTEM_LOWER_PROCESSOR)
if(EXISTS $ENV{ASCEND_HOME_PATH}/${SYSTEM_LOWER_PROCESSOR}-linux/ccec_compiler/bin)
    set(AICPU_COMPILER_PATH $ENV{ASCEND_HOME_PATH}/${SYSTEM_LOWER_PROCESSOR}-linux/ccec_compiler/bin)
else()
    set(AICPU_COMPILER_PATH "$ENV{ASCEND_HOME_PATH}/compiler/ccec_compiler/bin")
endif()
find_program(CMAKE_AICPU_COMPILER 
    NAMES "bisheng" 
    PATHS "${AICPU_COMPILER_PATH}" 
    DOC "AICPU Compiler"
)

mark_as_advanced(CMAKE_AICPU_COMPILER)

message(STATUS "Detecting AICPU compiler: " ${CMAKE_AICPU_COMPILER})

set(CMAKE_AICPU_SOURCE_FILE_EXTENSIONS aicpu)
set(CMAKE_AICPU_COMPILER_ENV_VAR "AICPU")

# configure all variables set in this file
configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeAICPUCompiler.cmake.in
	${CMAKE_PLATFORM_INFO_DIR}/CMakeAICPUCompiler.cmake
    @ONLY
)
