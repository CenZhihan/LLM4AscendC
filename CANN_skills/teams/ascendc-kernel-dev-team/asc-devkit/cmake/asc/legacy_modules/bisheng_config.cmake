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
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/ccec_compiler AND SYSTEM_LOWER_PROCESSOR STREQUAL "x86_64")
    set(ASCENDC_DEVKIT_PATH ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ccec_compiler AND SYSTEM_LOWER_PROCESSOR STREQUAL "aarch64")
    set(ASCENDC_DEVKIT_PATH ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/ccec_compiler)
    set(ASCENDC_DEVKIT_PATH ${ASCEND_CANN_PACKAGE_PATH}/tools)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/ccec_compiler)
    set(ASCENDC_DEVKIT_PATH ${ASCEND_CANN_PACKAGE_PATH}/compiler)
else()
    set(ASCENDC_DEVKIT_PATH ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit)
endif()


set(CCEC_PATH           ${ASCENDC_DEVKIT_PATH}/ccec_compiler/bin)
set(CMAKE_C_COMPILER    "${CCEC_PATH}/bisheng")
set(CMAKE_CXX_COMPILER  "${CCEC_PATH}/bisheng")
set(CMAKE_LINKER        "${CCEC_PATH}/ld.lld")
set(CMAKE_AR            "${CCEC_PATH}/llvm-ar")
set(CMAKE_STRIP         "${CCEC_PATH}/llvm-strip")
set(CMAKE_OBJCOPY       "${CCEC_PATH}/llvm-objcopy")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_SKIP_RPATH TRUE)
