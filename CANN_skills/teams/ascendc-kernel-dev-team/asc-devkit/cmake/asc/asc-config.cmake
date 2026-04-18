# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ERROR: ASCEND_HOME_PATH environment variable is not set! source set_env.sh in cann install directory first.")
else()
    if(NOT EXISTS "$ENV{ASCEND_HOME_PATH}")
        message(FATAL_ERROR "ERROR: ASCEND_HOME_PATH directory does not exist!")
    endif()
endif()

if(_ASC_MODULE_LOADED)
    return()
endif()
set(_ASC_MODULE_LOADED FALSE)
include($ENV{ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/config.cmake)
include($ENV{ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/func.cmake)
include($ENV{ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/intf.cmake)

# plugin support ASC language
list(APPEND CMAKE_MODULE_PATH "$ENV{ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/asc_modules")
include($ENV{ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/asc_modules/FindASC.cmake)
set(_ASC_MODULE_LOADED TRUE)
