# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

if(CUSTOM_ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH ${CUSTOM_ASCEND_CANN_PACKAGE_PATH})
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
elseif(DEFINED ENV{ASCEND_OPP_PATH})
    get_filename_component(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_OPP_PATH}/.." ABSOLUTE)
else()
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann")
endif()

if (NOT EXISTS "${ASCEND_CANN_PACKAGE_PATH}")
    message(FATAL_ERROR "${ASCEND_CANN_PACKAGE_PATH} does not exist, please install the cann package and set environment variables.")
endif()

if (CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/_CPack_Packages/makeself_staging" CACHE STRING "path for install()" FORCE)
endif()

set(HI_PYTHON "python3" CACHE STRING "python executor")
set(PRODUCT_SIDE host)
set(COMPILE_BASE_ON_SUBGROUP OFF BOOL)

if (ENABLE_TEST)
    set(CMAKE_SKIP_RPATH FALSE)
else()
    set(CMAKE_SKIP_RPATH TRUE)
endif()

if (CMAKE_BUILD_TYPE STREQUAL Release)
    set(DEFAULT_BUILD_TYPE "Release")
elseif (CMAKE_BUILD_TYPE STREQUAL Debug)
    set(DEFAULT_BUILD_TYPE "Debug")
else()
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the build type: Release/Debug" FORCE)
endif()

get_filename_component(ASCENDC_API_ADV_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)