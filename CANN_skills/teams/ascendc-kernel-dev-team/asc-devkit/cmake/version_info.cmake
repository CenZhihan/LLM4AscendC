# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

function(parse_semantic_version version_str out_major out_minor out_patch out_prerelease out_version_num out_timestamp)
    # 解析主/次/补丁版本（格式：MAJOR.MINOR.PATCH）
    string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(-.*)?" VERSION_MATCH "${version_str}")
    if(NOT VERSION_MATCH)
        message(FATAL_ERROR "Invalid version format: ${version_str}. Expected x.y.z[-prerelease]")
    endif()

    set(major ${CMAKE_MATCH_1})
    set(minor ${CMAKE_MATCH_2})
    set(patch ${CMAKE_MATCH_3})
    set(prerelease ${CMAKE_MATCH_4})

    # 移除预发布版本前的 '-'
    string(REGEX REPLACE "^-" "" prerelease "${prerelease}")

    # 计算 base_value = major*10^7 + minor*10^5 + patch*10^3
    math(EXPR base_value "${major} * 10000000 + ${minor} * 100000 + ${patch} * 1000")
    set(version_num ${base_value})

    # 处理预发布版本（alpha/beta/rc）
    if(prerelease)
        string(REGEX MATCH "^(alpha|beta|rc)\\.([0-9]+)" PR_MATCH "${prerelease}")
        if(NOT PR_MATCH)
            message(FATAL_ERROR "Invalid prerelease format: ${prerelease}. Use 'alpha.N' / 'beta.N' / 'rc.N'")
        endif()

        set(pr_type ${CMAKE_MATCH_1})
        set(pr_num ${CMAKE_MATCH_2})

        # 预发布权重（alpha 最低，rc 最高）
        if(pr_type STREQUAL "alpha")
            set(weight 300)
        elseif(pr_type STREQUAL "beta")
            set(weight 200)
        elseif(pr_type STREQUAL "rc")
            set(weight 100)
        endif()

        # 计算带预发布的版本数值
        math(EXPR version_num "${base_value} - ${weight} + ${pr_num}")
    endif()

    execute_process(
        COMMAND date +%Y%m
        OUTPUT_VARIABLE timestamp
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # 输出变量（传递到函数外部）
    set(${out_major} ${major} PARENT_SCOPE)
    set(${out_minor} ${minor} PARENT_SCOPE)
    set(${out_patch} ${patch} PARENT_SCOPE)
    set(${out_prerelease} "${prerelease}" PARENT_SCOPE)
    set(${out_version_num} ${version_num} PARENT_SCOPE)
    set(${out_timestamp} ${timestamp} PARENT_SCOPE)
endfunction()

# --------------------------
# 3. 调用函数解析版本号
# --------------------------
parse_semantic_version(
    ${CANN_VERSION_asc-devkit_VERSION}
    MAJOR
    MINOR
    PATCH
    PRERELEASE
    VERSION_NUM
    TIMESTAMP
)

# -------------------------
# 4. 生成 asc_devkit_version.h
# -------------------------
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.h.in
    ${CPACK_CMAKE_BINARY_DIR}/scripts/package/asc-devkit/asc_devkit_version.h
    @ONLY
)

set(VERSION_FILES
    ${CPACK_CMAKE_BINARY_DIR}/scripts/package/asc-devkit/asc_devkit_version.h
)

install(FILES ${VERSION_FILES}
    DESTINATION asc-devkit
)