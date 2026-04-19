#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set -e

ASCEND_CANN_PACKAGE_PATH=$1
PACKAGE_NAME=$2
CURRENT_VERSION_INFO_FILE=$3
CANN_VERSION_INFO_FILE=${ASCEND_CANN_PACKAGE_PATH}/${PACKAGE_NAME}/version.info

function main()
{
    if [ ! -f "${CURRENT_VERSION_INFO_FILE}" ];then
        echo "Error: ${CURRENT_VERSION_INFO_FILE} does not exist."
        exit 1
    fi

    if [ ! -f "${CANN_VERSION_INFO_FILE}" ];then
        echo "Error: ${CANN_VERSION_INFO_FILE} does not exist, please check whether the cann package is installed."
        exit 1
    fi

    cann_version=$(grep -w "Version"  ${CANN_VERSION_INFO_FILE} | cut -d"=" -f2)
    local_version=$(grep -w "Version"  ${CURRENT_VERSION_INFO_FILE} | cut -d"=" -f2)
    
    _cann_version=$(echo ${cann_version} | cut -d'.' -f1-4)
    _local_version=$(echo ${local_version} | cut -d'.' -f1-4)

    if [ "$_cann_version" != "$_local_version" ]; then
        echo "Error: The version number of the current code is ${local_version}, and the version number of the cann package used is ${cann_version}. Please install version ${local_version} of the cann package."
        exit 1
    fi

    echo "${cann_version}"
}

main






