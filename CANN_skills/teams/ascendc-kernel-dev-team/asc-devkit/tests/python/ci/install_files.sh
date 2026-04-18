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

TOP_DIR=$1
INSTALL_PATH=$2
DST_DIR=$3

DST_PATH=${INSTALL_PATH}/${DST_DIR}
echo "TOP_DIR is $TOP_DIR"
echo "INSTALL_PATH is $INSTALL_PATH"
echo "DST_PATH is ---- $DST_PATH"

[ -n "$DST_PATH" ] && rm -rf $DST_PATH
mkdir -p $DST_PATH

# copy basic api
mkdir -p $DST_PATH/
cp -rf ${TOP_DIR}/impl ${DST_PATH}/impl
cp -rf ${TOP_DIR}/include ${DST_PATH}/include