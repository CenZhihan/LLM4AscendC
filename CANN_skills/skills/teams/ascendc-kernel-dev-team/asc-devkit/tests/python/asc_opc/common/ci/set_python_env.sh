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

#current dir
cur_dir=$1
test_type=$2
product=$3
product_model=${product}
cmake_binary_dir=$4
version=$5

echo "$(date +"%Y-%m-%d %H:%M:%S") exec [set_python_env.sh] $cur_dir $test_type $product $cmake_binary_dir version: ${version}."

if [ "onetrack" = "${product}" ]; then
    product_model="mini"
fi

if [ "Xpy_ut" = "X$test_type" ];then
    test_type="ut"
    # fix set ddk version issue
    export PYTHONPATH=$cur_dir/llt/opc/ut/testcase:$PYTHONPATH
    export PYTHONPATH=$cur_dir/llt/opc/st/testcase:$PYTHONPATH
elif [ "Xpy_st" = "X$test_type" ];then
    test_type="st"
fi

# mini
TOOLCHAIN_HOME="build/bin/toolchain/x86/ubuntu/ccec_libs/ccec_x86_ubuntu_18_04_adk"
if [ "v300" = "${version}" ]; then
    TOOLCHAIN_HOME="build/bin/ccec_for_milan"
fi

#PYTHONPATH
export PYTHONPATH=$cur_dir/atc/opcompiler/opc/python/opc_tool:$PYTHONPATH
export PYTHONPATH=$cur_dir/atc/opcompiler/ascendc_compiler/framework/tools/build/asc_opc/python/asc_opc_tool:$PYTHONPATH
export PYTHONPATH=$cur_dir/llt/atc/opcompiler/opc/stub:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:cur_dir/open_source/tvm/tiling/:$cur_dir/open_source/tvm/tiling/auto_tune/:$cur_dir/asl/ops/cann/ops/built-in/tbe
export PYTHONPATH=$PYTHONPATH:$cur_dir/open_source/tvm/python:$cur_dir/open_source/tvm/python/te:$cur_dir/open_source/tvm/topi/python

#PATH
#export PATH=$cur_dir/${TOOLCHAIN_HOME}/bin:$PATH
#echo "PATH: ${PATH}"
#whereis ccec
#which ccec

#LD_LIBRARY_PATH

#export LD_LIBRARY_PATH=$cmake_binary_dir/no_asan/:$cmake_binary_dir/asl/aoetools/opat/auto_tune/auto_tiling/lib/:$cmake_binary_dir/llt/tensor_engine/ut/:$cmake_binary_dir/llt/abl/platform/:$cmake_binary_dir/abl/platform/:$cmake_binary_dir/abl/slog/slog/slog/host/:$cmake_binary_dir/c_sec_build-prefix/src/c_sec_build-build/:$cmake_binary_dir/open_source/tvm/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=$cur_dir/tmp/llt_python-prefix/src/llt_python-build/llt/tensor_engine/ut/:$cur_dir/tmp/llt_python-prefix/src/llt_python-build/llt/abl/platform/:$cur_dir/tmp/llt_python-prefix/src/llt_python-build/abl/platform/:$cur_dir/tmp/llt_python-prefix/src/llt_python-build/abl/slog/slog/slog/host/:$cur_dir/tmp/llt_python-prefix/src/llt_python-build/c_sec_build-prefix/src/c_sec_build-build/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=$cmake_binary_dir/llt/abl/slog:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=$cur_dir/tmp/securec_llt_gccnative-prefix/src/securec_llt_gccnative-build/:$LD_LIBRARY_PATH
#echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

#TVM_AICPU_INCLUDE_PATH and TVM_AICPU_LIBRARY_PATH
#export ASCEND_OPP_PATH=$cur_dir/out/onetrack/llt/$test_type/obj
