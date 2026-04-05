#!/bin/bash
# 动态获取脚本所在目录，而非硬编码 /workspace
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUSTOM_OPP_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

export ASCEND_CUSTOM_OPP_PATH=$CUSTOM_OPP_ROOT:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=$CUSTOM_OPP_ROOT/op_api/lib/:${LD_LIBRARY_PATH}