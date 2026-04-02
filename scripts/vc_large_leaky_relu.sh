#!/usr/bin/env bash
# MKB 大形状：batch=4096, dim=393216（见 output/leaky_relu.txt / vendor/mkb/reference 注释）
set -euo pipefail
REPO_ROOT="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/LLM4AscendC"
export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/ascend_custom_opp"
cd "$REPO_ROOT"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
exec python3 tools/eval_operator.py --txt output/leaky_relu.txt --mode full
