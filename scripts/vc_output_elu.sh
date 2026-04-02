#!/usr/bin/env bash
# 评测 output/elu.txt（MKB op key: elu）；集群上自定义 OPP 目录见 LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH
set -euo pipefail
REPO_ROOT="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/LLM4AscendC"
export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/ascend_custom_opp"
cd "$REPO_ROOT"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
exec python3 tools/eval_operator.py --txt output/elu.txt --mode full
