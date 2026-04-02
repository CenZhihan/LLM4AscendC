#!/usr/bin/env bash
# kernelbench165：x 固定 [16384,4096]，w [4096,4096] 等大张量，易触发编译/运行内存压力
set -euo pipefail
REPO_ROOT="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/LLM4AscendC"
export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/ascend_custom_opp"
cd "$REPO_ROOT"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
exec python3 tools/eval_operator.py --txt output/kernelbench165_txt/matmul_scaling_residual_add.txt --mode full
