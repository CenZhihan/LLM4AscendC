#!/usr/bin/env bash
# gpt-4o matmul（matmul_gpt4o_run0）：5 workers、单卡（--npu 1）、full、force 清理构建目录
#
#   bash scripts/xlance-lab/run_matmul_gpt4o_workers5_npu1_force.sh
set -euo pipefail

export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/ascend_custom_opp"

REPO_ROOT="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/LLM4AscendC"
cd "$REPO_ROOT"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

exec python3 tools/eval_operator.py \
  --txt-dir output/gpt-4o_selected_shot/matmul_gpt4o_run0 \
  --workers 5 \
  --npu 1 \
  --clean-policy force \
  --mode full
