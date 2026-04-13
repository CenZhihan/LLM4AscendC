#!/usr/bin/env bash
# 仅评测本轮「第三大类(14个)」算子：
# - 输入目录：output/kernelbench165_txt/category3_strong_constraint_14
# - 配置：4 workers + 4 张 NPU
# - LLM4ASCENDC_REF_ON_CPU=1 -> reference 在 CPU，自定义算子仍在 NPU
set -euo pipefail

export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/ascend_custom_opp"
export LLM4ASCENDC_REF_ON_CPU=1

REPO_ROOT="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/LLM4AscendC"
cd "$REPO_ROOT"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

exec python3 tools/eval_operator.py \
  --txt-dir output/kernelbench165_txt/category3_strong_constraint_14 \
  --workers 4 \
  --npu 4 \
  --clean-policy force \
  --mode full
