#!/usr/bin/env bash
# 仅评测 output/kernelbench165_txt/pybind_reference_aligned 下 64 个已对齐 reference 的 txt（1 NPU + 5 workers）
set -euo pipefail

export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/ascend_custom_opp"

REPO_ROOT="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/LLM4AscendC"
cd "$REPO_ROOT"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

exec python3 tools/eval_operator.py \
  --txt-dir output/kernelbench165_txt/pybind_reference_aligned \
  --workers 5 \
  --npu 1 \
  --clean-policy force \
  --mode full
