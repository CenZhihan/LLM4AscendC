#!/usr/bin/env bash
# 并行评测 output/kernelbench165_txt/test_10_new_reference 下 *.txt（10 个算子 + 新 reference）
# 配置：4 workers + 4 张 NPU（worker i -> 物理卡 i%4）
# --clean-policy force：每次构建前清空 workspace/pybind
#
# 与 run_kernelbench165_workers4_npu4.sh 一致，仅 --txt-dir 不同。
set -euo pipefail

export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/ascend_custom_opp"

REPO_ROOT="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/LLM4AscendC"
cd "$REPO_ROOT"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

exec python3 tools/eval_operator.py \
  --txt-dir output/kernelbench165_txt/test_10_new_reference \
  --workers 4 \
  --npu 4 \
  --clean-policy force \
  --mode full
