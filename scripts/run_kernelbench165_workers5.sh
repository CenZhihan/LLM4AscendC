#!/usr/bin/env bash
# 集群 / 本机：并行评测 output/kernelbench165_txt 下全部 *.txt
# 要求：1 张 NPU、5 个 worker（同卡并行，--npu 1）
# --clean-policy force：每次构建前清空 workspace/pybind，避免沿用上次不完整状态
#
# vc 示例见 README 或仓库内说明；工作目录须为 LLM4AscendC 根目录。
set -euo pipefail

export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/ascend_custom_opp"

REPO_ROOT="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/LLM4AscendC"
cd "$REPO_ROOT"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

exec python3 tools/eval_operator.py \
  --txt-dir output/kernelbench165_txt \
  --workers 5 \
  --npu 1 \
  --clean-policy force \
  --mode full
