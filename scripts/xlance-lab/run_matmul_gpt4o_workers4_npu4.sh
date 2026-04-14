#!/usr/bin/env bash
# 在本机 Docker 容器内跑并行评测（gpt-4o matmul：matmul_gpt4o_run0）
# 配置：--workers 4 + --npu 4（物理卡 0..3 分配给 4 个 worker）
#
# 用法：
#   bash scripts/xlance-lab/run_matmul_gpt4o_workers4_npu4.sh
#
# 如遇到容器内找不到 set_env.sh，请按其它脚本的方式自行补环境变量。
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
  --workers 4 \
  --npu 4 \
  --mode full

