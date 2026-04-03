#!/usr/bin/env bash
# 在 Docker 容器内并行评测 gpt-4o matmul（17 个 txt），3 workers。
#
# 宿主机先进入容器（容器 ID 若变更请改此处）：
#   docker exec -it b31f8ebb90b9 bash
# 再在容器内 cd 到 LLM4AscendC 根目录后执行：
#   bash scripts/run_matmul_gpt4o_parallel3.sh
set -euo pipefail
export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/ascend_custom_opp"
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
exec python3 tools/eval_operator.py \
  --txt-dir output/gpt-4o_selected_shot/matmul_gpt4o_run0 \
  --workers 3 \
  --mode full
