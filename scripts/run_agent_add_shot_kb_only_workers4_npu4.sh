#!/usr/bin/env bash
# 并行评测 output/agent_add_shot_tools_kb_only 下全部 *.txt（约 300 个）
# 配置：4 workers + 4 张 NPU（device_id = worker_id % 4）
# --clean-policy force：每次构建前清空 workspace/pybind
#
# 集群提交示例：vc submit ... -g 4 -m 512G ...
set -euo pipefail

export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/ascend_custom_opp"

REPO_ROOT="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/LLM4AscendC"
cd "$REPO_ROOT"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

exec python3 tools/eval_operator.py \
  --txt-dir output/agent_add_shot_tools_kb_only \
  --workers 4 \
  --npu 4 \
  --clean-policy force \
  --mode full
