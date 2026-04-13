#!/usr/bin/env bash
# 补跑缺 result 的两个算子（见 output/agent_add_shot_tools_kb_only_missing2，内为指向全量目录的符号链接）
# 单进程顺序跑：1 worker（默认），单卡即可；--clean-policy force
#
# vc 示例：-n 1 -g 1 -m 48G ...
set -euo pipefail

export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/ascend_custom_opp"

REPO_ROOT="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/LLM4AscendC"
cd "$REPO_ROOT"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

exec python3 tools/eval_operator.py \
  --txt-dir output/agent_add_shot_tools_kb_only_missing2 \
  --workers 1 \
  --clean-policy force \
  --mode full
