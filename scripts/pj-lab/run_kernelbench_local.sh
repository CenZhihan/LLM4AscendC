#!/usr/bin/env bash
# 本机 / 昇腾容器：顺序评测 output/kernelbench 下全部 *.txt（如 relu、gelu）。
# 与集群脚本不同：不写死 /aistor 路径；仓库根目录由本脚本位置推导。
#
# 用法（在仓库根目录）：
#   bash scripts/pj-lab/run_kernelbench_local.sh              # 默认 smart + full
#   bash scripts/pj-lab/run_kernelbench_local.sh force full   # clean-policy + mode
#
# 环境：
#   - 若未设置 LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH，默认使用 /workspace/ascend_custom_opp
#     （与 tools/common/env.py 一致；无 /workspace 时可自行 export 到可写目录）
#   - 建议先：source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CLEAN_POLICY="${1:-smart}"
MODE="${2:-full}"

if [[ -z "${LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH:-}" ]]; then
  export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="/workspace/ascend_custom_opp"
fi

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

exec python3 tools/eval_operator.py \
  --txt-dir output/kernelbench \
  --clean-policy "${CLEAN_POLICY}" \
  --mode "${MODE}"
