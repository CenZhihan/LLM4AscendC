#!/usr/bin/env bash
# 本机 / 昇腾容器：顺序评测 output/kernelbench165_txt 下全部 *.txt（170 个算子）。
# 与 kernelbench 脚本结构相同，仅切换目标目录和增加 profiler 支持。
#
# 用法（在仓库根目录）：
#   bash scripts/pj-lab/run_kernelbench165_local.sh              # 默认 smart + full，扫描全部算子
#   bash scripts/pj-lab/run_kernelbench165_local.sh force full   # clean-policy + mode
#   bash scripts/pj-lab/run_kernelbench165_local.sh force full PipeUtilization 3
#     # 第3/4参数分别传给 msprof op 的 --aic-metrics / --launch-count
#   bash scripts/pj-lab/run_kernelbench165_local.sh force full PipeUtilization 3 4
#     # 第5参数为并行 worker 数（需多 NPU 时配合 --npu）
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
PROFILER_AIC_METRICS="${3:-PipeUtilization}"
PROFILER_LAUNCH_COUNT="${4:-1}"
WORKERS="${5:-1}"

if [[ -z "${LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH:-}" ]]; then
  export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="/workspace/ascend_custom_opp"
fi

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck source=/dev/null
  set +u
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  set -u
fi

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  set +u
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate multi-kernel-bench
  set -u
fi

exec python3 tools/eval_operator.py \
  --txt-dir output/kernelbench165_txt \
  --clean-policy "${CLEAN_POLICY}" \
  --mode "${MODE}" \
  --with-profiler \
  --profiler-aic-metrics "${PROFILER_AIC_METRICS}" \
  --profiler-launch-count "${PROFILER_LAUNCH_COUNT}" \
  --workers "${WORKERS}"
