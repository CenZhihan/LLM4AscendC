#!/usr/bin/env bash
# 在本机 Docker 容器内跑并行评测（gpt-4o matmul：matmul_gpt4o_run0）
#
# 说明：
#   workers 与 npu 通过 --workers/--npu 传给 tools/eval_operator.py。
#   多卡绑定逻辑：device_id = worker_id % npu_count，并在 worker 子进程内设置 ASCEND_VISIBLE_DEVICES。
#
# 用法（容器内执行）：
#   bash scripts/xlance-lab/run_matmul_gpt4o_workers_npu.sh 4 4
# 或：
#   bash scripts/xlance-lab/run_matmul_gpt4o_workers_npu.sh <workers> <npu>
#
# 如果参数不传，默认 workers=4, npu=4。
set -euo pipefail

workers="${1:-4}"
npu="${2:-4}"

export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/ascend_custom_opp"

REPO_ROOT="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/LLM4AscendC"
cd "$REPO_ROOT"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

exec python3 tools/eval_operator.py \
  --txt-dir output/gpt-4o_selected_shot/matmul_gpt4o_run0 \
  --workers "${workers}" \
  --npu "${npu}" \
  --mode full

