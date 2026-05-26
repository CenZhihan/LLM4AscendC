#!/usr/bin/env bash
# CUDA-Agent torch#1 (204 rows): multi-round agent + eval, same stack as activation run3.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${ROOT}/cuda_torch1_mem_run2.log"

exec > >(tee -a "${LOG}") 2>&1

echo "=== $(date -Is) start cuda_torch1_mem_run2 ==="

unset https_proxy http_proxy all_proxy HTTPS_PROXY HTTP_PROXY ALL_PROXY no_proxy NO_PROXY
export PYTHONPATH=""
# shellcheck source=/dev/null
source "${ROOT}/scripts/activate_czh_environ.sh"
cd "${ROOT}"

export USE_API_CONFIG=1
export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="${ROOT}/ascend_custom_opp"
export LLM4ASCENDC_REF_ON_CPU=1

TORCH1_IDX="$(
  python3 -c "
import json
from pathlib import Path
p = Path('data/external/CUDA-Agent-Ops-6K/cuda_agent_ops_6k.jsonl')
print(' '.join(
    str(i) for i, l in enumerate(p.open())
    if l.strip() and json.loads(l).get('data_source') == 'torch#1'
))
"
)"
N=$(echo "${TORCH1_IDX}" | wc -w)
echo "[INFO] torch#1 rows: ${N} (expect 204)"
if [[ "${N}" != "204" ]]; then
  echo "[ERROR] unexpected row count"
  exit 2
fi

echo "[INFO] API smoke test (optional gate)..."
if ! python3 generator/scripts/smoke_llm_chat.py --model deepseek-v4-flash; then
  echo "[ERROR] smoke_llm_chat failed (often HTTP 402 = quota/billing)."
  echo "        Fix generator/local_api_config.py account balance, then re-run this script."
  exit 1
fi

python3 generator/scripts/run_agent_cuda_agent_multi_rounds.py \
  --model deepseek-v4-flash \
  --indices ${TORCH1_IDX} \
  --tool-mode no_tool \
  --strategy one_shot \
  --use-repair-memory \
  --run 2 \
  --max-attempts 10 \
  --parallel-ops 4 \
  --eval-workers 1 \
  --eval-npu 4 \
  --eval-mode full \
  --clean-policy force \
  --dataset-path "${ROOT}/data/external/CUDA-Agent-Ops-6K/cuda_agent_ops_6k.jsonl"

echo "=== $(date -Is) done ==="
