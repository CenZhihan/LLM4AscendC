#!/usr/bin/env bash
set -u

# Serial probe for one operator across physical NPUs 0..3.
# Requirement: keep all settings identical except ASCEND_VISIBLE_DEVICES.
# Each run uses --clean-policy force --mode full.
#
# Usage:
#   cd /root/czh/LLM4AscendC
#   bash tools/npu_serial_probe_one_op.sh

REPO_ROOT="/root/czh/LLM4AscendC"
OP_KEY="conv_standard_3d_square_input_asymmetric_kernel"
TXT_PATH="output/kernelbench165_txt/${OP_KEY}.txt"
ART_RESULT_DIR="artifacts/kernelbench165_txt/${OP_KEY}"
RESULT_JSON_NAME="result_${OP_KEY}.json"

CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
ASCEND_ENV="/usr/local/Ascend/ascend-toolkit/set_env.sh"
DEFAULT_OPP_PATH="/root/czh/LLM4AscendC/ascend_custom_opp"

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "[fatal] repo root not found: ${REPO_ROOT}" >&2
  exit 1
fi

cd "${REPO_ROOT}" || exit 1

if [[ ! -f "${TXT_PATH}" ]]; then
  echo "[fatal] txt not found: ${REPO_ROOT}/${TXT_PATH}" >&2
  exit 1
fi

if [[ -f "${ASCEND_ENV}" ]]; then
  # shellcheck source=/dev/null
  source "${ASCEND_ENV}"
fi

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate czh_environ
fi

export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="${LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH:-${DEFAULT_OPP_PATH}}"
export LLM4ASCENDC_REF_ON_CPU=1

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="artifacts/npu_serial_probe/${OP_KEY}/${STAMP}"
mkdir -p "${OUT_ROOT}"

echo "[info] repo=${REPO_ROOT}" | tee "${OUT_ROOT}/run_info.txt"
echo "[info] txt=${TXT_PATH}" | tee -a "${OUT_ROOT}/run_info.txt"
echo "[info] opp_path=${LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH}" | tee -a "${OUT_ROOT}/run_info.txt"
echo "[info] mode=full clean_policy=force workers=1" | tee -a "${OUT_ROOT}/run_info.txt"

for npu in 0 1 2 3; do
  RUN_DIR="${OUT_ROOT}/npu${npu}"
  mkdir -p "${RUN_DIR}"
  export ASCEND_VISIBLE_DEVICES="${npu}"
  echo "[info] ===== npu=${npu} ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES} =====" | tee -a "${OUT_ROOT}/run_info.txt"

  set +e
  python3 tools/eval_operator.py \
    --txt "${TXT_PATH}" \
    --workers 1 \
    --clean-policy force \
    --mode full \
    2>&1 | tee "${RUN_DIR}/run_console.log"
  RC=$?
  set -e

  echo "${RC}" > "${RUN_DIR}/exit_code.txt"

  RESULT_PATH="${ART_RESULT_DIR}/${RESULT_JSON_NAME}"
  if [[ -f "${RESULT_PATH}" ]]; then
    cp -a "${RESULT_PATH}" "${RUN_DIR}/${RESULT_JSON_NAME}"
  else
    echo "[warn] result json missing after npu=${npu}: ${RESULT_PATH}" | tee -a "${RUN_DIR}/run_console.log"
  fi

  LATEST_EVAL_LOG="$(ls -1t "${ART_RESULT_DIR}/logs/"*-06-eval.log 2>/dev/null | head -n 1)"
  if [[ -n "${LATEST_EVAL_LOG}" && -f "${LATEST_EVAL_LOG}" ]]; then
    cp -a "${LATEST_EVAL_LOG}" "${RUN_DIR}/06-eval.log"
    printf "%s\n" "${LATEST_EVAL_LOG}" > "${RUN_DIR}/source_eval_log_path.txt"
  else
    echo "[warn] eval log missing after npu=${npu}: ${ART_RESULT_DIR}/logs/*-06-eval.log" | tee -a "${RUN_DIR}/run_console.log"
  fi
done

python3 - <<'PY'
import json
from pathlib import Path

repo = Path("/root/czh/LLM4AscendC")
op_key = "conv_standard_3d_square_input_asymmetric_kernel"
root = repo / "artifacts" / "npu_serial_probe" / op_key
run_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
if not run_dirs:
    raise SystemExit(0)

target = run_dirs[-1]
summary = {"op_key": op_key, "run_root": str(target), "cards": []}
for npu in range(4):
    d = target / f"npu{npu}"
    item = {
        "npu": npu,
        "ascend_visible_devices": str(npu),
        "exit_code": None,
        "compiled": None,
        "correctness": None,
        "correctness_info": None,
        "result_json": str(d / f"result_{op_key}.json"),
        "eval_log": str(d / "06-eval.log"),
    }
    ec = d / "exit_code.txt"
    if ec.exists():
        item["exit_code"] = ec.read_text(encoding="utf-8", errors="replace").strip()
    rj = d / f"result_{op_key}.json"
    if rj.exists():
        try:
            obj = json.loads(rj.read_text(encoding="utf-8"))
            row = (obj.get("result") or {}).get(op_key) or {}
            item["compiled"] = row.get("compiled")
            item["correctness"] = row.get("correctness")
            item["correctness_info"] = row.get("correctness_info")
        except Exception as e:
            item["correctness_info"] = f"failed to parse result json: {e}"
    summary["cards"].append(item)

(target / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[info] manifest generated: {target / 'manifest.json'}")
PY

echo "[done] serial probe finished. output root: ${OUT_ROOT}"
