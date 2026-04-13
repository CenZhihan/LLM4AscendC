#!/usr/bin/env bash
# 仅评测第三大类中 4 个 GRU 子集（reference 在 CPU 上用 float32 算 RNN，见 vendor/mkb/reference/arch/gru*.py）
# - 输入目录：output/kernelbench165_txt/category3_gru_cpu_fp32_ref_4
# - 配置：4 workers + 4 张 NPU
# - LLM4ASCENDC_REF_ON_CPU=1 -> reference 在 CPU，自定义算子仍在 NPU
set -euo pipefail

export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/ascend_custom_opp"
export LLM4ASCENDC_REF_ON_CPU=1

REPO_ROOT="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/LLM4AscendC"
cd "$REPO_ROOT"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

exec python3 tools/eval_operator.py \
  --txt-dir output/kernelbench165_txt/category3_gru_cpu_fp32_ref_4 \
  --workers 4 \
  --npu 4 \
  --clean-policy force \
  --mode full
