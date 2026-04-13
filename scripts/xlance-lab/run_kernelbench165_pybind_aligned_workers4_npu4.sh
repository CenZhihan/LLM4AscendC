#!/usr/bin/env bash
# 并行评测 output/kernelbench165_txt/pybind_reference_aligned 下 *.txt（仅 64 个曾改 reference 的算子，见 artifacts/pybind_reference_aligned_op_keys.txt）
# 用于 reference 已与 pybind 契约对齐后的复跑。
# 配置：4 workers + 4 张 NPU（worker i -> 物理卡 i%4，即 0..3 各一个进程）
# --clean-policy force：每次构建前清空 workspace/pybind
#
# 集群提交：请同时申请 4 张 Ascend（例如 vc submit -g 4）与足够内存（例如 -m 256G）。
set -euo pipefail

export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/ascend_custom_opp"

REPO_ROOT="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/LLM4AscendC"
cd "$REPO_ROOT"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

exec python3 tools/eval_operator.py \
  --txt-dir output/kernelbench165_txt/pybind_reference_aligned \
  --workers 4 \
  --npu 4 \
  --clean-policy force \
  --mode full
