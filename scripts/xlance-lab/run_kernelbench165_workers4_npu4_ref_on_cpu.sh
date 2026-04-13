#!/usr/bin/env bash
# 集群 / 本机：并行评测 output/kernelbench165_txt 下全部 *.txt（165 个）
# 配置：4 workers + 4 张 NPU（worker i -> 物理卡 i%4，即 0..3 各一个进程）
# 特殊：LLM4ASCENDC_REF_ON_CPU=1 -> MKB reference 在 CPU，自定义算子仍在 NPU（避免 reference 在 NPU 触发图融合/建图问题）
# --clean-policy force：每次构建前清空 workspace/pybind
#
# 提交集群时请同时申请 4 张 Ascend（例如 vc submit -g 4）与足够内存（建议 >=128G）。
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
  --txt-dir output/kernelbench165_txt \
  --workers 4 \
  --npu 4 \
  --clean-policy force \
  --mode full

