#!/usr/bin/env bash
# 串行评测 gpt-4o selected_shot 的 matmul 类 17 个算子（来源：LLM4KERNEL_From_LT/.../gpt-4o/run0）
#
# 集群提交时工作目录必须与 kernelbench165 等任务一致：
#   vc submit ... -d "/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/LLM4AscendC" ...
# 不要使用 LLM_bench 父目录作为 -d，否则相对路径 tools/、output/ 会对不上。
set -euo pipefail
export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/ascend_custom_opp"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
exec python3 tools/eval_operator.py --txt-dir output/gpt-4o_selected_shot/matmul_gpt4o_run0 --mode full
