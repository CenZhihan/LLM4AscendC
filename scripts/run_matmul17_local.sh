#!/usr/bin/env bash
# 在本机 Docker 容器里跑评测（需 CANN，PATH 中有 msopgen）。
#
# 1) 宿主机先进入容器：
#      docker exec -it b31f8ebb90b9 bash
# 2) 在容器内进入仓库根（按你侧载方式二选一）：
#      cd /workspace/LLM4AscendC
#    或
#      cd /aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/LLM4AscendC
# 3) 执行：
#      bash scripts/run_matmul17_local.sh
#
# 若容器内没有 /aistor/... 路径，请先把本脚本里的 LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH
# 改成容器内可见路径（例如侧载为 /workspace 时可用 /workspace/ascend_custom_opp）。
set -euo pipefail
export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/ascend_custom_opp"
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
exec python3 tools/eval_operator.py --txt-dir output/gpt-4o_selected_shot/matmul_gpt4o_run0 --mode full
