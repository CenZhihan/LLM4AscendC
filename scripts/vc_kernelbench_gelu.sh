#!/usr/bin/env bash
# 集群容器内：共享盘路径与登录节点一致（见 vc submit -d）
# 集群默认无侧载 /workspace，不能写 /workspace/ascend_custom_opp；自定义 OPP 装到工程父目录下
set -euo pipefail
REPO_ROOT="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/LLM4AscendC"
export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/ascend_custom_opp"
cd "$REPO_ROOT"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
exec python3 tools/eval_operator.py --txt output/kernelbench/gelu.txt --mode full
