#!/usr/bin/env bash
# 集群 / 本机：并行评测 output/gpt-5-rag-code 下全部 *.txt (共 300 个)
# 配置：4 workers + 4 张 NPU（worker i -> 物理卡 i%4，即 0..3 各一个进程）
# --clean-policy smart：源码未变时跳过构建，节省时间
#
# 提交集群时请同时申请 4 张 Ascend（例如 vc submit -g 4）与足够内存（例如 -m 256G）。
set -euo pipefail

# --- 自定义算子 OPP 安装路径（必需，多 worker 会创建子目录）---
export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="/aistor/sjtu/hpc_stor01/home/liuxiang/LLM4NPU/LLM4AscendC/ascend_custom_opp"

REPO_ROOT="/aistor/sjtu/hpc_stor01/home/liuxiang/LLM4NPU/LLM4AscendC"
cd "$REPO_ROOT"

# --- CANN 环境配置 ---
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# --- 驱动库路径 ---
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/devlib:$LD_LIBRARY_PATH

# --- 创建 OPP 目录 ---
mkdir -p "$LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH"

# --- 日志目录 ---
LOG_DIR="eval_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/eval_gpt5_rag_code_${TIMESTAMP}.log"

echo "[INFO] Starting batch evaluation: output/gpt-5-rag-code (300 operators)"
echo "[INFO] Config: 4 workers, 4 NPUs, clean-policy=smart"
echo "[INFO] Log file: $LOG_FILE"

exec python3 tools/eval_operator.py \
  --txt-dir output/gpt-5-rag-code \
  --workers 4 \
  --npu 4 \
  --clean-policy smart \
  --mode full \
  2>&1 | tee "$LOG_FILE"