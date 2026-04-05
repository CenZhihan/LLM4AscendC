#!/bin/bash
set -e

# ============================================================
# LLM4AscendC 批量评测脚本 - output/kernelbench 目录
# 用法：在集群节点上直接执行此脚本（默认工作目录为 LLM4AscendC）
# ============================================================

# --- CANN 环境配置 ---
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# --- 驱动库路径 ---
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/devlib:$LD_LIBRARY_PATH

# --- 自定义算子 OPP 环境（如果已安装）---
# 使用项目根目录下的 ascend_custom_opp
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/ascend_custom_opp/vendors/customize/bin/set_env.bash" ]; then
    source "$SCRIPT_DIR/ascend_custom_opp/vendors/customize/bin/set_env.bash"
fi

# # --- Python 依赖安装 ---
# echo "[INFO] Installing Python dependencies..."
# pip install -q -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# # --- torch_npu 安装（如果尚未安装）---
# if ! python -c "import torch_npu" 2>/dev/null; then
#     echo "[INFO] Installing torch_npu from CANN toolkit..."
#     TORCH_NPU_WHEEL="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/torch_npu/torch_npu-2.1.0.post12-cp310-cp310-linux_aarch64.whl"
#     if [ -f "$TORCH_NPU_WHEEL" ]; then
#         pip install -q "$TORCH_NPU_WHEEL"
#     else
#         echo "[WARN] torch_npu wheel not found at expected path, skipping..."
#     fi
# fi

# --- 评测参数配置 ---
TXT_DIR="output/gpt-5-rag-code"
MODE="full"              # full | build-only | eval-only
CLEAN_POLICY="smart"     # smart | force
LOG_DIR="eval_logs"

# 创建日志目录
mkdir -p "$LOG_DIR"

# --- 执行批量评测 ---
echo "[INFO] Starting batch evaluation: $TXT_DIR"
echo "[INFO] Mode: $MODE, Clean Policy: $CLEAN_POLICY"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/eval_kernelbench_${TIMESTAMP}.log"

python3 tools/eval_operator.py \
    --txt-dir "$TXT_DIR" \
    --mode "$MODE" \
    --clean-policy "$CLEAN_POLICY" \
    2>&1 | tee "$LOG_FILE"

echo "[INFO] Evaluation completed. Log saved to: $LOG_FILE"