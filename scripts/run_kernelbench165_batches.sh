#!/usr/bin/env bash
set -euo pipefail

# Activate conda env
source /root/miniconda3/etc/profile.d/conda.sh
conda activate multi-kernel-bench

# Run warmup/eval for each batch directory (sorted)
for d in $(find output/kernelbench165_batches -maxdepth 1 -mindepth 1 -type d | sort); do
  echo "[warmup] === $d ==="
  python3 tools/eval_operator.py --txt-dir "$d" --clean-policy smart || true
done
