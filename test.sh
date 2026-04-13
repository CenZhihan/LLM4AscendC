#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate multi-kernel-bench
python3 tools/eval_operator.py \
    --txt-dir output/kernelbench165_batches/batch_03/ \
    --mode full \
    --clean-policy force
