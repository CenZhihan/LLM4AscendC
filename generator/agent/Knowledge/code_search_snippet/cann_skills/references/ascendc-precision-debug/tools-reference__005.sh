#!/bin/bash
# batch_test.sh

# 测试不同的输入规模
shapes=("8:8:8" "16:16:8" "32:16:8" "64:32:8")
dtypes=("fp32" "fp16")

for shape in "${shapes[@]}"; do
    for dtype in "${dtypes[@]}"; do
        IFS=':' read -r M N K <<< "$shape"
        echo "Testing: M=$M, N=$N, K=$K, dtype=$dtype"

        ./env_setup.sh "cd ops/my_operator/build && ./my_operator $M $N $K $dtype"

        if [ $? -eq 0 ]; then
            echo "  PASS"
        else
            echo "  FAIL"
        fi
    done
done
