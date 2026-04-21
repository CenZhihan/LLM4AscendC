# 进入 Docker 容器运行
./env_setup.sh "cd ops/my_operator/build && ./my_operator"

# 带参数运行
./env_setup.sh "cd ops/my_operator/build && ./my_operator 16 16 8 fp32"

# FP16 测试
./env_setup.sh "cd ops/my_operator/build && ./my_operator 16 16 8 fp16"
