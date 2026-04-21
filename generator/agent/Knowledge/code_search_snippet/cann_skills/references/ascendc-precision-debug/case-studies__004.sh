# 测试硬件约束边界
./softmaxv5 8 8 8 fp32    # 最小列数=8
./softmaxv5 16 8 8 fp16   # FP16 最小列数

# 测试大规模
./softmaxv5 1024 256 8 fp32

# 测试非方形
./softmaxv5 256 512 4 fp32
