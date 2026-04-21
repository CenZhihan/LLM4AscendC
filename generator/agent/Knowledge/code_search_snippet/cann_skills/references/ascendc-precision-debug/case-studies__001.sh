# 从最小形状开始，32字节对齐
./softmaxv5 16 16 8 fp32

# 如果通过，测试 FP16
./softmaxv5 16 16 8 fp16
