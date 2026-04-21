# 1. 检查环境变量
echo $ASCEND_HOME_PATH

# 2. 检查编译器
ls $ASCEND_HOME_PATH/aarch64-linux/ccec_compiler/bin/bisheng

# 3. 检查头文件
ls $ASCEND_HOME_PATH/include/kernel_operator.h

# 4. 检查库文件
ls $ASCEND_HOME_PATH/lib64/libregister.so

# 5. 检查 asc-devkit
ls asc-devkit/

# 6. 检查 NPU 设备（如果有）
npu-smi info
