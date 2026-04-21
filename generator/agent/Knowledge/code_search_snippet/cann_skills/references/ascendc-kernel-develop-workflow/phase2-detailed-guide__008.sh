# 1. 检查 Tiling 计算位置
grep -n "Compute.*Tiling" *.h

# 2. 检查 CMakeLists.txt 配置
cat CMakeLists.txt | grep -E "find_package|add_executable|target_link"

# 3. 检查 API 使用（根据具体算子）
# - ReduceMax/ReduceSum 参数
# - Exp/Sub/Div count 参数
# - Cast RoundMode

# 4. 运行验证脚本
python verify_cmake_config.py CMakeLists.txt
