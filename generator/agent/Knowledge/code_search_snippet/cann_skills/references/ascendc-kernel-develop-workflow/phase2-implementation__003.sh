# 1. ⚠️ 强制检查：Tiling 计算位置
grep -n "Compute.*Tiling" *.h  # 确认 Tiling 在 common.h 或 Host 侧计算
# 如果在 Kernel 的 Init() 或 Process() 中发现计算逻辑 → 直接不通过

# 2. ⚠️ 强制检查：Host侧API调用顺序
# 查找 aclrtGetDeviceInfo 的行号
grep -n "aclrtGetDeviceInfo" *.asc
# 查找 aclrtSetDevice 的行号
grep -n "aclrtSetDevice" *.asc
# 查找 aclInit 的行号
grep -n "aclInit" *.asc
# 验证顺序：aclInit → aclrtSetDevice → aclrtGetDeviceInfo
# 如果 aclrtGetDeviceInfo 在 aclrtSetDevice 之前 → 直接不通过

# 3. ⚠️ 强制检查：动态核数计算
grep -n "numBlocks.*=.*[0-9]" *.asc *.cpp  # 检查是否写死核数
grep -n "usedCoreNum.*=.*[0-9]" *.asc *.cpp
# 如果发现 numBlocks=8 或 usedCoreNum=40 等写死的值 → 直接不通过
# 正确：usedCoreNum 通过 aclrtGetDeviceInfo 获取

# 4. 检查多核切分合理性
# - 数据量与核数是否匹配
# - 尾核处理是否正确（检查 rowsTail/elementsTail 逻辑）

# 5. 检查 CMakeLists.txt
cat CMakeLists.txt | grep -E "find_package|add_executable|target_link"

# 6. ⚠️ 强制检查：GM-UB 数据搬运 API
grep -n "DataCopy.*Gm\|DataCopy.*Local" *.asc *.cpp
# 如果发现 DataCopy(xLocal, xGm, ...) 或 DataCopy(yGm, yLocal, ...)
# 直接不通过，必须使用 DataCopyPad

# 7. 检查 Kernel 侧 API 使用
# - ReduceMax/ReduceSum 参数是否正确
# - Exp/Sub/Div count 参数是否匹配
# - Cast RoundMode 是否正确

# 8. 检查数据类型
# - FP16 输入是否使用 FP32 中间计算
# - Cast 转换方向是否正确

# 9. 运行验证脚本（如有）
python verify_cmake_config.py CMakeLists.txt
