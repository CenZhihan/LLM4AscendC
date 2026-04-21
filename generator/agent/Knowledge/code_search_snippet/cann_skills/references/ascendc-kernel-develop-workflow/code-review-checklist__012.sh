# 检查是否使用了 DataCopy（GM-UB 搬运）
grep -n "DataCopy.*Gm\|DataCopy.*Local" *.asc *.cpp

# 如果发现类似以下代码，直接不通过：
# DataCopy(xLocal, xGm, ...)  // ⛔️ 应使用 DataCopyPad
# DataCopy(yGm, yLocal, ...)  // ⛔️ 应使用 DataCopyPad
