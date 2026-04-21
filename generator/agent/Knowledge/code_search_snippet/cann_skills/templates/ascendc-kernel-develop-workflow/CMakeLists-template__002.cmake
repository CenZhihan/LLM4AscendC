# ✅ 正确
project(<operator_name> LANGUAGES ASC CXX)

# ❌ 错误 - 缺少 ASC 语言声明
project(<operator_name> LANGUAGES CXX)
