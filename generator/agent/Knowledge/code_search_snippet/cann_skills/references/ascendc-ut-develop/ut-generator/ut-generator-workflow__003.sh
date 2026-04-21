# 从 build.sh 提取支持的 SoC 列表
grep "SUPPORT_COMPUTE_UNIT_SHORT" $REPO/build.sh | sed 's/.*(\(.*\)).*/\1/' | tr -d '"' | tr ',' '\n'

# 从源码提取支持的 dtype
grep -rn "ge::DT_" $OP_PATH/op_host/*.cpp 2>/dev/null | grep -oE "DT_[A-Z0-9]+" | sort -u

# 从源码提取支持的 format
grep -rn "FORMAT_" $OP_PATH/op_host/*.cpp 2>/dev/null | grep -oE "FORMAT_[A-Z0-9]+" | sort -u
