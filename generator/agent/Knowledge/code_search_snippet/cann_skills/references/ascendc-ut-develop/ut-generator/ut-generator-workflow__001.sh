# 路径格式：<repo>/<category>/<op>
# 示例：ops-math/math/add

# 提取各部分
REPO=$(echo $OP_PATH | cut -d'/' -f1)      # ops-math
CATEGORY=$(echo $OP_PATH | cut -d'/' -f2)  # math
OP_NAME=$(echo $OP_PATH | cut -d'/' -f3)   # add
