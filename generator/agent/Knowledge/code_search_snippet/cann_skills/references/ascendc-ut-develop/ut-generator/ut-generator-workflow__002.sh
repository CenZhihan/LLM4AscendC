# 检测各层级目录是否存在
ls $OP_PATH | grep -E "op_host|op_api|op_kernel|op_kernel_aicpu" 2>/dev/null
