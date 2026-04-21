lcov --extract ops.info_filtered \
    "*/<category>/<op>/op_api/*" \
    "*/<category>/<op>/op_host/*" \
    "*/<category>/<op>/op_kernel/*" \
    "*/<category>/<op>/op_kernel_aicpu/*" \
    "*/<category>/<op>/op_tiling/*" \
    -o /tmp/<op>_cov.info

# 查看摘要
lcov --summary /tmp/<op>_cov.info
