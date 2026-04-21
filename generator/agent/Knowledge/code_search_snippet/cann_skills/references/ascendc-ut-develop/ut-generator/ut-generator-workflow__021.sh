lcov --extract ops.info_filtered \
    "*/<category>/<op>/op_api/*" \
    "*/<category>/<op>/op_host/*" \
    "*/<category>/<op>/op_kernel/*" \
    -o /tmp/<op>_cov.info

lcov --summary /tmp/<op>_cov.info
