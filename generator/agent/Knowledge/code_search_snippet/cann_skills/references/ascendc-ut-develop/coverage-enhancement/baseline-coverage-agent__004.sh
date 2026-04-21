# 覆盖率报告路径（编译后在 build 目录生成）
COV_PATH={PROJECT_DIR}/build/tests/ut/cov_report/cpp_utest

# 查看覆盖率摘要
lcov --summary $COV_PATH/ops.info_filtered

# 查找未覆盖的代码行
lcov --list $COV_PATH/ops.info_filtered | grep ":0"
