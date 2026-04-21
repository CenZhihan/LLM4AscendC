# 编译 op_host UT
bash build.sh -u --ophost --ops='<op_name>' --soc='<soc_version>'

# 编译 op_api UT
bash build.sh -u --opapi --ops='<op_name>' --soc='<soc_version>'

# 编译 op_kernel UT
bash build.sh -u --opkernel --ops='<op_name>' --soc='<soc_version>'

# 编译并生成覆盖率
bash build.sh -u --ophost --ops='<op_name>' --soc='<soc_version>' --cov
