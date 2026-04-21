# 在项目根目录下执行
python skills/ascendc-st-design/scripts/generate_factor_values.py <求解配置.yaml> <约束定义.yaml> <测试因子.yaml> <输出.csv> [选项]

# 示例
python skills/ascendc-st-design/scripts/generate_factor_values.py \
    result/{operator_name}/06_求解配置.yaml \
    result/{operator_name}/05_约束定义.yaml \
    result/{operator_name}/04_测试因子.yaml \
    result/{operator_name}/07_因子值.csv \
    --max-cases 10000
