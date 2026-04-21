# 在项目根目录下执行

# 步骤1: 生成 L0 用例（单因子覆盖，≤200条）
python skills/ascendc-st-design/scripts/generate_test_cases.py \
    result/{operator_name}/03_参数定义.yaml \
    result/{operator_name}/04_测试因子.yaml \
    result/{operator_name}/07_因子值.csv \
    result/{operator_name}/ \
    --level L0 \
    --verbose

# 步骤2: 生成 L1 用例（两两组合覆盖，500~700条）
python skills/ascendc-st-design/scripts/generate_test_cases.py \
    result/{operator_name}/03_参数定义.yaml \
    result/{operator_name}/04_测试因子.yaml \
    result/{operator_name}/07_因子值.csv \
    result/{operator_name}/ \
    --level L1 \
    --target-count 500 \
    --seed 42 \
    --verbose
