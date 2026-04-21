# 示例：用户需求 shape=[1,1024], dtype=float16
python3 gen_golden.py "1,1024" -1 float16 test_specific_case

# 可选：边界值测试（仅限用户指定的 shape/dtype）
python3 gen_golden.py "1,1024" -1 float16 test_boundary_extreme
