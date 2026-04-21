# Level 0: 验证基本逻辑
python3 gen_golden.py "8" -1 float32 test_level0_1
python3 gen_golden.py "8" -1 float16 test_level0_2

# Level 1: 验证正常功能
python3 gen_golden.py "1024" -1 float32 test_level1_1

# Level 2: 验证边界条件
python3 gen_golden.py "1024" -1 float32 test_level2_extreme
