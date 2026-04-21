# 生成 3 个基础用例（不同 shape/dtype）
python3 gen_golden.py "8" -1 float32 test_case_1
python3 gen_golden.py "1024" -1 float16 test_case_2
python3 gen_golden.py "64,128" -1 float32 test_case_3
