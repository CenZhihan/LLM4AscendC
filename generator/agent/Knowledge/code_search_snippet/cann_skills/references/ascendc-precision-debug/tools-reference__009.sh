# FP16 验证（宽松容差）
python3 verify_result.py output.npy expected.npy 1e-3 1e-4

# FP32 验证（标准容差）
python3 verify_result.py output.npy expected.npy 1e-5 1e-6

# Reduce 算子验证（更宽松）
python3 verify_result.py output.npy expected.npy 5e-3 1e-4
