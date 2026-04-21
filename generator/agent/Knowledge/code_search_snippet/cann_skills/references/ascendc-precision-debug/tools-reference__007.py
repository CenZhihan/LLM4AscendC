# gen_boundary_data.py
import numpy as np

def generate_boundary_tests(dtype):
    """生成边界值测试数据"""
    np_type = np.float16 if dtype == 'fp16' else np.float32

    cases = {
        "zero": 0.0,
        "tiny": 1e-10,
        "small": 1e-6,
        "normal": 1.0,
        "large": 1e6,
        "huge": 1e10,
        "negative": -1.0,
    }

    if dtype == 'fp16':
        cases["saturation"] = 65504.0
        cases["negative_saturation"] = -65504.0

    for name, value in cases.items():
        data = np.full((8, 16), value, dtype=np_type)
        np.save(f"boundary_{name}_{dtype}.npy", data)
        print(f"生成: boundary_{name}_{dtype}.npy (value={value})")

if __name__ == "__main__":
    generate_boundary_tests("fp32")
    generate_boundary_tests("fp16")
