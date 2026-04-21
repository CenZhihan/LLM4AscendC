# verify_result.py
import numpy as np
import sys

def verify_result(output_file, expected_file, rtol=1e-5, atol=1e-6):
    """验证算子输出结果"""
    output = np.load(output_file)
    expected = np.load(expected_file)

    # 检查形状
    if output.shape != expected.shape:
        print(f"形状不匹配: output={output.shape}, expected={expected.shape}")
        return False

    # 计算误差
    abs_error = np.abs(output - expected)
    rel_error = abs_error / (np.abs(expected) + atol)

    max_abs_error = abs_error.max()
    max_rel_error = rel_error.max()

    # 打印结果
    print("=" * 60)
    print("验证结果")
    print("=" * 60)
    print(f"最大绝对误差: {max_abs_error:.6e}")
    print(f"最大相对误差: {max_rel_error:.6e}")

    # 判断通过
    pass_mask = np.logical_or(abs_error < atol, rel_error < rtol)
    pass_count = pass_mask.sum()
    total_count = pass_mask.size
    pass_rate = pass_count / total_count * 100

    print(f"通过率: {pass_count}/{total_count} ({pass_rate:.2f}%)")

    if pass_rate >= 99.0:
        print("验证: PASS")
        return True
    else:
        print("验证: FAIL")

        # 打印失败样本
        fail_indices = np.where(~pass_mask)
        if len(fail_indices[0]) > 0:
            print("\n失败样本（前10个）:")
            fail_count = min(10, len(fail_indices[0]))
            for i in range(fail_count):
                idx = tuple(dim[i] for dim in fail_indices)
                print(f"  @{idx}: output={output[idx]:.6f}, "
                      f"expected={expected[idx]:.6f}, "
                      f"abs_err={abs_error[idx]:.2e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python3 verify_result.py <output.npy> <expected.npy> [rtol] [atol]")
        sys.exit(1)

    rtol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
    atol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-6

    success = verify_result(sys.argv[1], sys.argv[2], rtol, atol)
    sys.exit(0 if success else 1)
