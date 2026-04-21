import numpy as np

def compare_results(output, expected, rtol=1e-5, atol=1e-6):
    """对比结果并打印详细误差信息"""
    abs_error = np.abs(output - expected)
    rel_error = abs_error / (np.abs(expected) + atol)

    print(f"Max abs error: {abs_error.max():.2e}")
    print(f"Mean abs error: {abs_error.mean():.2e}")
    print(f"Max rel error: {rel_error.max():.2e}")
    print(f"Mean rel error: {rel_error.mean():.2e}")

    # 通过率
    pass_mask = np.logical_or(abs_error < atol, rel_error < rtol)
    pass_rate = pass_mask.sum() / pass_mask.size * 100
    print(f"Pass rate: {pass_rate:.2f}%")

    # 最差样本
    worst_idx = abs_error.argmax()
    print(f"Worst case @ {np.unravel_index(worst_idx, output.shape)}:")
    print(f"  Output: {output.flatten()[worst_idx]:.6f}")
    print(f"  Expected: {expected.flatten()[worst_idx]:.6f}")
    print(f"  Abs error: {abs_error.flatten()[worst_idx]:.2e}")

    return pass_rate > 99.0  # 99% 通过为合格
