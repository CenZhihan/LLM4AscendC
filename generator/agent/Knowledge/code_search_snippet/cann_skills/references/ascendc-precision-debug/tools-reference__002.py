# error_analysis.py
import numpy as np
import sys

def analyze_error(pred_file, truth_file, rtol=1e-5, atol=1e-6):
    pred = np.load(pred_file)
    truth = np.load(truth_file)

    abs_error = np.abs(pred - truth)
    rel_error = abs_error / (np.abs(truth) + atol)

    print("=" * 60)
    print("误差分析报告")
    print("=" * 60)
    print(f"预测文件: {pred_file}")
    print(f"真值文件: {truth_file}")
    print()

    # 绝对误差统计
    print("绝对误差统计:")
    print(f"  最大值: {abs_error.max():.6e}")
    print(f"  平均值: {abs_error.mean():.6e}")
    print(f"  中位数: {np.median(abs_error):.6e}")
    print(f"  标准差: {abs_error.std():.6e}")
    print()

    # 相对误差统计
    print("相对误差统计:")
    print(f"  最大值: {rel_error.max():.6e}")
    print(f"  平均值: {rel_error.mean():.6e}")
    print(f"  中位数: {np.median(rel_error):.6e}")
    print(f"  95分位: {np.percentile(rel_error, 95):.6e}")
    print(f"  99分位: {np.percentile(rel_error, 99):.6e}")
    print()

    # 通过率
    pass_mask = np.logical_or(abs_error < atol, rel_error < rtol)
    pass_rate = pass_mask.sum() / pass_mask.size * 100
    print(f"通过率: {pass_rate:.2f}%")
    print()

    # 误差分布
    print("误差分布:")
    for threshold in [1e-3, 1e-4, 1e-5, 1e-6]:
        count = (abs_error > threshold).sum()
        rate = count / abs_error.size * 100
        print(f"  误差 > {threshold:.0e}: {count} ({rate:.2f}%)")
    print()

    # 最差样本
    worst_idx = abs_error.argmax()
    worst_pos = np.unravel_index(worst_idx, pred.shape)
    print(f"最差样本 @ {worst_pos}:")
    print(f"  预测值: {pred[worst_pos]:.6f}")
    print(f"  真值: {truth[worst_pos]:.6f}")
    print(f"  绝对误差: {abs_error[worst_pos]:.6e}")
    print(f"  相对误差: {rel_error[worst_pos]:.6e}")

    return pass_rate > 99.0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python3 error_analysis.py <output.npy> <expected.npy>")
        sys.exit(1)

    success = analyze_error(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
