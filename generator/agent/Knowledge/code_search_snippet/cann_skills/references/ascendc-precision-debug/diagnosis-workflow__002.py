import numpy as np

# 基础误差统计
errors = abs(pred - truth)
print(f"最大误差: {errors.max():.6f}")
print(f"平均误差: {errors.mean():.6f}")
print(f"95分位误差: {np.percentile(errors, 95):.6f}")
print(f"中位数误差: {np.median(errors):.6f}")

# 找出误差最大的样本
worst_idx = errors.argmax()
print(f"最差样本: idx={worst_idx}, pred={pred[worst_idx]}, truth={truth[worst_idx]}")

# 相对误差分析
rel_errors = abs(pred - truth) / (abs(truth) + 1e-9)
print(f"最大相对误差: {rel_errors.max():.2e}")
print(f"平均相对误差: {rel_errors.mean():.2e}")

# 误差分布分析
print(f"误差 > 1e-3 的比例: {(errors > 1e-3).sum() / len(errors) * 100:.2f}%")
print(f"误差 > 1e-4 的比例: {(errors > 1e-4).sum() / len(errors) * 100:.2f}%")
