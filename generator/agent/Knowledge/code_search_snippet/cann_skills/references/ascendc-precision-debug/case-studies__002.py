# 分析误差分布
pred = np.load('output.npy')
truth = np.load('expected.npy')
error = np.abs(pred - truth)

print(f"Max error: {error.max():.2e}")
print(f"Mean error: {error.mean():.2e}")

# 找出最差样本
worst_idx = error.argmax()
print(f"Worst @{worst_idx}: pred={pred.flat[worst_idx]}, truth={truth.flat[worst_idx]}")
