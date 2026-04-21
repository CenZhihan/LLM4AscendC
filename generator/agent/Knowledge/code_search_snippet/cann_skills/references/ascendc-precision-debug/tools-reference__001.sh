# 最大误差和平均误差
python3 -c "import numpy as np; pred=np.load('output.npy'); truth=np.load('expected.npy'); \
  print(f'Max: {abs(pred-truth).max():.2e}, Mean: {abs(pred-truth).mean():.2e}')"

# 找出最差样本
python3 -c "import numpy as np; pred=np.load('output.npy'); truth=np.load('expected.npy'); \
  err=abs(pred-truth); idx=err.argmax(); \
  print(f'Worst@{idx}: pred={pred.flat[idx]}, truth={truth.flat[idx]}')"

# 完整误差统计（包括相对误差和分位数）
python3 -c "import numpy as np; pred=np.load('output.npy'); truth=np.load('expected.npy'); \
  err=abs(pred-truth); rel_err=err/(abs(truth)+1e-9); \
  print(f'Max abs: {err.max():.2e}, Max rel: {rel_err.max():.2e}, 95th: {np.percentile(rel_err, 95):.2e}')"
