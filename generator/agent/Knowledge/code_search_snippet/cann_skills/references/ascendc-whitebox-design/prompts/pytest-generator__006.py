# ✅ 正确：从 cases.json 中内嵌全部参数
PARAMS = [{"x_dtype": "float16", "D": 32}, {"x_dtype": "bfloat16", "D": 64}, ...]

# ❌ 错误：手写参数
PARAMS = [{"x_dtype": "float16", "D": 32}]  # 只有一组，遗漏了其他

# ✅ 正确：NPU 环境守护
torch_npu = pytest.importorskip("torch_npu")

# ❌ 错误：直接 import（无 NPU 时整个文件报错）
import torch_npu

# ✅ 正确：精度容差根据算子调整
torch.testing.assert_close(npu_y.cpu().float(), ref_y.float(), rtol=0.1, atol=0.1)

# ❌ 错误：用 == 做浮点对比
assert (npu_y.cpu() == ref_y).all()

# ✅ 正确：没有 reference 时只验证属性
if p["dst_type"] == 2:
    torch.testing.assert_close(...)
else:
    assert y_npu.shape == expected_shape  # 只验 shape

# ❌ 错误：跳过没有 reference 的参数组合
if p["dst_type"] != 2:
    pytest.skip("no reference")  # 浪费了枚举预算
