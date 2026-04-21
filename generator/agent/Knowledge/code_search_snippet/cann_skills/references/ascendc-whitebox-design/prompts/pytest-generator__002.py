# 1. 导入
import pytest
import torch
torch_npu = pytest.importorskip("torch_npu")

# 2. PARAMS 列表（内嵌，来自 cases.json 的全部内容）
PARAMS = [
    {"x_dtype": "float16", "D": 32, "mode": 0, "_group": "group_a"},
    ...
]

# 3. 常量映射
DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, ...}

# 4. Reference 实现
def reference_{op_name}(...):
    """CPU 参考实现。复用已有或自行编写。"""
    ...

# 5. 测试函数
@pytest.mark.parametrize("p", PARAMS,
    ids=[f"D{p['D']}_{p['x_dtype']}_m{p['mode']}" for p in PARAMS])
def test_{op_name}(p):
    # a. 构造输入 tensor
    # b. 调用算子（NPU）
    # c. 调用 reference（CPU）
    # d. 断言（shape + dtype + 数值精度）
    ...
