# torch.library注册自定义算子直调样例

## 概述

本样例基于Add算子展示如何使用PyTorch的torch.library机制注册自定义算子。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── torch_library
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── add_custom_test.py      // PyTorch调用脚本
│   └── add_custom.asc          // Ascend C样例实现 & 自torch.library注册
```

## 样例描述

- 样例功能：

  Add计算公式为：

  ```
  z = x + y
  ```

- 样例规格：
  <table border="2" align="center">
  <caption>表1：AddCustom样例规格描述</caption>
  <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="4" align="center">AddCustom</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[8, 2048]</td><td align="center">float16</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">[8, 2048]</td><td align="center">float16</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">样例输出</td><td align="center">z</td><td align="center">[8, 2048]</td><td align="center">float16</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>
  </table>

- 自定义算子注册：

  本样例在`add_custom.asc`中定义了一个名为`ascendc_ops`的命名空间，并在其中注册了`ascendc_add`函数。

  PyTorch提供`TORCH_LIBRARY`宏作为自定义样例注册的核心接口，用于创建并初始化自定义算子库，注册后在Python侧可以通过`torch.ops.namespace.op_name`方式进行调用，例如：

  ```c++
  TORCH_LIBRARY(ascendc_ops, m) {
      m.def(ascendc_add"(Tensor x, Tensor y) -> Tensor");
  }
  ```

  `TORCH_LIBRARY_IMPL`用于将算子绑定到特定的`DispatchKey`（PyTorch设备调度标识）。针对NPU设备，需要将算子实现注册到`PrivateUse1`这一专属的`DispatchKey`上，例如：

  ```c++
  TORCH_LIBRARY_IMPL(ascendc_ops, PrivateUse1, m)
  {
      m.impl("ascendc_add", TORCH_FN(ascendc_ops::ascendc_add));
  }
  ```

  在`ascendc_add`函数中通过`c10_npu::getCurrentNPUStream()`函数获取当前NPU上的流，并通过内核调用符`<<<>>>`调用自定义的Kernel函数`add_custom`，在NPU上执行算子。

- Python测试脚本

  在`add_custom_test.py`调用脚本中，通过`torch.ops.load_library`加载生成的自定义算子库，调用注册的`ascendc_add`函数，并通过对比NPU输出与CPU标准加法结果来验证自定义算子的数值正确性。

## 编译运行

- 安装PyTorch以及Ascend Extension for PyTorch插件

  请参考[pytorch: Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch)开源代码仓或[Ascend Extension for PyTorch昇腾社区](https://hiascend.com/document/redirect/Pytorch-index)的安装说明，选取支持的`Python`版本配套发行版，完成`torch`和`torch-npu`的安装。

- 安装前置依赖

  ```bash
  pip3 install expecttest
  ```

- 配置环境变量

  请根据当前环境上CANN开发套件包的[安装方式](../../../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
  - 默认路径，root用户安装CANN软件包

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - 默认路径，非root用户安装CANN软件包

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - 指定路径install_path，安装CANN软件包

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- 样例执行

  在本样例根目录下执行如下步骤，运行该样例。

  ```bash
  mkdir -p build; cd build
  cmake ..; make -j
  python3 ../add_custom_test.py
  ```

  执行结果如下，说明精度对比成功。

  ```bash
  Ran 1 test in **s
  OK
  ```