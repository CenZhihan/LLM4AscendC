# torch.library调用集成Profiling上报Shape信息

## 概述

本样例展示基于torch.library调用方式，如何在Profiling采集性能数据时，上报算子的Shape信息，辅助用户进行算子性能分析。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── torch_library
│   ├── CMakeLists.txt           // 编译工程文件
│   ├── add_custom_test.py       // PyTorch调用脚本
│   ├── add_custom.asc           // 集成Profiling采集Shape信息
```

## 样例描述

- 样例功能

  以Add计算为例，计算公式为：

  ```
  z = x + y
  ```

- 样例规格

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

- 样例实现

  本样例在`add_custom.asc`中定义了一个名为`ascendc_ops`的命名空间，并在其中注册了`ascendc_add`函数，在 `<<<>>>` 前后调用`aclprofRangePushEx` 和 `aclprofRangePop` 上报Shape信息到Profiling。

  样例信息上报需要构造 `aclprofEventAttributes` 结构体，包含版本、大小、类型和tensor信息，其中 `messageType` 固定MESSAGE_TYPE_TENSOR_INFO类型为0，`aclprofTensorInfo` 为上报的tensor信息。`opNameId`、`opTypeId` 字段通过 `aclprofStr2Id` 接口转化样例名、样例类型获取，每包数据 `tensorNum` 不超过5，超过5请分成多包数据上报，`tensors` 为 `aclprofTensor` 结构，从样例的 `at::Tensor` 获取。

- Python测试脚本

  在`add_custom_test.py`调用脚本中，通过`torch.ops.load_library`加载生成的自定义样例库，调用注册的`ascendc_add`函数。

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
  msprof --application="python3 ../add_custom_test.py" --output="../result"
  ```

  执行结果如下，说明数据上报成功，其中DIR_NAME和PATH分别为落盘文件名和数据存放目录。

  ```bash
  [INFO] Query all data in ${DIR_NAME} done.
  [INFO] Profiling finished.
  [INFO] Process profiling data complete. Data is saved in ${PATH}
  ```

- Shape信息展示

  打开`PROF_000001_*/mindstudio_profiler_output/op_summary_*.csv`，查看Shape信息，样例中Shape信息写入如下字段。

  <table>
    <tr>
      <td align="center">...</td>
      <td align="center">Op Name</td>
      <td align="center">Op Type</td>
      <td align="center">Input Shapes</td>
      <td align="center">Input Data Types</td>
      <td align="center">Input Formats</td>
      <td align="center">Output Shapes</td>
      <td align="center">Output Data Types</td>
      <td align="center">Output Formats</td>
      <td align="center">...</td>
    </tr>
    <tr>
      <td align="center">...</td>
      <td align="center">_Z10add_customPhS_S_j</td>
      <td align="center">_Z10add_customPhS_S_j</td>
      <td align="center">"8,2048;8:2048"</td>
      <td align="center">FLOAT16;FLOAT16</td>
      <td align="center">ND;ND</td>
      <td align="center">"8,2048"</td>
      <td align="center">FLOAT16</td>
      <td align="center">ND</td>
      <td align="center">...</td>
    </tr>
  </table>