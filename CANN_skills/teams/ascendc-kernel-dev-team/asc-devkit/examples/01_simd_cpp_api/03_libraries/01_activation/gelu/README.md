# Gelu类样例

## 概述

本样例在激活函数场景下，基于Gelu高阶API对输入Tensor按元素做GELU（Gaussian Error Linear Unit）计算。本样例使用float数据类型，输入Tensor元素个数为32，通过CopyIn-Compute-CopyOut三阶段流水线完成计算。更多API详细信息请参考[Gelu API文档](../../../../../docs/api/context/Gelu.md)。

> **进阶接口提示：** 除本样例使用的 `Gelu` 接口外，Ascend C 还提供了以下进阶 GELU 接口，调用方式与 `Gelu` 一致，只需替换函数名即可切换：
> - **[FasterGelu](../../../../../docs/api/context/FasterGelu.md)**：GELU 的加速版本，适用于对性能要求更高的场景。将 `gelu.asc` 中 `AscendC::Gelu` 替换为 `AscendC::FasterGelu`。
> - **[FasterGeluV2](../../../../../docs/api/context/FasterGeluV2.md)**：GELU 的进一步优化版本，能降低算力需求。将 `gelu.asc` 中 `AscendC::Gelu` 替换为 `AscendC::FasterGeluV2`。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── gelu
│   ├── scripts
│   │   └── gen_data.py         // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   ├── README.md               // 样例说明文档
│   └── gelu.asc                // Ascend C算子实现 & 调用样例
```

## 样例描述

- 样例功能：  
  本样例对输入Tensor按元素做GELU激活计算，将计算结果写入输出Tensor。

  计算公式如下：
  $$dstLocal_i = GELU(srcLocal_i)$$
  $$GELU(x)=0.5 * x * (1 + tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715 * x^3)))$$
  $$GELU(x)=\frac{x}{1 + e^{-1.59576912 * (x + 0.044715 * x^3)}}$$

- 样例规格：  
  <table>
  <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="4" align="center"> gelu </td></tr>

  <tr><td rowspan="3" align="center">样例输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src</td><td align="center">[1, 32]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">样例输出</td></tr>
  <tr><td align="center">dst</td><td align="center">[1, 32]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">gelu_custom</td></tr>
  </table>

- 样例实现：  
  本样例中实现的是固定shape为输入src[1, 32]，输出dst[1, 32]的gelu_custom样例。

  - Kernel实现

    核心计算步骤：将输入数据从Global Memory搬运到Local Tensor后，调用 `AscendC::Gelu(dstLocal, srcLocal, dataSize)` 完成GELU计算，再将结果搬回Global Memory。

  - 调用实现  
    使用内核调用符<<<>>>调用核函数。

## 编译运行

在本样例根目录下执行如下步骤，编译并执行样例。
- 配置环境变量
  请根据当前环境上CANN开发套件包的[安装方式](../../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
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
  ```bash
  mkdir -p build && cd build;
  cmake -DNPU_ARCH=dav-2201 ..;make -j; # 默认npu模式
  python3 ../scripts/gen_data.py
  ./demo
  ```

  使用 CPU调试 或 NPU仿真 模式时，添加 `-DRUN_MODE=cpu` 或 `-DRUN_MODE=sim` 参数即可。

  示例如下：
  ```bash
  cmake -DRUN_MODE=cpu -DNPU_ARCH=dav-2201 ..;make -j; # cpu调试模式
  cmake -DRUN_MODE=sim -DNPU_ARCH=dav-2201 ..;make -j; # NPU仿真模式
  ```

  > **注意：** 切换编译模式前需清理 cmake 缓存，可在 build 目录下执行 `rm CMakeCache.txt` 后重新 cmake。

- 编译选项说明

  | 选项 | 可选值 | 说明 |
  |------|--------|------|
  | `RUN_MODE` | `npu`（默认）、`cpu`、`sim` | 运行模式：NPU 运行、CPU调试、NPU仿真 |
  | `NPU_ARCH` | `dav-2201`（默认）、`dav-3510` | NPU 架构：dav-2201 对应 Atlas A2/A3 系列，dav-3510 对应 Ascend 950PR/Ascend 950DT |

- 执行结果

  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```
