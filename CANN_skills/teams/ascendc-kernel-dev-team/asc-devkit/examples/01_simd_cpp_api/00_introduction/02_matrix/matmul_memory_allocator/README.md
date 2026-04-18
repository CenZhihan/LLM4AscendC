# 基于静态Tensor编程实现Matmul计算

## 概述

本样例基于静态Tensor编程范式实现多核矩阵乘计算。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── normal_matmul
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── matmul.asc              // Ascend C样例实现 & 调用样例
```

## 样例描述

- 样例功能：

  Matmul计算公式：
  $$
  C = A * B
  $$
- 样例规格：

  本样例参数M = 512, N = 1024, K = 512，调用4个核完成计算，输入规格如下表所示：
  <table>
  <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">[M, K]</td><td align="center">float16</td><td align="center">ND</td></tr>
  <tr><td align="center">B</td><td align="center">[K, N]</td><td align="center">float16</td><td align="center">ND</td></tr>

  </tr>
  </tr>
  <tr><td rowspan="1" align="center">样例输出</td><td align="center">C</td><td align="center">[M, N]</td><td align="center">float16</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">matmul_custom</td></tr>
  </table>

- 样例实现：
  - 实现流程：
    <table>
    <tr><th align="left">步骤</th><th align="left">操作</th><th align="left">功能</th><th align="left">格式转换</th></tr>
    <tr><td align="left">1</td><td align="left">常量化Tiling参数</td><td align="left">通过模板参数传入kernel</td><td align="left">不涉及</td></tr>
    <tr><td align="left">2</td><td align="left">CopyInA、CopyInB</td><td align="left">将A矩阵和B矩阵数据从GM搬运到L1</td><td align="left">ND->NZ格式转换</td></tr>
    <tr><td align="left">3</td><td align="left">DataLoadA、DataLoadB</td><td align="left">将数据从L1搬运到L0A和L0B</td><td align="left">L1->L0A: NZ->NZ<br>L1->L0B: NZ->ZN</td></tr>
    <tr><td align="left">4</td><td align="left">Compute</td><td align="left">完成矩阵乘加计算</td><td align="left">矩阵乘结果为NZ格式</td></tr>
    <tr><td align="left">5</td><td align="left">CopyOut</td><td align="left">将L0C中的计算结果搬运到GM</td><td align="left">NZ->ND格式转换</td></tr>
    </table>
  - 约束条件

    1. baseM/baseK/baseN满足16对齐
    2. baseM/baseK/baseN能被singleCoreM/singleCoreK/singleCoreN整除
    3. singleCoreM/singleCoreK/singleCoreN能被M/K/N整除，不支持非整切场景

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
  mkdir -p build && cd build;   # 创建并进入build目录
  cmake -DNPU_ARCH=dav-2201 ..;make -j;             # 编译工程（默认NPU模式）
  python3 ../scripts/gen_data.py   # 生成测试输入数据
  ./demo                        # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```

  使用NPU仿真模式时，添加 `-DRUN_MODE=sim` 参数即可。

  示例如下：
  ```bash
  cmake -DRUN_MODE=sim -DNPU_ARCH=dav-2201 ..;make -j; # NPU仿真模式
  ```

  > **注意：** 切换编译模式前需清理 cmake 缓存，可在 build 目录下执行 `rm CMakeCache.txt` 后重新 cmake。

- 编译选项说明

  | 选项 | 可选值 | 说明 |
  |------|--------|------|
  | `RUN_MODE` | `npu`（默认）、`sim` | 运行模式：NPU 运行、CPU调试、NPU仿真 |
  | `NPU_ARCH` | `dav-2201`（默认）、`dav-3510` | NPU 架构，dav-2201 对应 Atlas A2/A3 系列，dav-3510 对应 Ascend 950PR/Ascend 950DT |

  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```