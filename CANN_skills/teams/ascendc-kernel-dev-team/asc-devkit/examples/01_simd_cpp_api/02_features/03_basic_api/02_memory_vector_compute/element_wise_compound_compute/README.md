# ElementWiseCompoundCompute样例

## 概述

本样例展示了复合计算类接口的使用方法。复合计算接口将多个计算操作融合在一条指令中完成，相比分开调用多个基础接口，可有效减少指令数量、降低中间存储开销、提升计算效率。接口资料参考CastDequant/AddRelu/Axpy。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── element_wise_compound_compute
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── element_wise_compound_compute.asc    // Ascend C样例实现 & 调用样例
```

## 样例描述

- 样例规格：
  <table border="2">
  <caption>表1：样例规格对照表</caption>
  <tr>
    <th align="left">场景编号(SCENARIO_NUM)</th>
    <th align="left">接口名称</th>
    <th align="left">功能说明</th>
    <th align="left">计算公式</th>
    <th align="left">输入类型</th>
    <th align="left">输出类型</th>
  </tr>
  <tr>
    <td align="left">1</td>
    <td align="left">CastDequant</td>
    <td align="left">反量化与类型转换融合</td>
    <td align="left">dst = (src * scale) + offset</td>
    <td align="left">int16</td>
    <td align="left">uint8</td>
  </tr>
  <tr>
    <td align="left">2</td>
    <td align="left">AddRelu</td>
    <td align="left">向量加法与ReLU激活融合</td>
    <td align="left">dst = max(src0 + src1, 0)</td>
    <td align="left">half</td>
    <td align="left">half</td>
  </tr>
  <tr>
    <td align="left">3</td>
    <td align="left">Axpy</td>
    <td align="left">标量乘法与向量加法融合</td>
    <td align="left">dst = dst + src * scalar</td>
    <td align="left">half</td>
    <td align="left">half</td>
  </tr>
  </table>

  输入输出shape均为[1, 512]，format为ND，核函数名为`element_wise_compound_compute_custom`。

- 样例实现：
  输入数据从Global Memory搬运至LocalTensor，根据SCENARIO_NUM编译参数执行对应的复合计算，结果再搬出到Global Memory。实现流程包含CopyIn、Compute、CopyOut三个任务，使用内核调用符`<<<>>>`调用核函数。

## 编译运行  

在本样例根目录下执行如下步骤，编译并执行样例。
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
  ```bash
  SCENARIO_NUM=2
  mkdir -p build && cd build
  cmake -DNPU_ARCH=dav-2201 .. -DSCENARIO_NUM=$SCENARIO_NUM
  make -j
  python3 ../scripts/gen_data.py -scenario=$SCENARIO_NUM
  ./demo
  python3 ../scripts/verify_result.py -scenario=$SCENARIO_NUM output/output.bin output/golden.bin
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```

  使用 CPU调试 或 NPU仿真 模式时，添加 `-DRUN_MODE=cpu` 或 `-DRUN_MODE=sim` 参数即可。

  示例如下：
  ```bash
  cmake -DRUN_MODE=cpu -DNPU_ARCH=dav-2201 .. -DSCENARIO_NUM=$SCENARIO_NUM  # cpu调试模式
  cmake -DRUN_MODE=sim -DNPU_ARCH=dav-2201 .. -DSCENARIO_NUM=$SCENARIO_NUM  # NPU仿真模式
  ```

  > **注意：** 切换编译模式前需清理 cmake 缓存，可在 build 目录下执行 `rm CMakeCache.txt` 后重新 cmake。

- 编译选项说明

  | 选项 | 可选值 | 说明 |
  |------|--------|------|
  | `RUN_MODE` | `npu`（默认）、`cpu`、`sim` | 运行模式：NPU 运行、CPU调试、NPU仿真 |
  | `NPU_ARCH` | `dav-2201`（默认）、`dav-3510` | NPU 架构：dav-2201 对应 Atlas A2/A3 系列，dav-3510 对应 Ascend 950PR/Ascend 950DT |