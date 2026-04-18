# BlockReduce类接口多场景示例

## 概述

本样例在归约场景下，基于BlockReduceMax、BlockReduceMin、BlockReduceSum实现BlockReduce类接口的多场景归约功能，对输入Tensor的每个datablock（32字节）内所有元素进行归约运算（求最大值、最小值或求和）。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── block_reduce_min_max_sum
│   ├── scripts
│   │   ├── gen_data.py                    // 输入数据和真值数据生成脚本
│   │   └── verify_result.py              // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt                    // 编译工程文件
│   ├── data_utils.h                      // 数据读入写出函数
│   └── block_reduce_min_max_sum.asc      // Ascend C样例实现 & 调用样例
```

## 场景说明

本样例通过编译参数 `SCENARIO_NUM` 选择不同的归约场景，所有场景数据格式为 ND，核函数名为 `block_reduce_custom`。

**场景1：BlockReduceMax<half>**
- 输入：[1, 256]个half元素，mask=128（256/sizeof(half)），repeat=2
- 输出：[1, 16]个half元素（对应16个datablock各自的最大值）
- 实现：`BlockReduceMax<half>(dstLocal, srcLocal, repeat=2, mask=128, dstRepStride=1, srcBlkStride=1, srcRepStride=8)`
- 说明：对每个datablock内所有元素求最大值，一个datablock处理32字节即16个half元素，256个元素共16个datablock，输出16个最大值

**场景2：BlockReduceMin<half>**
- 输入：[4, 128]个half元素，mask=128（256/sizeof(half)），repeat=4
- 输出：[4, 8]个half元素（对应32个datablock各自的最小值）
- 实现：`BlockReduceMin<half>(dstLocal, srcLocal, repeat=4, mask=128, dstRepStride=1, srcBlkStride=1, srcRepStride=8)`
- 说明：对每个datablock内所有元素求最小值，一个datablock处理32字节即16个half元素，512个元素共32个datablock，输出32个最小值

**场景3：BlockReduceSum<float>**
- 输入：[1, 128]个float元素，mask=64（256/sizeof(float)），repeat=2
- 输出：[1, 16]个float元素（对应16个datablock各自的求和结果）
- 实现：`BlockReduceSum<float>(dstLocal, srcLocal, repeat=2, mask=8, dstRepStride=1, srcBlkStride=1, srcRepStride=8)`
- 说明：对每个datablock内所有元素求和，源操作数相加采用二叉树方式两两相加，一个datablock处理32字节即8个float元素，128个元素共16个datablock，输出16个求和结果

## 样例规格

<table border="2">
<caption>表1：样例输入输出规格（场景1：BlockReduceMax）</caption>
<tr><td rowspan="2" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[1, 256]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">样例输出</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">y</td><td align="center">[1, 16]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">block_reduce_custom</td></tr>
</table>

<table border="2">
<caption>表2：样例输入输出规格（场景2：BlockReduceMin）</caption>
<tr><td rowspan="2" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[4, 128]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">样例输出</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">y</td><td align="center">[4, 8]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">block_reduce_custom</td></tr>
</table>

<table border="2">
<caption>表3：样例输入输出规格（场景3：BlockReduceSum）</caption>
<tr><td rowspan="2" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[1, 128]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">样例输出</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">y</td><td align="center">[1, 16]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">block_reduce_custom</td></tr>
</table>

## 编译运行

在本样例根目录下执行如下步骤，编译并执行样例。

### 编译选项说明

| 参数 | 说明 | 可选值 | 默认值 |
|------|------|---------|--------|
| RUN_MODE | 运行模式 | npu, cpu, sim | npu |
| NPU_ARCH | NPU硬件架构 | dav-2201, dav-3510 | dav-2201 |

### 配置环境变量

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

### 样例执行

**NPU模式（默认）：**
```bash
SCENARIO_NUM=1
mkdir -p build && cd build;      # 创建并进入build目录
cmake .. -DSCENARIO_NUM=$SCENARIO_NUM -DNPU_ARCH=dav-2201;make -j;    # 编译工程
python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO_NUM   # 生成测试输入数据
./demo                           # 执行编译生成的可执行程序，执行样例
python3 ../scripts/verify_result.py -scenarioNum=$SCENARIO_NUM ./output/output.bin ./output/golden.bin  # 验证输出结果是否正确
```

**CPU调试模式：**
```bash
SCENARIO_NUM=1
mkdir -p build && cd build;      # 创建并进入build目录
cmake .. -DSCENARIO_NUM=$SCENARIO_NUM -DRUN_MODE=cpu -DNPU_ARCH=dav-2201;make -j;    # 编译工程
python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO_NUM   # 生成测试输入数据
./demo                           # 执行编译生成的可执行程序，执行样例
python3 ../scripts/verify_result.py -scenarioNum=$SCENARIO_NUM ./output/output.bin ./output/golden.bin  # 验证输出结果是否正确
```

**NPU仿真模式：**
```bash
SCENARIO_NUM=1
mkdir -p build && cd build;      # 创建并进入build目录
cmake .. -DSCENARIO_NUM=$SCENARIO_NUM -DRUN_MODE=sim -DNPU_ARCH=dav-2201;make -j;    # 编译工程
python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO_NUM   # 生成测试输入数据
./demo                           # 执行编译生成的可执行程序，执行样例
python3 ../scripts/verify_result.py -scenarioNum=$SCENARIO_NUM ./output/output.bin ./output/golden.bin  # 验证输出结果是否正确
```

> **注意：** 切换编译模式前需清理 cmake 缓存，可在 build 目录下执行 `rm CMakeCache.txt` 后重新 cmake。

执行结果如下，说明精度对比成功。
```bash
test pass!
```
