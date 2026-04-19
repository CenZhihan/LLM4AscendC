# PairReduceSum样例

## 概述

本样例在归约场景下，基于接口PairReduceSum对数据（a1, a2, a3, a4, a5, a6...）的相邻两个元素求和为（a1+a2, a3+a4, a5+a6, ......），即对每个pair（偶数下标与奇数下标组成的相邻元素对）进行求和归约，输出元素个数为输入的一半。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── pair_reduce_sum
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── pair_reduce_sum.asc     // Ascend C 样例实现 & 调用
```

## 样例描述

- 样例功能：  

  本样例中调用PairReduceSum接口，对输入元素中所有的相邻元素对执行求和计算。样例规格如下表所示：

  <table border="2">
  <caption>表1：PairReduceSum样例规格</caption>
  <tr>
  <td rowspan="1" align="center">样例类型(OpType)</td>
  <td colspan="4" align="center">PairReduceSum</td></tr>
  <tr><td rowspan="2" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 128]</td><td align="center">half</td><td align="center">ND</td></tr>
   <tr><td rowspan="2" align="center">样例输出</td></tr>
   <tr><td align="center">y</td><td align="center">[1, 64]</td><td align="center">half</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">pair_reduce_sum_custom</td></tr>
  </table>

- 样例实现：  
  本样例中实现的是固定shape为输入x[1, 128]，输出y[1, 64]的PairReduceSum样例。

  Compute任务负责对srcLocal中每对相邻元素（偶数下标与奇数下标）求和，并将结果存储在dstLocal中。

## 编译运行

在本样例根目录下执行如下步骤，编译并运行样例。

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
mkdir -p build && cd build;   # 创建并进入build目录
cmake .. -DNPU_ARCH=dav-2201;make -j;             # 编译工程
python3 ../scripts/gen_data.py   # 生成测试输入数据
./demo                        # 执行编译生成的可执行程序，执行样例
python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
```

**CPU调试模式：**
```bash
mkdir -p build && cd build;
cmake .. -DRUN_MODE=cpu -DNPU_ARCH=dav-2201;make -j;
python3 ../scripts/gen_data.py
./demo
python3 ../scripts/verify_result.py output/output.bin output/golden.bin
```

**NPU仿真模式：**
```bash
mkdir -p build && cd build;
cmake .. -DRUN_MODE=sim -DNPU_ARCH=dav-2201;make -j;
python3 ../scripts/gen_data.py
./demo
python3 ../scripts/verify_result.py output/output.bin output/golden.bin
```

> **注意：** 切换编译模式前需清理 cmake 缓存，可在 build 目录下执行 `rm CMakeCache.txt` 后重新 cmake。

执行结果如下，说明精度对比成功。
```bash
test pass!
```
