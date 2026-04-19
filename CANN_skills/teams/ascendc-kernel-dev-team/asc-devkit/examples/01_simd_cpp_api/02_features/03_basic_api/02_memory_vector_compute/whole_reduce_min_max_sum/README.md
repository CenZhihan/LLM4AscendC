# 归约类接口多场景示例

## 概述

本样例介绍归约类接口在多种场景下的使用方法，包括WholeReduceMax、WholeReduceMin、WholeReduceSum、RepeatReduceSum，以及WholeReduceMax/Min配合GetReduceRepeatMaxMinSpr获取全局极值及索引的使用方法。 这些接口用于对LocalTensor中每个repeat内所有元素进行归约运算（求最大值、最小值或求和），归约结果存放到目的LocalTensor中。

注：`GetReduceRepeatMaxMinSpr`为 CANN 9.0.0 重命名后的 API。CANN 8.5.0 及之前版本请使用`GetReduceMaxMinCount`。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── whole_reduce_min_max_sum
│   ├── scripts
│   │   ├── gen_data.py                    // 输入数据和真值数据生成脚本
│   │   └── verify_result.py              // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt                    // 编译工程文件
│   ├── data_utils.h                      // 数据读入写出函数
│   └── whole_reduce_min_max_sum.asc      // Ascend C样例实现 & 调用样例
```

## 场景详细说明

本样例通过编译参数 `SCENARIO_NUM` 选择不同的归约场景，所有场景数据格式为 ND，核函数名为 `reduce_custom`。

**场景1：WholeReduceMax**
- 输入：[1, 1024]个half元素，mask=128，repeat=8（1024/128）
- 输出：[1, 8]个half元素（8个最大值，不返回索引）
- 实现：`WholeReduceMax<half>(dstLocal, srcLocal, mask=128, repeat=8, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_VALUE)`
- 说明：使用ORDER_ONLY_VALUE参数，只返回最大值不返回索引，输出存储顺序为[max0, max1, max2, ...]

**场景2：WholeReduceMin**
- 输入：[1, 1024]个half元素，mask=128，repeat=8（1024/128）
- 输出：[1, 16]个half元素（8个最小值 + 8个索引，交错存储）
- 实现：`WholeReduceMin<half>(dstLocal, srcLocal, mask=128, repeat=8, 1, 1, 8)`
- 说明：使用默认order（ORDER_VALUE_INDEX），输出按 [min0, idx0, min1, idx1, ...] 交错存储

**场景3：WholeReduceSum**
- 输入：[1, 2048]个float元素，mask=64（float类型32位，mask范围为[1,64]），repeat=32（2048/64）
- 输出：[1, 32]个float元素（每个repeat的求和结果）
- 实现：`WholeReduceSum<float>(dstLocal, srcLocal, mask=64, repeat=32, 1, 1, 8)`
- 说明：每个repeat独立求和，共输出32个求和结果

**场景4：RepeatReduceSum**
- 输入：[1, 2048]个float元素，mask=64（float类型32位，mask范围为[1,64]），repeat=32（2048/64）
- 输出：[1, 32]个float元素（累加模式）
- 实现：先用 `Duplicate` 初始化dstLocal为0，再调用 `RepeatReduceSum<float>(dstLocal, srcLocal, repeat=32, mask=64, dstBlkStride=0, 1, 1, 8)`
- 说明：dstBlkStride=0表示将所有repeat的结果累加到同一位置，每个repeat结果依次存放

**场景5：WholeReduceMax + GetReduceRepeatMaxMinSpr**
- 输入：[1, 1024]个half元素，mask逐bit模式（uint64_t[2]全1），repeat=8（1024/128）
- 输出：[1, 16]个half元素（仅前2个元素有效：全局最大值 + 全局最大值索引）
- 实现：先调用 `WholeReduceMax<half>(dstLocal, srcLocal, mask, repeat=8, 1, 1, 8)` 使用逐bit mask进行多repeat归约，再调用 `GetReduceRepeatMaxMinSpr<half>(val, idx)` 从硬件寄存器获取全局最大值及其索引，最后通过 `SetFlag<HardEvent::V_S>` / `WaitFlag<HardEvent::V_S>` 同步后写入dstLocal
- 说明：WholeReduceMax对8个repeat分别求局部最大值，GetReduceRepeatMaxMinSpr从硬件寄存器中读取所有repeat中的全局最大值及其在源数据中的索引位置，结果写入dstLocal的前两个元素

**场景6：WholeReduceMin + GetReduceRepeatMaxMinSpr**
- 输入：[1, 1024]个half元素，mask逐bit模式（uint64_t[2]全1），repeat=8（1024/128）
- 输出：[1, 16]个half元素（仅前2个元素有效：全局最小值 + 全局最小值索引）
- 实现：先调用 `WholeReduceMin<half>(dstLocal, srcLocal, mask=uint64_t[2]{-1,-1}, repeat=8, 1, 1, 8)`，再调用 `GetReduceRepeatMaxMinSpr<half>(val, idx)` 获取全局最小值及其索引，通过 `SetFlag<HardEvent::V_S>` / `WaitFlag<HardEvent::V_S>` 同步向量计算到标量计算
- 说明：WholeReduceMin对8个repeat分别求局部最小值，GetReduceRepeatMaxMinSpr从硬件寄存器中读取所有repeat中的全局最小值及其在源数据中的索引位置，结果写入dstLocal的前两个元素

**场景7：WholeReduceSum非对齐场景**
- 输入：[13, 57]个float元素（13行×57列，列数57×4字节=228字节，非32字节对齐）
- 输出：[1, 13]个float元素（每行的求和结果）
- 实现：使用 `DataCopyPad` 搬入非对齐数据，`WholeReduceSum<float>(dstLocal, srcLocal, mask=57, repeat=13, 1, 1, srcStride)` 对每行求和
- 说明：演示非对齐数据场景下的归约操作。每行57个float元素（228字节），使用DataCopyPad搬运时按32字节对齐填充至232字节（58个float元素），WholeReduceSum的srcRepeatStride按对齐后的block数计算

## 样例规格

<table border="2">
<caption>表1：样例输入输出规格（场景1）</caption>
<tr><td rowspan="2" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[1, 1024]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">样例输出</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">z</td><td align="center">[1, 8]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">reduce_custom</td></tr>
</table>

<table border="2">
<caption>表2：样例输入输出规格（场景2/5/6）</caption>
<tr><td rowspan="2" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[1, 1024]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">样例输出</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">z</td><td align="center">[1, 16]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">reduce_custom</td></tr>
</table>

<table border="2">
<caption>表3：样例输入输出规格（场景3/4）</caption>
<tr><td rowspan="2" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[1, 2048]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">样例输出</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">z</td><td align="center">[1, 32]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">reduce_custom</td></tr>
</table>

<table border="2">
<caption>表4：样例输入输出规格（场景7）</caption>
<tr><td rowspan="2" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[13, 57]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">样例输出</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">z</td><td align="center">[1, 13]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">reduce_custom</td></tr>
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
mkdir -p build && cd build;
cmake .. -DSCENARIO_NUM=$SCENARIO_NUM -DRUN_MODE=cpu -DNPU_ARCH=dav-2201;make -j;
python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO_NUM
./demo
python3 ../scripts/verify_result.py -scenarioNum=$SCENARIO_NUM ./output/output.bin ./output/golden.bin
```

**NPU仿真模式：**
```bash
SCENARIO_NUM=1
mkdir -p build && cd build;
cmake .. -DSCENARIO_NUM=$SCENARIO_NUM -DRUN_MODE=sim -DNPU_ARCH=dav-2201;make -j;
python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO_NUM
./demo
python3 ../scripts/verify_result.py -scenarioNum=$SCENARIO_NUM ./output/output.bin ./output/golden.bin
```

> **注意：** 切换编译模式前需清理 cmake 缓存，可在 build 目录下执行 `rm CMakeCache.txt` 后重新 cmake。

执行结果如下，说明精度对比成功。
```bash
test pass!
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
  SCENARIO_NUM=1
  mkdir -p build && cd build;      # 创建并进入build目录
  cmake .. -DSCENARIO_NUM=$SCENARIO_NUM;make -j;    # 编译工程
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO_NUM   # 生成测试输入数据
  ./demo                           # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py -scenarioNum=$SCENARIO_NUM ./output/output.bin ./output/golden.bin  # 验证输出结果是否正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```
