# Gather类样例

## 概述

本样例基于[GatherMask](../../../../../../docs/api/context/GatherMask.md)、[Gather](../../../../../../docs/api/context/Gather.md)、[Gatherb](../../../../../../docs/api/context/GatherB.md)等接口完成多种场景模式下的数据选择功能，包括内置固定模式、用户自定义模式、张量偏移模式、DataBlock偏移模式，实现从源操作数中选取元素写入目的操作数。样例支持通过编译参数切换不同场景，便于开发者理解Gather类接口的使用方法和实现差异。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── gather
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── gather.asc              // Ascend C样例实现 & 调用样例
```

## 场景详细说明
本样例通过编译参数`SCENARIO_NUM` 来切换不同的掩码生成场景：

**场景1：内置固定模式**
- 说明：通过`src1Pattern`选择对应的二进制作为掩码，来获取数据
- 输入：src0Local=[1, 256]
- 输出：[1, 256]
- 数据类型：uint32
- 实现：
    ```cpp
    AscendC::GatherMask(dstLocal, src0Local, src1Pattern, reduceMode, mask, gatherMaskParams, rsvdCnt);
    ```
- 参数：dstLocal和src0Local采用地址复用，并使用内置固定模式src1Pattern=2进行元素选取，reduceMode=false（Normal模式），mask=0，gatherMaskParams={1, 4, 8, 0}

**场景2：用户自定义模式**
- 说明：通过用户输入的`src1Local`对应的二进制作为掩码，来获取数据
- 输入：src0Local=[1, 256], src1Local=[1, 32]
- 输出：[1, 256]
- 数据类型：uint32
- 实现：
    ```cpp
    AscendC::GatherMask (dstLocal, src0Local, src1Local, reduceMode, mask, gatherMaskParams, rsvdCnt);
    ```
- 参数：使用用户提供的Tensor进行元素选取，reduceMode=true（Counter模式），mask=70，gatherMaskParams={1, 2, 4, 0}

**场景3：张量偏移模式**
- 说明：根据用户输入的地址偏移张量`srcOffset`进行地址偏移，来获取数据
- 输入：src0Local=[1, 128], srcOffset=[1, 128]
- 输出：[1, 128]
- 数据类型：输入输出uint16，srcOffset类型为uint32
- 实现：
    ```cpp
    AscendC::Gather(dstLocal, src0Local, srcOffset, srcBaseAddr, count);
    ```
- 参数：使用用户提供的srcOffset按元素地址偏移，srcBaseAddr=0表示源操作数的起始地址，count=128表示执行处理的数据个数

**场景4：DataBlock偏移模式**
- 说明：根据用户输入的地址偏移张量`srcOffset`（按照DataBlock的粒度）进行地址偏移，来获取数据
- 输入：src0Local=[1, 128], srcOffset=[1, 8]
- 输出：[1, 128]
- 数据类型：输入输出uint16，srcOffset类型为uint32
- 实现：
    ```cpp
    AscendC::Gatherb<T>(dstLocal, src0Local, srcOffset, repeatTime, params);
    ```
- 参数：用户输入的srcOffset是每个datablock在源操作数中对应的地址偏移，repeatTime=1表示重复迭代次数，params={1,8}

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
  SCENARIO_NUM=1  # 设置场景编号
  mkdir -p build && cd build;      # 创建并进入build目录
  cmake .. -DNPU_ARCH=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j;    # 编译工程
  python3 ../scripts/gen_data.py -scenario_num=$SCENARIO_NUM   # 生成测试输入数据
  ./demo                           # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py ./output/output.bin ./output/golden.bin  # 验证输出结果是否正确
  ```

  使用CPU调试或NPU仿真模式时，添加 `-DRUN_MODE=cpu` 或 `-DRUN_MODE=sim` 参数即可。
  示例如下：
 	```bash
 	cmake -DRUN_MODE=cpu -DNPU_ARCH=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j; # CPU调试模式
  cmake -DRUN_MODE=sim -DNPU_ARCH=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j; # NPU仿真模式
 	```
  若需详细了解CPU调试相关内容，请参考[03_cpudebug样例](../../../01_utilities/03_cpudebug)。

- 编译选项说明

| 选项 | 可选值 | 说明 |
|------|--------|------|
| `RUN_MODE` | `npu`（默认）、`cpu`、`sim` | 运行模式：NPU 运行、CPU调试、NPU仿真 |
| `NPU_ARCH` | `dav-2201`（默认）、`dav-3510` | NPU 架构：dav-2201 对应 Atlas A2/A3 系列、dav-3510 对应 Ascend 950PR/Ascend 950DT |
| `SCENARIO_NUM` | `1`（默认）、`2`、`3`、`4` | 场景编号：1（内置固定模式）、2（用户自定义模式）、3（张量偏移模式）、4（DataBlock偏移模式） |

- 执行结果
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```