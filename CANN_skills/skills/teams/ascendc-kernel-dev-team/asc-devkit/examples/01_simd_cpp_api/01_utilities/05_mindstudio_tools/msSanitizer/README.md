# msSanitizer样例

## 概述

开发者可以使用算子异常检测工具msSanitizer来在早期阶段发现并修复异常，确保算子的质量和稳定性。本样例以静态Tensor编程方法的add算子来展现该工具如何检测异常。

请参考[算子开发工具](https://www.hiascend.com/document/redirect/CannCommercialToolOpDev)中的“环境准备”章节，获取详细的安装指南和步骤。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── 06_msSanitizer
│   ├── scripts
│   │   ├── gen_data.py             // 输入数据和真值数据生成脚本
│   │   └── verify_result.py        // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt              // 编译工程文件
│   ├── data_utils.h                // 数据读入写出函数
│   └── add_custom.asc              // Ascend C算子实现 & 调用样例
```

## 算子描述

算子实现的是固定shape为72×4096的Add算子。

Add的计算公式为：

```python
z = x + y
```

- x：输入，形状为\[72, 4096]，数据类型为float；
- y：输入，形状为\[72, 4096]，数据类型为float；
- z：输出，形状为\[72, 4096]，数据类型为float；

## 算子规格描述

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">72 * 4096</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">72 * 4096</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">72 * 4096</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>
</table>

- 算子实现：

  Add算子的数学表达式为：

  ```
  z = x + y
  ```

  计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用计算接口完成两个输入参数相加，得到最终结果，再搬出到外部存储上。

  Add算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor xGm和yGm搬运到Local Memory，分别存储在xLocal、yLocal，Compute任务负责对xLocal、yLocal执行加法操作，计算结果存储在zLocal中，CopyOut任务负责将输出数据从zLocal搬运至Global Memory上的输出Tensor zGm中。

## 异常场景检测

本样例代码为正确实现。用户可以按照下述方式来复现各个异常场景，体验msSanitizer的异常检测能力。
- **内存检测**
  - 非法读写：由于访问了未分配的内存导致的异常。
    
    用户可以注释掉正确的DataCopy，使用错误的DataCopy复现该场景。LocalTensor xLocal分配的大小为TILE_LENGTH，但是搬运中大小错误的填为TILE_LENGTH * 2，大于xLocal分配的大小，因此触发非法读写。
    ```
    // 1. correct datacopy
    AscendC::DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);    
    // 2. illegal read of xGm (TILE_LENGTH*2)
    // AscendC::DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH * 2);
    ```

    工具报错内容：
    ```
    ====== ERROR: illegal write of size 16384
    ======    at 0x0 on UB in add_custom
    ======    in block aiv(0-7) on device 0
    ```

  - 非对齐访问：内存访问未满足字节对齐要求
    
    用户可以注释掉正确的DataCopy，使用错误的DataCopy复现该场景。DataCopy GM->UB的搬运中，UB侧地址应该要满足32B对齐，但是xLocal[5]不满足32B对齐(5 * sizeof(float) = 20)，因此触发非对齐访问。
    ```
    // 1. correct datacopy
    AscendC::DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);    
    // 3. misaligned access of xLocal (should be 32Byte aligned)
    // AscendC::DataCopy(xLocal[5], xGm[i * TILE_LENGTH], TILE_LENGTH);
    ```

    工具报错内容：
    ```
    ====== ERROR: misaligned access of size 32
    ======    at 0x14 on UB in add_custom
    ======    in block aiv(0-7) on device 0
    ```
  - 内存泄漏：申请内存使用后未释放，导致程序在运行过程中内存占用持续增加的异常
    
    用户可以注释掉这行aclrtFree来复现该场景。注释前tilingDevice被正常释放，注释后tilingDevice在使用后未释放，因此触发内存泄漏。

    注意：调用mssanitizer时需要传入--leak-check=yes来开启分配内存泄露检查。
    ```
    // 1. correct free for memory. If deleted, it will trigger memory leak check.  
    aclrtFree(tilingDevice);
    ```

    工具报错内容：
    ```
    ====== ERROR: LeakCheck: detected memory leaks

    ======    Direct leak of 64 byte(s)
    ======      at 0x12c0c0013000 on GM
    ======      allocated in :0 (serialNo:0)
    ```
  - 分配内存未使用：对内存分配后未使用导致的异常
    
    用户可以注释掉正确的aclrtMalloc, 使用错误的aclrtMalloc来复现该场景。inputDevice[i]需要分配的大小为inputsInfo[i].length，但是实际分配了inputsInfo[i].length * 5，其中有inputsInfo[i].length * 4未使用，因此触发分配内存未使用。

    注意：调用mssanitizer时需要传入--check-unused-memory=yes来开启分配内存未使用检查。
    ```
    // 1. correct malloc for memory
    aclrtMalloc((void **)(&inputDevice[i]), inputsInfo[i].length, ACL_MEM_MALLOC_HUGE_FIRST);
    // 2. needed inputsInfo[i].length, but malloc length * 5, therefore trigger unused memory
    // aclrtMalloc((void **)(&inputDevice[i]), inputsInfo[i].length * 5, ACL_MEM_MALLOC_HUGE_FIRST);
    ```

    工具报错内容：
    ```
    ====== WARNING: Unused memory of 4718624 byte(s)
    ======    at 0x12c041200000 on GM
    ======    code in :0 (serialNo:260)
    ```

- **竞争检测**
  - 竞争检测：用于解决在并行计算环境中内存访问竞争的问题。
    
    用户可以注释掉SetFlag和WaitFlag来复现该场景。SetFlag和WaitFlag是用来保证MTE2 GM->UB的搬运和Vector计算Add的时序，删除后可能会导致先计算Add再搬运数据导致精度异常，因此触发竞争检测。

    注意：调用mssanitizer时需要传入--tool=racecheck来开启竞争检测。
    ```
    // dependency of PIPE_MTE2 & PIPE_V caused by xLocal/yLocal in one single loop
    // If SetFlag and WaitFlag are deleted, will trigger RAW 
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    ```

    工具报错内容：
    ```
    ====== ERROR: Potential RAW hazard detected at UB in add_custom on device 0:
    ======    PIPE_MTE2 Write at RAW()+0x4000 in block 0 (aiv) on device 0 at pc current 0xd08 (serialNo:25)
    ======    xxxxx
    ======    PIPE_V Read at RAW()+0x4000 in block 0 (aiv) on device 0 at pc current 0x1578 (serialNo:28)
    ======    xxxxx
    ```

- **未初始化检测**
  - 未初始化检测：内存申请后为未初始化状态，未对内存进行写入，直接读取未初始化的值导致的异常。
    
    用户可以注释掉下述的SetGlobalBuffer来复现该场景。对于device侧的zGm在使用前未初始化，因此触发未初始化检测。

    注意：调用mssanitizer时需要传入--tool=initcheck来开启竞争检测。
    ```
    // correct initialize of zGm. 
    // If deleted, it will trigger uninitialized read
    zGm.SetGlobalBuffer((__gm__ float *)z + AscendC::GetBlockIdx() * singleCoreLength, singleCoreLength);
    ```

    工具报错内容：
    ```
    ====== ERROR: uninitialized read of size 1179648
    ======    at 0x12c041600000 on GM
    ```

## 编译运行

在本样例根目录下执行如下步骤，编译并执行算子。
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
  mkdir -p build && cd build;      # 创建并进入build目录
  cmake ..;make -j;                # 编译工程
  python3 ../scripts/gen_data.py   # 生成测试输入数据

  # 执行编译生成的可执行程序，使用mssanitizer执行样例
  # 根据业务需求执行对应的mssanitizer命令
  mssanitizer ./demo                             # 非法读写 / 非对齐访问
  mssanitizer ./demo --leak-check=yes            # 开启内存泄漏检查
  mssanitizer ./demo --check-unused-memory=yes   # 开启分配内存未使用检查
  mssanitizer ./demo --tool=racecheck            # 竞争检测
  mssanitizer ./demo --tool=initcheck            # 未初始化检测
                           
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```
