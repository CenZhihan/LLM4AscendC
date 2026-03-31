
# Batch Mmad样例

## 概述

本样例介绍在输入为float数据类型并且左、右矩阵均不转置的场景下，带batch的矩阵乘法，其中从GM-->L1、L0C-->GM、L0C-->L1这三条通路分别采用了DataCopy ND2NZ和Fixpipe批量搬运数据，从L1-->L0A/L0B以及Mmad执行矩阵乘这两个步骤则是循环batch次，每次循环内只处理一对左、右矩阵。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```plain
├── batch_mmad
│   ├── scripts
│   │   ├── gen_data.py             // 输入数据和真值数据生成脚本
│   │   └── verify_result.py        // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt              // 编译工程文件
│   ├── data_utils.h                // 数据读入写出函数
│   └── batch_mmad.asc                    // Ascend C算子实现 & 调用样例
```

## 算子描述

### 1.batch mmad定义

批量矩阵乘法（batch mmad）是普通矩阵乘法在批次维度（Batch Dimension） 上的扩展，核心逻辑是：对一个包含多个矩阵的批次数据，逐一对批次内的矩阵执行标准矩阵乘法，最终输出同批次数量的结果矩阵。

简单来说，若有两个批量矩阵 A 和 B，它们的形状分别为 [B, M, K] 和 [B, K, N]（其中 B 是批次大小，M/K/N 是矩阵维度），批量矩阵乘法会为每一个批次索引 i（i ∈ [0, B-1]），取 A[i]（形状 [M, K]）和 B[i]（形状 [K, N]）执行普通矩阵乘法，最终得到形状为 [B, M, N] 的批量结果矩阵 C。对任意批次 i（0 ≤ i < B），C 的第 i 个矩阵满足：
C[i]=A[i]×B[i]。

需要注意的是不同批次的矩阵之间不会互相计算。

### 2.矩阵批量搬入

根据batch mmad的定义可知，共计B对A、B矩阵进行矩阵乘法。数据从GM-->L1通路时，如下所示调用随路转换ND2NZ搬运接口时，通过配置`nd2nzA1Params.ndNum = B`，实现一次性搬入B对A、B矩阵。

```cpp
// GM-->L1,搬运A矩阵
AscendC::Nd2NzParams nd2nzA1Params;
// 传输ND矩阵的数目
nd2nzA1Params.ndNum = B;
// ND矩阵的行数
nd2nzA1Params.nValue = m;
// ND矩阵的列数
nd2nzA1Params.dValue = k;
// 源操作数相邻ND矩阵起始地址间的偏移,单位是元素
nd2nzA1Params.srcNdMatrixStride = m * k;
// 源操作数同一ND矩阵的相邻行起始地址间的偏移，单元是元素
nd2nzA1Params.srcDValue = k;

// ND转换到NZ格式后，源操作数中的一行会转换为目的操作数的多行。
// 该参数表示，目的NZ矩阵中，来自源操作数同一行的多行数据相邻行起始地址间的偏移，单位：C0_SIZE（32B）。
// 数据搬运到L1时会进行对齐
nd2nzA1Params.dstNzC0Stride = CeilAlign(m, cubeShape[0]);
// 目的NZ矩阵中，Z型矩阵相邻行起始地址之间的偏移。单位：C0_SIZE（32B）。
nd2nzA1Params.dstNzNStride = 1;
// 目的NZ矩阵中，相邻NZ矩阵起始地址间的偏移，单位是元素
nd2nzA1Params.dstNzMatrixStride = aSizeAlignL0;
```

### 3.mmad循环执行B次

由于mmad指令每次只能计算一对A、B矩阵矩阵乘的结果，因此Process()函数中在外层循环B次。

```cpp
// 循环batchSize次，迭代计算batchSize对A、B矩阵的矩阵乘结果
for (int32_t batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        SplitA(a1Local[batchIndex * aSizeAlignL0]);
        SplitBTranspose(b1Local[batchIndex * bSizeAlignL0]);
        Compute(batchIndex, c1Local);
}
```

for循环内部将每次计算得到的C[i]进行偏移后存储到输出tensor的相应位置，在最后一次迭代中将输出tensor放入输出队列中。

```cpp
AscendC::Mmad(c1Local[batchIndex * CeilAlign(m, cubeShape[0]) * CeilAlign(n, cubeShape[0])],
        a2Local, b2Local, mmadParams);
if (batchIndex == B - 1) {
        outQueueCO1.EnQue<float>(c1Local);
} 
```

### 4.矩阵批量搬出

数据从L0C-->GM通路时，如下所示调用fixpipe搬运接口时，通过配置fixpipeParams.ndNum = B，实现一次性搬出B对C矩阵。注意的是L0C中的C矩阵是对齐后的，而搬出到GM的C矩阵是原始非对齐的shape。

```cpp
// 源NZ矩阵在N方向上的大小。
fixpipeParams.nSize = n;
// 源NZ矩阵在M方向上的大小。
fixpipeParams.mSize = m;
// 源NZ矩阵中相邻Z排布的起始地址偏移，单位：C0_Size(16*sizeof(T)，T为src的数据类型)
fixpipeParams.srcStride = CeilAlign(m, cubeShape[0]);
// 使能NZ2ND功能时，代表目的ND矩阵每一行中的元素个数，取值不为0 ，单位：元素
fixpipeParams.dstStride = n;
// 源NZ矩阵的数目，也就是传输ND矩阵的数目
fixpipeParams.ndNum = B;
// 不同NZ矩阵起始地址之间的间隔，单位：1024B
fixpipeParams.srcNdStride = (CeilAlign(m, cubeShape[0]) * CeilAlign(n, cubeShape[0])) 
                                / (cubeShape[0] * cubeShape[0]);
// 目的相邻ND矩阵起始地址之间的偏移，单位：element
```

### 5.避免数据占用总内存超过存储空间限制

用户应该保证batch mmad整个过程中的数据所占总的内存不超过存储空间限制。
用户可以通过PlatformAscendC类成员函数[GetCoreMemSize](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/API/ascendcopapi/atlasascendc_api_07_1034.html)，获取获取硬件平台中L1、L0A、L0B、L0C存储空间的内存大小。

## 编译运行

在本样例根目录下执行如下步骤，编译并执行算子。

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
  B=4 M=30 K=40 N=70
  mkdir -p build && cd build;      # 创建并进入build目录
  cmake .. -DB_SIZE=$B -DM_SIZE=$M -DK_SIZE=$K -DN_SIZE=$N;make -j;    # 编译工程
  python3 ../scripts/gen_data.py -b=$B -m=$M -k=$K -n=$N   # 生成测试输入数据
  ./demo                           # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```

  执行结果如下，说明精度对比成功。

  ```bash
  test pass
  ```
