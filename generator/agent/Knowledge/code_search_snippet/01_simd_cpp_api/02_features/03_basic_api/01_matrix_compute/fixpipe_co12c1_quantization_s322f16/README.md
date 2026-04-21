# fixpipe_co12c1_quantization_s322f16样例
## 概述
本样例介绍如何使用基础API Fixpipe将矩阵乘的结果从CO1搬入C1，并使能随路量化将矩阵乘结果由int32_t类型数据量化为half类型。

## 支持的产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍
```
├── fixpipe_co12c1_quantization_s322f16
│   ├── scripts
│   │   ├── gen_data.py                                   // 输入数据和真值数据生成脚本
│   │   └── verify_result.py                              // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt                                    // 编译工程文件
│   ├── data_utils.h                                      // 数据读入写出函数
│   └── fixpipe_co12c1_quantization_s322f16.asc           // Ascend C算子实现 & 调用样例
```

## 算子描述
- 算子功能：  

  本样例中实现的是[m, n, k]固定为[32, 32, 32]的Matmul乘算子，对应的数学表达式为：
  ```
  C = A * B
  ```
  其中A的形状为[M, K], B的形状为[K, N], C的形状为[M, N]。

- 算子规格：

  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">(32, 32)</td><td align="center">int8_t</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">(32, 32)</td><td align="center">int8_t</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">(32, 32)</td><td align="center">half</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">fixpipe_co12c1_quantization_s322f16</td></tr>
  </table>
- 算子实现：  
  - Kernel实现
    计算逻辑是：Ascend C提供的矩阵乘计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储并进行分形转换，然后使用计算接口完成两个输入参数矩阵乘运算，得到最终结果，再搬出到外部存储上。  

    Matmul算子的实现流程分为基本任务：CopyIn，SplitA，SplitB，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入inputGM搬运到Local Memory A1/B1中，搬运过程中进行ND至NZ分形转换。SplitA/SplitB分别将数据进一步搬运至接口所要求Local Memory A2/B2，Compute任务负责对数据进行矩阵乘运算，计算结果存储在Local Memory CO1中。CopyOut任务负责将输出数据从CO1搬运至Local Memory C1，使能量化将int32_t类型数据转换为half类型，通过Fixpipe接口完成。再将输出数据从C1搬运至Global Memory，通过DataCopy接口完成。输出数据的搬运过程中不进行NZ到ND的格式转换，最终输出到Global Memory的数据为NZ格式。
    ```
    // CO1 -> C1
    AscendC::FixpipeParamsV220 fixpipeParams;
    fixpipeParams.nSize = N;
    fixpipeParams.mSize = M;
    fixpipeParams.srcStride = M;
    fixpipeParams.dstStride = M * AscendC::BLOCK_CUBE * sizeof(C_T) / AscendC::ONE_BLK_SIZE;
    fixpipeParams.quantPre = QuantMode_t::DEQF16;
    uint64_t deqScalar = static_cast<uint64_t>(*reinterpret_cast<uint32_t*>(&val));
    fixpipeParams.deqScalar = deqScalar;
    AscendC::Fixpipe<C_T, dstCO1_T, AscendC::CFG_NZ>(c1Local, co1Local, fixpipeParams);
    outQueueCO1.FreeTensor(co1Local);
    // C1 -> GM
    event_t eventIDFIXToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::FIX_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::FIX_MTE3>(eventIDFIXToMTE3);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE3>(eventIDFIXToMTE3);
    AscendC::DataCopyParams dataCopyInfo;
    dataCopyInfo.blockCount = 1;
    dataCopyInfo.blockLen = N * M * sizeof(C_T) / AscendC::ONE_BLK_SIZE;
    AscendC::DataCopy(cGM, c1Local, dataCopyInfo);
    inQueueC1.FreeTensor(c1Local);
    ```

  - 调用实现  
    使用内核调用符<<<>>>调用核函数。

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
  mkdir -p build && cd build;      # 创建并进入build目录
  cmake ..;make -j;                # 编译工程
  python3 ../scripts/gen_data.py   # 生成测试输入数据
  ./demo                           # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```