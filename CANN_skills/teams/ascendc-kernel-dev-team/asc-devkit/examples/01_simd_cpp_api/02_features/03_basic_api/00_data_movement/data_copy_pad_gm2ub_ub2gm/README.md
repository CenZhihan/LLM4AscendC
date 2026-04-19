# DataCopyPad样例

## 概述

本样例在数据搬运场景下，基于DataCopyPad API实现非32字节对齐数据的搬运及填充功能，配合SetPadValue API实现自定义数据填充。DataCopyPad API支持从Global Memory到Local Memory的非对齐数据搬运，并可在数据左侧或右侧填充指定数值，解决硬件要求的32字节对齐约束。

数据搬运过程包括：Global Memory（GM）→VECIN队列（使用DataCopyPad进行非对齐搬运并填充）、VECIN→VECOUT（计算处理）、VECOUT队列→Global Memory（GM）。本样例演示了如何将20个half元素（40字节，非32字节对齐）从GM搬运到VECIN，并在右侧填充12个half元素（值为1），形成32个half元素（64字节，满足32字节对齐要求）的数据块。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── data_copy_pad_gm2ub_ub2gm
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── data_copy_pad.asc       // Ascend C样例实现 & 调用样例
```

## 样例描述

- 样例功能：  

  样例展示了使用DataCopyPad将非32字节对齐的数据从Global Memory搬入VECIN并进行数据填充的使用方法。非32字节对齐指的是数据字节数不满足32字节对齐要求（例如40字节的20个half元素），硬件要求LocalTensor必须32字节对齐，因此需要通过填充操作补齐到64字节（32个half元素）。填充方法为使用DataCopyPadExtParams结构体配置填充参数，配合SetPadValue API设置填充值。更多API详细信息请参考[DataCopyPad API文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0265.html)和[SetPadValue API文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0266.html)。
- 样例规格：  

  <table>
  <tr><td rowspan="3" align="center">样例输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[1,20]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">样例输出</td></tr>
  <tr><td align="center">z</td><td align="center">[1,32]</td><td align="center">half</td><td align="center">ND</td></tr>
  
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">data_copy_pad_custom</td></tr>
  </table>


- 样例实现：
  - kernel实现   
    计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用DataCopyPad基础API接口将输入srcLocal的数值搬运到dstLocal中，再搬出到外部存储上。

    DataCopyPadCustom样例的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。  
    - CopyIn任务  
      调用DataCopyPad负责将Global Memory上的输入Tensor srcGlobal连续非对齐搬入srcLocal中。  
      本样例中使用DataCopyExtParams结构体自定义搬入时数据块个数为1，数据量为20个half（40字节），源操作数和目的操作数的数据块间隔为0，右侧填充12个half元素（值为1），形成32个half元素（64字节）的完整数据块。参数说明：
      - blockCount=1：数据块个数为1
      - blockLen=20*sizeof(half)：每个数据块长度为40字节（20个half元素）
      - isPad=false：填充自定义值，配合SetPadValue使用
      - leftPadding=0，rightPadding=12：左侧不填充，右侧填充12个元素
  
      ```
      AscendC::DataCopyExtParams copyParams{1, 20 * sizeof(half), 0, 0, 0}; 
      AscendC::DataCopyPadExtParams<half> padParams;
      padParams.isPad = false;
      padParams.leftPadding = 0;
      padParams.rightPadding = 12;

      // SetPadValue设置填充值为1，isPad=true时生效，isPad=false时填充随机值
      AscendC::SetPadValue((half)1);
      
      // 非32字节对齐数据拷贝：从全局内存到本地缓冲区，并填充12个half元素
      AscendC::DataCopyPad(srcLocal, srcGlobal, copyParams, padParams); 
      ```
    - Compute任务
      负责使用Adds对srcLocal和scalar求和，并将结果存储到dstLocal中。  
    - CopyOut任务
      负责将输出数据从dstLocal连续非对齐搬运至Global Memory上的输出Tensor dstGlobal。

  - 调用实现  
    使用内核调用符<<<>>>调用核函数。

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
  mkdir -p build && cd build;   # 创建并进入build目录
  cmake ..;make -j;             # 编译工程
  python3 ../scripts/gen_data.py   # 生成测试输入数据
  ./demo                        # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```