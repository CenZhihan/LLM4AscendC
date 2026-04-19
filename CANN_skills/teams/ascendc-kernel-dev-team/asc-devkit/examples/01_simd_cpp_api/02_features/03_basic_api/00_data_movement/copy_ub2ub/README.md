# Copy样例

## 概述

本样例在数据搬运场景下，基于Copy API实现VECIN和VECOUT存储单元之间的数据搬运，支持mask操作和DataBlock间隔操作。Copy API是Ascend C提供的矢量计算基础接口，用于在LocalTensor之间进行数据拷贝，支持灵活的元素选择和间隔控制。

数据搬运过程包括：Global Memory（GM）→VECIN队列、VECIN→VECOUT（使用Copy API）、VECOUT队列→Global Memory（GM）。其中Copy API通过mask参数控制参与计算的元素数量，通过DataBlock参数控制数据块的地址步长，实现对数据搬运过程的精细化管理。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── copy_ub2ub
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── copy.asc                // Ascend C样例实现 & 调用样例
```

## 样例描述

- 样例功能：  

  本样例是展示使用Copy进行VECIN，VECOUT之间的数据搬运，支持mask操作和DataBlock间隔操作。Copy的具体参数介绍可以参考[Copy API文档](../../../../../../docs/api/context/Copy.md)。
- 样例规格：  

  <table border="2" align="center">
  <caption>表1：样例输入输出规格</caption>
  <tr><td rowspan="3" align="center">样例输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[1,512]</td><td align="center">int32_t</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">样例输出</td></tr>
  <tr><td align="center">z</td><td align="center">[1,512]</td><td align="center">int32_t</td><td align="center">ND</td></tr>
  
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">copy_custom</td></tr>
  </table>


- 样例实现：
  - kernel实现   
    计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用Copy基础API接口将输入srcLocal的数值搬运到dstLocal中，再搬出到外部存储上。Copy API的参数说明：
    - mask：控制每次迭代内参与计算的元素数量，本样例中mask=64表示前64个元素参与计算
    - repeat：重复次数，本样例中repeat=8表示执行8次拷贝操作
    - DataBlock参数：{1, 1, 8, 8}表示同一迭代内datablock的地址步长为1，相邻迭代间的地址步长为8

  - 调用实现  
    使用内核调用符<<<>>>调用核函数。

## 编译运行  

在本样例根目录下执行如下步骤，编译并执行。
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