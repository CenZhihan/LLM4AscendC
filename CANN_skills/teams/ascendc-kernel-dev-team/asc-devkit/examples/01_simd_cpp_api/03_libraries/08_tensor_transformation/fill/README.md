# Fill样例

## 概述

本样例在需要预先初始化Global Memory数据的场景下，基于Fill高阶API实现将Global Memory上的数据初始化为指定值，并配合Add接口完成向量加法计算。Fill接口支持在数据搬运前对输出空间进行初始化，常用于workspace地址或输出数据清零场景。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── fill
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── fill.asc                // Ascend C样例实现 & 调用样例
```

## 样例描述
- 样例功能：  
  本样例演示Fill API的使用场景。首先使用Fill接口将输出Global Memory初始化为当前核的blockIdx值（单核场景下为0），然后将两个输入向量从Global Memory搬运到Local Memory，使用Add接口进行向量加法计算，最后将计算结果通过SetAtomicAdd累加到已初始化的输出Global Memory上。更多API详细信息请参考[Fill API文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0891.html)。

- 样例规格：  
  <table border="2" align="center">
  <caption>表1：样例输入输出规格</caption>
  <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="4" align="center">fill</td></tr>

  <tr><td rowspan="4" align="center">样例输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">input_x</td><td align="center">[1,256]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">input_y</td><td align="center">[1,256]</td><td align="center">float</td><td align="center">ND</td></tr>


  <tr><td rowspan="2" align="center">样例输出</td></tr>
  <tr><td align="center">output_z</td><td align="center">[1,256]</td><td align="center">float</td><td align="center">ND</td></tr>


  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">init_global_memory_custom</td></tr>
  </table>

- 样例实现：  
  本样例中实现的是固定shape为输入input_x[1,256]，input_y[1,256]，输出output_z[1,256]的fill样例，所用API详细介绍请参考[Fill API文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0891.html)。

  - Kernel实现

    计算逻辑包含以下步骤：
    1. 使用Fill高阶API将输出Global Memory初始化为当前核的blockIdx值
    2. 设置MTE3_MTE2同步事件，确保Fill操作完成后再进行后续的Unified Buffer操作
    3. 将输入数据从Global Memory搬运到Local Memory（Unified Buffer）
    4. 使用Add接口在Local Memory进行向量加法计算
    5. 使用SetAtomicAdd模式将计算结果累加到输出Global Memory
    
    整个过程展示了Fill API的典型使用场景：在正式计算前预先初始化输出空间。

  - Tiling实现

    本样例中，无需tiling实现。

  - 调用实现  
    使用内核调用符<<<>>>调用核函数。

## 编译运行  

在本样例根目录下执行如下步骤，编译并执行样例。
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
  mkdir -p build && cd build;   # 创建并进入build目录
  cmake ..;make -j;             # 编译工程
  python3 ../scripts/gen_data.py   # 生成测试输入数据
  ./demo                        # 执行编译生成的可执行程序，执行样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```