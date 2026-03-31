# DataCopy多维数据搬运样例

## 概述

本样例介绍如何使用多维数据搬运接口实现GM(Global Memory)到UB(Unified Buffer)通路的数据搬运，并在搬运过程中进行随路Padding。

## 支持的产品
- Ascend 950PR/Ascend 950DT

## 目录结构介绍
```
├── multidimensional_data_movement
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── multidimensional_data_movement.asc      // Ascend C样例实现 & 调用样例
```

## 样例描述

- 样例功能

  本样例基于2D Padding场景，通过多维数据搬运接口将数据从GM(Global Memory)搬运至UB(Unified Buffer)。

- 样例规格

  <table border="2" align="center">
  <tr><td rowspan="1" align="center">类型</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">样例输入</td><td align="center">x</td><td align="center">[2, 8]</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">样例输出</td><td align="center">z</td><td align="center">[4, 16]</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">核函数名</td><td colspan="4" align="center">datacopy_custom</td></tr>
  </table>

- 样例实现：
  - kernel实现   

    输入数据从GM(Global Memory)搬运至UB(Unified Buffer)时，通过配置搬入的维度和对应的Stride，实现2D Padding，将GM(Global Memory)上[2, 8]的数据，搬运至UB(Unified Buffer)并Padding为[4, 16]。

    ```cpp
    // 2D Padding场景，2维数据搬运
    // xGmShape：[2, 8]，搬运8列2行数据，左Padding 3，上Padding 1，右Padding 5，下Padding 1，xLocalShape：[4, 16]
    AscendC::NdDmaLoopInfo<2> loopInfo{{1, 8}, {1, 16}, {8, 2}, {3, 1}, {5, 1}};
    AscendC::NdDmaParams<T, 2> params{loopInfo, 0};  // padding的值为0
    AscendC::NdDmaDci();  // 刷新cache
    static constexpr AscendC::NdDmaConfig dmaConfig;  // 使用默认参数，也可以不传
    AscendC::DataCopy<T, 2, dmaConfig>(xLocal, xGm, params);
    ```

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