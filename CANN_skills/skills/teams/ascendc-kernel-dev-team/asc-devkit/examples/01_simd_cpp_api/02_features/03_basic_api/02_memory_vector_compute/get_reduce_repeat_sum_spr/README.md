# GetReduceRepeatSumSpr样例

## 概述

本样例在归约场景下，展示了ReduceSum和GetReduceRepeatSumSpr的配合使用，用于在连续场景下获取求和结果。ReduceSum接口执行实际的求和计算，GetReduceRepeatSumSpr接口从硬件寄存器中读取ReduceSum的计算结果。这种方式适用于需要高效获取归约计算结果的场景，如统计聚合、求和计算等，能够避免从输出Tensor中读取数据，直接从硬件寄存器获取精确的求和结果。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── get_reduce_repeat_sum_spr
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── get_reduce_repeat_sum_spr.asc         // Ascend C算子实现 & 调用样例
```

## 样例描述

- 样例功能：
   GetReduceRepeatSumSpr样例返回ReduceSum接口的计算结果。

- 样例规格：
   <table border="2" align="center">
   <caption>表1：GetReduceRepeatSumSpr样例规格</caption>
   <tr><td rowspan="2" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
   <tr><td align="center">x</td><td align="center">[256]</td><td align="center">float</td><td align="center">ND</td></tr>
   <tr><td rowspan="1" align="center">样例输出</td><td align="center">z</td><td align="center">[256]</td><td align="center">float</td><td align="center">ND</td></tr>
   <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">kernel_get_reduce_repeat_sum_spr</td></tr>
   </table>

- Kernel实现
   GetReduceRepeatSumSpr样例调用一次ReduceSum接口，然后调用GetReduceRepeatSumSpr获取计算结果，并将结果搬运至Global Memory上的输出Tensor dstGm。

- 调用实现
   使用内核调用符<<<>>>调用核函数。

## 编译运行

在本样例根目录下执行如下步骤，编译并执行算子。

- 配置环境变量  
  请根据当前环境上CANN开发套件包的[安装方式](../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
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
  test pass
  ```
