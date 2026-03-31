# GetReduceRepeatMaxMinSpr样例

## 概述

本样例在归约场景下，展示了WholeReduceMax/WholeReduceMin和GetReduceRepeatMaxMinSpr的配合使用，用于在连续场景下获取最大/最小值及其索引位置。WholeReduceMax/WholeReduceMin接口执行实际的归约计算，GetReduceRepeatMaxMinSpr接口从硬件寄存器中读取计算结果和对应的索引位置。这种配合使用方式适用于需要同时获取极值和其位置信息的场景，如ArgMax/ArgMin操作、特征定位等，能够完成极值计算和索引定位。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── get_reduce_repeat_max_min_spr
│   ├── CMakeLists.txt                    // 编译工程文件
│   └── get_reduce_repeat_max_min_spr.asc      // Ascend C样例实现 & 调用样例
```

## 样例描述

- 样例功能：
   获取ReduceMax、ReduceMin连续场景下的最大/最小值以及相应的索引值。

- 样例规格：
   <table border="2" align="center">
   <caption>表1：GetReduceRepeatMaxMinSpr样例规格</caption>
   <tr><td rowspan="2" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
   <tr><td align="center">x</td><td align="center">[256]</td><td align="center">float</td><td align="center">ND</td></tr>
   <tr><td rowspan="1" align="center">样例输出</td><td align="center">z</td><td align="center">[256]</td><td align="center">float</td><td align="center">ND</td></tr>
   <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">kernel_get_reduce_repeat_max_min_spr</td></tr>
   </table>

- Kernel实现

   GetReduceRepeatMaxMinSpr样例主要使用WholeReduceMax/WholeReduceMin获取准确的索引值，然后调用GetReduceRepeatMaxMinSpr获取最大/最小值，以及其对应的索引值，并将结果搬运至Global Memory上的输出Tensor dstGm。

- 调用实现

   使用内核调用符<<<>>>调用核函数。

## 编译运行

在本样例根目录下执行如下步骤，编译并执行样例。

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
  ./demo                        # 执行编译生成的可执行程序，执行样例
  ```

  执行结果如下，说明精度对比成功。

  ```bash
  test pass!
  ```
