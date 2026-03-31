# BroadCastVecToMM样例

## 概述
本样例基于BroadCastVecToMM实现数据广播搬运，适用于Unified Buffer与L0C Buffer之间的数据传输。

## 支持的产品
- Atlas 推理系列产品AI Core

## 目录结构介绍
```
├── broad_cast_vec_to_mm
│   ├── scripts
│   │   ├── gen_data.py               // 输入数据和真值数据生成脚本
│   │   └── verify_result.py          // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt                // 编译工程文件
│   ├── data_utils.h                  // 数据读入写出函数
│   └── broad_cast_vec_to_mm.asc      // Ascend C算子实现 & 调用样例
```

## 算子描述
- 算子功能：  

  broad_cast_vec_to_mm算子将位于Unified Buffer的数据进行广播并搬运到L0C Buffer。
- 算子规格：  

  <table>
  <tr><td rowspan="3" align="center">算子输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">16</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">算子输出</td></tr>
  <tr><td align="center">z</td><td align="center">256</td><td align="center">float</td><td align="center">ND</td></tr>
  
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">broad_cast_vec_to_mm_custom</td></tr>
  </table>


- 算子实现：
  - kernel实现   
    BroadCastVecToMMCustom算子的实现流程分为4个基本任务：CopyIn，BroadCast，Aggregate，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor srcGm存储在Unified Buffer上的srcLocal中。BroadCast任务负责对srcLocal执行BroadCastVecToMM指令，将shape为[16]的数据广播到[16, 16]，并搬运到L0C Buffer上的brcLocal中。Aggregate任务负责将brcLocal搬运到Unified Buffer上的dstLocal中。CopyOut任务负责将输出数据从dstLocal搬运至Global Memory上的输出Tensor dstGm。

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