# SyncAll样例
## 概述
本样例基于SyncAll实现核间同步，适用于以下场景：不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题，通过调用本接口来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。
## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍
```
├── sync_all
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── sync_all.asc      // Ascend C算子实现 & 调用样例
```

## 算子描述
- 算子功能：  
  功能为使用8个核进行数据处理，每个核均是处理32个float类型数据，对该数据乘2后再与其他核上进行同样乘2的数据进行相加，中间结果保存到workGm，调用SyncAll实现8个核间的同步。

- 算子规格：  
  <table>
 <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">SyncAll</td></tr>   
  <tr><td rowspan="3" align="center">算子输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">256</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">算子输出</td></tr>
  <tr><td align="center">z</td><td align="center">256</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">sync_all_custom</td></tr>
  </table>


- 算子实现：  
  - kernel实现   
    
    本算子实现了8个核的核间同步，Process中每个核都执行以下三个基本任务：CopyIn、FirstCompute、CopyToWorkGlobal。CopyIn任务负责将Global Memory上的输入Tensor srcGm存储在srcLocal中。Compute任务负责对srcLocal与标量2相乘，并将结果存储到dstLocal中。CopyToWorkGlobal任务负责将输出数据从dstLocal搬运至Global Memory上的输出Tensor WorkGlobal。接着Process调用SyncAll等待所有核都完成计算，最终累加结果需要等所有核都计算完成。

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