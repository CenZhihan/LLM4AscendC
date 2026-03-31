# SetAtomicType样例
## 概述
本样例基于SetAtomicType为原子操作设定不同的数据类型。
## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍
```
├── set_atomic_type
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── set_atomic_type.asc      // Ascend C算子实现 & 调用样例
```

## 算子描述
- 算子功能：  
  调用SetAtomicType接口后，可设置后续从VECOUT传输到GM的数据是否执行原子比较：将待拷贝的内容和GM已有内容进行比较，将最小值写入GM。可通过设置模板参数来设定不同的数据类型。
- 算子规格：  
  <table>  
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">SetAtomicType</td></tr> 
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center"></td></tr>  
  <tr><td align="center">x</td><td align="center">256</td><td align="center">half</td><td align="center">ND</td><td align="center"></td></tr>  
  <tr><td align="center">y</td><td align="center">256</td><td align="center">half</td><td align="center">ND</td><td align="center"></td></tr>  
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">256</td><td align="center">half</td><td align="center">ND</td><td align="center"></td></tr>  
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">set_atomic_type_custom</td></tr>  
  </table>

- 算子实现：  
  本样例中实现的是固定shape为256的SetAtomicType算子。
  - kernel实现   
    本算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的两个tensor搬运到位于Local Memory的两个LocalTensor中。Compute任务负责对两个输入的tensor求绝对值，并将结果保存在dst0Local和dst1Local中。CopyOut中后续的VECOUT/L0C/L1到GM的数据传输开启原子最小，输出数据从dstLocal搬运至Global Memory上时与Global Memory上已有数据进行比较，将最小值写入输出Tensor，最后再通过SetAtomicType将原子最小操作原来设置的int8类型重新设定为指定的数据类型：T。
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