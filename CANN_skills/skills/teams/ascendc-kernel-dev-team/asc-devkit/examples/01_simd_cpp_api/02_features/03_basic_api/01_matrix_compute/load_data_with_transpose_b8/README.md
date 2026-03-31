# load_data_with_transpose_b8样例
## 概述
本样例介绍基础API LoadDataWithTranspose b8数据类型下的使用样例。

## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍
```
├── load_data_with_transpose_b8
│   ├── scripts
│   │   ├── gen_data.py                                    // 输入数据和真值数据生成脚本
│   │   └── verify_result.py                               // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt                                     // 编译工程文件
│   ├── data_utils.h                                       // 数据读入写出函数
│   └── load_data_with_transpose_b8.asc                    // Ascend C算子实现 & 调用样例
```

## 算子描述
- 算子功能：  
  本样例中实现的是[m, n, k]固定为[16, 64, 32]的Matmul算子，并使用Ascend C基础Api LoadDataWithTranspose进行数据搬运。对应的数学表达式为：
  ```
  C = A * B
  ```
- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">16 * 32</td><td align="center">int8_t</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">32 * 64</td><td align="center">int8_t</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">16 * 64</td><td align="center">int32_t</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">load_data_with_transpose_b8</td></tr>
  </table>
- 算子实现：  
  Mmad算子的数学表达式为：
  ```
  C = A * B
  ```
  计算逻辑是：Ascend C提供的矩阵乘计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储并进行分形转换，然后使用计算接口完成两个输入参数矩阵乘运算，得到最终结果，再搬出到外部存储上。  

  Mmad算子的实现流程分为基本任务：CopyIn，SplitA，SplitB，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入inputGM搬运到Local Memory A1/B1中，搬运过程中进行ND至NZ分形转换。SplitA/SplitB分别将数据进一步搬运至接口所要求Local Memory A2/B2，Compute任务负责对数据进行矩阵乘运算，计算结果存储在Local Memory CO1中。CopyOut任务负责将输出数据从CO1搬运至Global Memory上的输出outputGm中，同时完成NZ到ND的分形转换。
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