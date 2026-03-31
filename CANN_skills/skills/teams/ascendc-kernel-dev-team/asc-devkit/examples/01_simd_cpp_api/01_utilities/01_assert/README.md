# Assert算子直调样例

## 概述

本样例通过Ascend C编程语言实现了Matmul算子，使用<<<>>>内核调用符来完成算子核函数在NPU侧运行验证的基础流程，同时在算子中添加assert调测，给出了对应的端到端实现。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── 01_assert
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── assert.asc              // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：    

  Matmul实现了快速的Matmul矩阵乘法的运算操作。  

  assert可以实现assert断言功能。 
  
  本样例通过样例正常执行未中断报错，从而判断样例是否执行成功。  

  Matmul的计算公式为：

  ```
  C = A * B
  ```

  - A、B为源操作数，A为左矩阵，形状为\[M, K]；B为右矩阵，形状为\[K, N]。
  - C为目的操作数，存放矩阵乘结果的矩阵，形状为\[M, N]。

  在上述算子中添加assert，可以添加断言功能，当断言条件不满足时，会抛出异常。

- 算子规格：

  在核函数直调样例中，算子实现支持的shape为：M = 512, N = 1024, K = 512。
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">M * K</td><td align="center">float16</td><td align="center">ND</td></tr>
  <tr><td align="center">b</td><td align="center">K * N</td><td align="center">float16</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">M * N</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">matmul_custom</td></tr>
  </table>

- 算子实现：  

  本调用样例中实现的是[M, K, N]固定为[512, 512, 1024]的Matmul算子，并调用assert进行调测。
  - Kernel实现  
    Matmul算子的数学表达式为：
    $$
    C = A * B
    $$
    其中A的形状为[512, 512]，B的形状为[512, 1024]，C的形状为[512, 1024]。

  - 调用实现  
    使用内核调用符<<<>>>调用核函数。

## 编译运行  

- 配置环境变量  
在本样例根目录下执行如下步骤，编译并执行算子。  
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
  test pass!
  ```