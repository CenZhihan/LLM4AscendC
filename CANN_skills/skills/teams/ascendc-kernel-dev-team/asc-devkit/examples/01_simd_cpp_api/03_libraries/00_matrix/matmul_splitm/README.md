# Matmul算子使用SplitM模板策略直调样例
## 概述
本样例介绍Matmul算子使用SplitM模板策略，A矩阵和B矩阵同时使能IBShare参数，每次Iterate的计算结果在矩阵的M方向一分为二，分别搬运到两个AIV核，再进行后续的数据处理。
## 支持的产品
- Ascend 950PR/Ascend 950DT
## 目录结构介绍
```
├── matmul_splitm
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── matmul_splitm.asc              // Ascend C算子实现 & 调用样例
```
## 算子描述
- 算子功能： 

  本样例中实现的是Matmul算子，Matmul算子的数学表达式为：
  $$
  C = A * B + Bias
  $$
  其中A的形状为[M, K], B的形状为[K, N], C的形状为[M, N], Bias的形状为[1, N]。

- 算子规格： 

  本样例默认执行的算子shape为：M = 127, N = 127, K = 63。
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">M * K</td><td align="center">float16</td><td align="center">ND</td></tr>
  <tr><td align="center">b</td><td align="center">K * N</td><td align="center">float16</td><td align="center">ND</td></tr>
  <tr><td align="center">bias</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">M * N</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">matmul_splitm_custom</td></tr>
  </table>

- 算子实现： 
 
  - Kernel实现  
    关键步骤：  
    1、使用Matmul API，关键配置如下：1）A矩阵和B矩阵MatmulType的IBSHARE同时为true，2）使用NORM模板，3）使用SplitM模板策略。  
    2、使用for循环搭配Iterate和GetTensorC接口，将每次Iterate的计算结果在矩阵的M方向一分为二后的数据分别获取到当前AIV核。  
    3、当前AIV核获取数据后进行处理，最后根据GetSubBlockIdx是0/1，将结果搬出到相应的GM。

  - 调用实现  
    使用内核调用符<<<>>>调用核函数。

## 编译运行
在本样例根目录下执行如下步骤，编译并执行算子。
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
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```