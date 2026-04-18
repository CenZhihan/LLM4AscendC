# Matmul 使用SplitM策略直调样例
## 概述
多核切M场景下的Matmul样例，将输入矩阵沿M轴切分后分配到多核上并行处理。适用于在AscendC分离模式，且AIC:AIV=1:N的架构下，在AIC侧执行一次Iterate计算后，中间结果按M轴切分成多份，分别在N个AIV进行后续处理的场景。

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
│   └── matmul_splitm.asc       // Ascend C样例实现 & 调用样例
```
## 样例描述
- 样例功能： 

  本样例实现是在AscendC分离模式，AIC:AIV=1:2的场景。本样例流程是调用Matmul高阶API，A矩阵和B矩阵同时使能IBShare参数，在AIC上将Iterate的中间结果输出至Unified Buffer，再在两个AIV上分别处理中间结果Unified Buffer的一半数据。

- 样例规格： 

  本样例中：M = 127, N = 127, K = 63。
  <table>
  <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">float16</td><td align="center">ND</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">float16</td><td align="center">ND</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">样例输出</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">matmul_splitm_custom</td></tr>
  </table>

- 样例实现： 
 
  - Kernel关键步骤  
    - 使用Matmul API，关键配置如下：1）A矩阵和B矩阵MatmulType的IBSHARE同时为true，2）使用NORM模板，3）使用SplitM模板策略。  
    - 使用for循环搭配Iterate和GetTensorC接口，将每次Iterate的计算结果在矩阵的M方向一分为二后的数据分别获取到当前AIV核。  
    - 当前AIV核获取数据后进行处理，最后根据GetSubBlockIdx是0/1，将结果搬出到相应的GM。

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
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```