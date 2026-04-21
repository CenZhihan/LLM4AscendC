# Matmul 多核切K场景直调样例
## 概述
多核切K场景下的Matmul样例，将输入矩阵沿K轴切分后分配到多核上并行处理。适用多核Matmul，且输入矩阵的M、N较小，无法在M、N方向多核切分的场景。

## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
## 目录结构介绍
```
├── matmul_splitk
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── matmul_splitk.asc       // Ascend C样例实现 & 调用样例
```
## 样例描述
- 样例功能： 

  本样例通过调用Matmul Tiling API时调用EnableMultiCoreSplitK使能多核切K，获取多核切K的tiling计算参数，将单次Iterate计算分配到多个核进行。在Kernel实现中，先将输出Global Memory清零，随后开启AtomicAdd累加，待同一次Iterate计算分配到多个的核的计算完成后，结果累加至输出Global Memory上。

- 约束条件
  - 使能多核切K的场景，获取C矩阵结果时仅支持输出到Global Memory。
  - 使能多核切K的场景，需在Kernel侧代码中首次将C矩阵分片的结果写入Global Memory之前，先清零Global Memory，随后在获取C矩阵分片的结果时，再开启AtomicAdd累加。
    如果不预先清零Global Memory，可能会因为累加Global Memory中的原始无效数据而产生精度问题。
  - 使能多核切K的场景，不支持开启Bias。

- 样例规格： 

  本样例中：M = 16, N = 16, K = 1024。
  <table>
  <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">float16</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">float16</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">样例输出</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">matmul_splitk_custom</td></tr>
  </table>
- 样例实现： 
  - Kernel关键步骤  
    - 对C矩阵的输出Global Memory地址清零。
      ```cpp
      Fill(cGlobal, tiling.M * tiling.N, (cType)0);
      ```
    - 开启AtomicAdd累加，完成矩阵乘操作。
      ```cpp
      uint8_t enAtomic = 1; // set AtomicAdd
      matmulObj.IterateAll(cGlobal, enAtomic);
      ```

  - Tiling关键步骤
    - 设置A、B、C、Bias的参数类型信息；M、N、Ka、Kb形状信息等，调用EnableMultiCoreSplitK使能多核切K。
      ```cpp
      cubeTiling->EnableMultiCoreSplitK(true);
      ```

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