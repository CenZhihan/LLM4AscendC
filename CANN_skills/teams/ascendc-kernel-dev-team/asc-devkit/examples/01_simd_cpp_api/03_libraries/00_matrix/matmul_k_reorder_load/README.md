# Matmul 使能K轴错峰加载数据直调样例
## 概述
使能K轴错峰加载数据特性的Matmul样例，减少多核对Global Memory地址访问冲突概率。

执行Matmul计算时，如果多核的左矩阵或者右矩阵相同，且存储于Global Memory，多个核一般会同时访问相同地址以加载矩阵数据，引发同地址访问冲突，影响性能。使能该参数后，多核执行Matmul时，将尽量在相同时间访问数据的不同Global Memory地址，减少地址访问冲突概率，提升性能。

建议K轴较大（K轴建议不小于4096）且左矩阵和右矩阵均非全载场景使能参数。

## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
## 目录结构介绍
```
├── matmul_k_reorder_load
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── matmul_k_reorder_load.asc              // Ascend C样例实现 & 调用样例
```
## 样例描述
- 样例功能： 

  本样例调用Matmul高阶API时，通过设置MatmulConfig参数enableKdimReorderLoad使能K轴错峰加载数据，该参数功能只支持MDL模板。  

- 样例规格： 

    本样例中：M = 768, N = 2048, K = 6144。
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
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">matmul_k_reorder_load_custom</td></tr>
  </table>
- 样例实现： 
 
  - Kernel关键步骤  
    - 创建Matmul对象：调用GetMDLConfig接口将enableKdimReorderLoad参数设置为true，获取自定义模板CFG_MDL_REORDER，通过传入模板参数创建Matmul对象。
      ```cpp
      // enableKdimReorderLoad set to true
      static constexpr auto CFG_MDL_REORDER = GetMDLConfig(false, false, 0, false, false, false,
          false, true, true, false, false, true);
      AscendC::Matmul<
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>,
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>,
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>,
          CFG_MDL_REORDER
          > matmulObj;
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