# Matmul 自定义TSCM输入直调样例
## 概述
使用数据来源为GM的用户自定义TSCM的输入的Matmul样例，开发者可以自主管理L1 Buffer以高效利用硬件资源。TSCM即Temp Swap Cache Memory，用于临时把数据交换到额外空间。该场景由开发者管理L1 Buffer，再将输入矩阵数据对应的L1 Buffer地址作为Matmul的输入。

## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
## 目录结构介绍
```
├── matmul_tscm
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── matmul_tscm.asc         // Ascend C样例实现 & 调用样例
```
## 样例描述
- 样例功能： 

  Matmul样例自定义A矩阵从Global Memory到L1 Buffer的数据搬入，使A矩阵全部数据常驻在L1 Buffer中，调用Matmul API计算时，A矩阵为TSCM输入，B矩阵设为GM输入，对输入的A、B矩阵做矩阵乘和加Bias偏置。

- 约束条件
  - TSCM输入的矩阵必须能在L1 Buffer上全载。

- 样例规格： 

  本样例中：M = 64, N = 64, K = 64。
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
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">matmul_tscm_custom</td></tr>
  </table>

- 样例实现： 
  - Kernel关键步骤
    - 创建Matmul对象。其中左矩阵A的MatmulType中，POSITION为TSCM，SRC_POSITION为默认值GM。
      ```cpp
      AscendC::MatmulType<AscendC::TPosition::TSCM, CubeFormat::NZ, AType>
      ```
    - 自定义左矩阵A从GM到L1的搬运，设置左矩阵A、右矩阵B、Bias，其中左矩阵A为TSCM输入。
      ```cpp
      AscendC::TSCM<AscendC::TPosition::GM, 1> scm;
      pipe->InitBuffer(scm, 1, tiling.M * tiling.Ka * sizeof(AType));
      auto scmTensor = scm.AllocTensor<AType>();
      DataCopy(scmTensor, aGlobal, tiling.M * tiling.Ka);
      scm.EnQue(scmTensor);
      AscendC::LocalTensor<AType> scmLocal = scm.DeQue<AType>();

      matmulObj.SetTensorA(scmLocal, IS_TRANS_A);
      matmulObj.SetTensorB(bGlobal, IS_TRANS_B);
      if (tiling.isBias) {
          matmulObj.SetBias(biasGlobal);
      }
      matmulObj.IterateAll(cGlobal);
      matmulObj.End();

      scm.FreeTensor(scmLocal);
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
  mkdir -p build && cd build;    # 创建并进入build目录
  cmake ..;make -j;    # 编译工程
  python3 ../scripts/gen_data.py    # 生成测试输入数据
  ./demo                        # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin    # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```