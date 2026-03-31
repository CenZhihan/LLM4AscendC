# MDL模板下A矩阵为TSCM输入的Matmul算子样例
## 概述
本样例介绍了A矩阵为自定义TSCM输入的Matmul算子，将多次计算的数据一次性搬到TSCM上，通过使能MDL模板，使用SetOrgShape接口，实现TSCM内存自主管理。
## 支持的产品
- Ascend 950PR/Ascend 950DT
## 目录结构介绍
```
├── matmul_tscm_mdl_setorgshape
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── matmul_tscm_mdl_setorgshape.asc              // Ascend C算子实现 & 调用样例
```
## 算子描述
- 算子功能： 

  本样例中实现的是Matmul算子，Matmul算子的数学表达式为：
  $$
  C = A * B + Bias
  $$
  其中A的形状为[M, K], B的形状为[K, N], C的形状为[M, N], Bias的形状为[1, N]。

- 算子规格： 

  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">(127, 63)</td><td align="center">int8</td><td align="center">ND</td></tr>
  <tr><td align="center">b</td><td align="center">(63, 127)</td><td align="center">int8</td><td align="center">ND</td></tr>
  <tr><td align="center">bias</td><td align="center">(127, )</td><td align="center">int32</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">(127, 127)</td><td align="center">int32</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">matmul_tscm_mdl_setorgshape_custom</td></tr>
  </table>
- 算子实现： 
 
  - Kernel实现  
    关键步骤：  
    1、使用MDL模板构造Matmul对象，指定A矩阵的TPosition为TSCM。   
    ```
    AscendC::Matmul<
      AscendC::MatmulType<AscendC::TPosition::TSCM, CubeFormat::ND, AType>,
      AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>,
      AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
      AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>, CFG_MDL> matmulObj;
    ```
    2、使用SetOrgShape接口，指定A在TSCM上、B在GM上的shape。   
    ```
    matmulObj.SetOrgShape(alignedSingleM, tiling.N, alignedSingleK, tiling.Kb, tiling.N);
    ```
    3、样例中每次计算一个[baseM, baseK] * [baseK, baseN]，为方便后期进行精度对比，结合for循环，将所有计算结果在L0C上累加后搬出到GM上。   
    ```
    for (int m = 0; m < mIter_; m++) {
        for (int n = 0; n < nIter_; n++) {
            for (int k = 0; k < kIter_; k++) {
              matmulObj.SetSingleShape(curBaseM, curBaseN,, curBaseK);
              ...
              matmulObj.Iterate(k != 0);
            }
            matmulObj.GetTensorC(cGlobal[m * tiling.baseM * tiling.N + n * tiling.baseN]);
        }
    }
    ```

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