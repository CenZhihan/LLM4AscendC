# Batch Matmul的IterateNBatch场景直调样例
## 概述
多次批量处理Matmul计算的样例，包括非异步场景和异步场景的实现。

## 支持的产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── batch_matmul_iterate_n_batch
│   └── scripts
│       ├── gen_data.py              // 输入数据和真值数据生成脚本文件
│       └── verify_result.py         // 真值对比文件
│   ├── CMakeLists.txt               // 编译工程文件
│   ├── data_utils.h                 // 数据读入写出函数
│   └── batch_matmul_iterate_n_batch.asc   // Ascend C样例实现 & 调用样例
```
## 样例描述
- 样例功能：  
  本样例对BatchMatmul计算批次进行拆分，拆分为nNum个主批次，每个主批次包括BatchNum个子批次的Matmul计算。通过修改batch_matmul_iterate_n_batch.asc代码中的IS_SYNCH参数，可以控制是否使用异步计算流程。

- 约束条件
  - 单Batch Matmul内计算遵循IterateBatch的约束条件；
  - 对于BSNGD、SBNGD、BNGS1S2 Layout格式，输入A、B矩阵多Batch数据总和应该小于L1 Buffer的大小；
- 样例规格：  

    本样例中：BatchNum = 3, M = 32, N = 256, K = 64。
    <table>
    <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="6" align="center">BatchMatmulCustom</td></tr>
    </tr>
    <tr><td rowspan="4" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td><td align="center">layout</td></tr>
    <tr><td align="center">a</td><td align="center">[BatchNum, M, K]</td><td align="center">float16</td><td align="center">ND</td><td align="center">true</td><td align="center">BSNGD</td></tr>
    <tr><td align="center">b</td><td align="center">[BatchNum, K, N]</td><td align="center">float16</td><td align="center">ND</td><td align="center">false</td><td align="center">BSNGD</td></tr>
    <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td><td align="center">-</td></tr>
    </tr>
    </tr>
    <tr><td rowspan="1" align="center">样例输出</td><td align="center">c</td><td align="center">[BatchNum, M, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td><td align="center">BSNGD</td></tr>
    </tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="6" align="center">batch_matmul_iterate_n_batch_custom</td></tr>
    </table>

- 样例实现： 
  - Kernel关键步骤
    - 创建Matmul对象时，自定义MatmulConfig参数，将其中的isNBatch参数设置为true，使能多Batch输入和多Batch输出；将其中的isBiasBatch参数设置为false，使能BatchMatmul的Bias复用功能，获得自定义的使用NORM模板的Matmul对象。
      ```cpp
      constexpr MatmulConfigMode configMode = MatmulConfigMode::CONFIG_NORM;
      constexpr MatmulBatchParams batchParams = {
        true, BatchMode::BATCH_LESS_THAN_L1, false /* isNBatch, batchMode, isBiasBatch */
      };
      constexpr MatmulConfig CFG_MM = GetMMConfig<configMode>(batchParams);
      AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_PARTIAL> matmulObj;
      ```
    - 计算nNum大小，根据A、B矩阵的Layout信息和BatchNum计算nNum大小。
      ```cpp
      int32_t g_lay = tiling.ALayoutInfoG > tiling.BLayoutInfoG ? tiling.ALayoutInfoG : tiling.BLayoutInfoG;
      int32_t nNum = tiling.ALayoutInfoB * tiling.ALayoutInfoN * g_lay / tiling.BatchNum;
      ```
    - 设置workspace。
      ```cpp
        matmulObj.SetWorkspace(cGlobal);
      ```
    - 完成多batch矩阵乘操作，并调用GetBatchTensorC接口获取结果。
      ```cpp
      matmulObj.template IterateNBatch<false>(nNum, batchA, batchB, false);
      for(int32_t j = 0; j < nNum; ++j){
          matmulObj.template GetBatchTensorC<false>(batchA, batchB, false);
      }
      ```

  - Tiling关键步骤
    - 调用SetALayout、SetBLayout、SetCLayout、SetBatchNum设置A/B/C的Layout轴信息和最大BatchNum数。
      ```cpp
      cubeTiling.SetALayout(1, M, 1, batchNum, K);
      cubeTiling.SetBLayout(1, N, 1, batchNum, K);
      cubeTiling.SetCLayout(1, M, 1, batchNum, N);
      cubeTiling.SetBatchNum(batchNum);
      ```

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