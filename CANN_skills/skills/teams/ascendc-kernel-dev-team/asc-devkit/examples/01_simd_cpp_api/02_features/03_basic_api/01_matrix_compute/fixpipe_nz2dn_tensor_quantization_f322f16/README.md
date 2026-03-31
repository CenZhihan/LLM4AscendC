# Matmul算子使用SplitM模板策略直调样例
## 概述
本样例介绍基础API的fixpipe接口的新特性NZ2DN，该特性优化了数据处理流程：在Mmad计算结束后，通过fixpipe接口将 L0C 中的结果直接搬移到 GM。在这一搬移过程中同时完成 ReLU 激活、F322F16 量化和NZ2DN分形转换，实现了高效的一站式数据处理。
## 支持的产品
- Ascend 950PR/Ascend 950DT
## 目录结构介绍
```
├── fixpipe_nz2dn_tensor_quantization_f322f16
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── fixpipe_nz2dn_tensor_quantization_f322f16.asc              // Ascend C算子实现 & 调用样例
```
## 算子描述
- 算子功能： 

  本样例中实现对矩阵进行Mmad操作后，通过基础Api的fixpipe接口将计算结果从C1搬出到GM，并进行ReLU激活、F322F16量化并附带NZ2DN分形转换操作。
  其中输入的矩阵A和B的形状为分别为[4, 128], [128, 128], 通过fixpipe接口搬运后输出矩阵C的形状为[4, 128]。
- 算子规格： 
  在核函数直调样例中，算子实现支持的shape为：M = 4, N = 128, K = 128。
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">FixpipeCustom</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">M * K </td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">B</td><td align="center">K * N</td><td align="center">half</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">C</td><td align="center">M * N</td><td align="center">half</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">fixpipe_nz2dn_tensor_quantization_f322f16_custom</td></tr>
  </table>
- 算子实现： 
 
  - Kernel实现  
    关键步骤：  
    对 L0C 中的计算结果通过Fixpipe接口从L0C搬运至 GM 中，并执行ReLU 激活、F32 转 F16 量化操作以及NZ2DN分形转换。

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