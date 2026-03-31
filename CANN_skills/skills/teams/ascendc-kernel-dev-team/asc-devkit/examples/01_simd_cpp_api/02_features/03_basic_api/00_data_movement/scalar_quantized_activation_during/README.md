# DataCopy Scalar随路量化激活搬运样例

## 概述

本样例在卷积场景下，基于DataCopy实现数据随路量化激活搬运，支持在数据搬运过程中通过Scalar量化将int32_t类型转换为half类型，并完成CO1(L0C Buffer)到GM(Global Memory)通路NZ到ND格式转换，同时支持Relu激活。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```plain
├── scalar_quantized_activation_during
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── scalar_quantized_activation_during.asc      // Ascend C样例实现 & 调用样例
```

## 样例描述
- 样例功能：  
  支持在数据搬运过程中进行Scalar量化和Relu激活操作，同时支持CO1(L0C Buffer)到GM(Global Memory)通路NZ到ND格式的转换。

- 样例规格：  
  <table>
  <tr><td rowspan="4" align="center">样例输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 1, 4, 4, 32]</td><td align="center">int8_t</td><td align="center">NC1HWC0</td></tr>
  <tr><td align="center">y</td><td align="center">[1, 2, 2, 128, 32]</td><td align="center">int8_t</td><td align="center">FRACTAL_Z</td></tr>
  <tr><td rowspan="2" align="center">样例输出</td></tr>
  <tr><td align="center">z</td><td align="center">[1, 8, 3, 3, 16]</td><td align="center">half</td><td align="center">NC1HWC0</td></tr>

  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">KernelFixpipeCustom</td></tr>
  </table>

- 样例实现：  
  - Kernel实现   
    - 调用Mmad接口，将卷积场景下的int8_t类型的输入Tensor[1, 1, 4, 4, 32]与卷积核Tensor[1, 2, 2, 128, 32]做矩阵乘法，得到int32_t类型的结果矩阵[1, 8, 3, 3, 16]。
    - 配置DataCopyCO12DstParams参数用于DataCopy随路量化激活搬运，设置量化模式DEQF16，并使用SetFixpipePreQuant接口设置scalar量化参数（本样例为0.5），将Mmad计算出的CO1(L0C Buffer)上的结果由int32_t量化成half，完成NZ到ND的格式转换，并完成Relu激活后搬出到GM(Global Memory)。
    - Scalar量化支持的量化模式如下：

      | 量化模式 | 描述 |
      | :--- | :--- |
      | F322F16 | float量化成half |
      | F322BF16 | float量化成bfloat16_t |
      | DEQF16 | int32_t量化成half |
      | QF322B8_PRE | float量化成int8_t/uint8_t |
      | REQ8 | int32_t量化成int8_t/uint8_t |

    - 接口详细描述参考Ascend C API DataCopy随路量化激活搬运。

  - 调用实现  
    使用内核调用符<<<>>>调用核函数。

## 编译运行 
 
在本样例根目录下执行如下步骤，编译并执行样例。
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
  python3 ../scripts/verify_result.py output/output.bin output/golden_data.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```

  执行结果如下，说明精度对比成功。

  ```bash
  test pass!
  ```
