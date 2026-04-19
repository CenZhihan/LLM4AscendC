# DataCopy量化激活搬运样例

## 概述

本样例在卷积场景下，基于DataCopy实现数据随路量化激活搬运，支持在数据搬运过程中通过量化将int32_t类型转换为half类型，并完成CO1（L0C Buffer）到GM（Global Memory）通路NZ到ND格式转换，同时支持Relu激活。本样例支持两种量化模式：Scalar量化和Tensor量化，并给出了2种不同的测试场景（scenario）。
    <table>
 	  	 	<tr>
 	  	 		<td>scenarioNum</td>
 	  	 		<td>量化模式</td>
 	  	 	</tr>
 	  	 	<tr>
 	  	 		<td>1</td>
 	  	 		<td>Scalar量化模式</td>
 	  	 	</tr>
 	  	 	<tr>
 	  	 		<td>2</td>
 	  	 		<td>Tensor量化模式</td>
 	  	 	</tr>
 	  	 </table>

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── data_copy_l0c2gm
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── data_copy_quantized_activation_during.asc      // Ascend C样例实现 & 调用样例
```

## 样例描述

- 样例功能

  支持在数据搬运过程中进行量化和Relu激活操作，同时支持CO1（L0C Buffer）到GM（Global Memory）通路NZ到ND格式的转换。本样例支持两种量化模式：
  - 场景1（Scalar量化）：使用SetFixpipePreQuant接口设置Scalar量化参数，量化模式为DEQF16，量化参数为0.5。
  - 场景2（Tensor量化）：使用SetFixPipeConfig接口设置Tensor量化参数，量化模式为VDEQF16，量化参数为1。
  - 接口资料参考[随路量化激活搬运](../../../../../../docs/api/context/随路量化激活搬运.md)。

- 样例规格

  <table>
  <tr><td rowspan="4" align="center">样例输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 1, 4, 4, 32]</td><td align="center">int8_t</td><td align="center">NC1HWC0</td></tr>
  <tr><td align="center">y</td><td align="center">[1, 2, 2, 128, 32]</td><td align="center">int8_t</td><td align="center">FRACTAL_Z</td></tr>
  <tr><td rowspan="2" align="center">样例输出</td></tr>
  <tr><td align="center">z</td><td align="center">[1, 8, 3, 3, 16]</td><td align="center">half</td><td align="center">NC1HWC0</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">KernelDataCopyL0c2Gm</td></tr>
  </table>

- 样例实现

  - Kernel实现

    - 调用DataCopy基础API，将数据从GM（Global Memory）搬运到A1（L1 Buffer）和B1（L1 Buffer）。

    - 调用LoadData接口，将数据从A1（L1 Buffer）和B1（L1 Buffer）搬运到A2（L0A Buffer）和B2(L0B Buffer)。

    - 调用Mmad接口，将卷积场景下的int8_t类型的输入Tensor[1, 1, 4, 4, 32]与卷积核Tensor[1, 2, 2, 128, 32]做矩阵乘法，得到int32_t类型的结果矩阵[1, 8, 3, 3, 16]。

    - 配置DataCopyCO12DstParams参数用于DataCopy随路量化激活搬运，将Mmad计算出的CO1（L0C Buffer）上的结果由int32_t量化成half，完成NZ到ND的格式转换，并完成Relu激活后搬出到GM（Global Memory）。

  - 量化模式说明
      | 量化模式 | 描述 |
      | :--- | :--- |
      | DEQF16 | int32_t量化成half（Scalar量化） |
      | VDEQF16 | int32_t量化成half（Tensor量化） |

- 调用实现

  使用内核调用符`<<<>>>`调用核函数。

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
  SCENARIO=1
  mkdir -p build && cd build;   # 创建并进入build目录
  cmake .. -DSCENARIO_NUM=$SCENARIO;make -j;    # 编译工程
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO   # 生成测试输入数据
  ./demo                        # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py output/output.bin output/golden_data.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功：
  ```bash
  test pass!
  ```
