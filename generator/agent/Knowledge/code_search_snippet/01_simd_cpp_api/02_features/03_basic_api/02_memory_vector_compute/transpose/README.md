# Transpose样例

## 概述

本样例介绍矩阵转置Transpose接口在普通转置和增强转置两种场景下的使用方法，适用以下两种转置功能：（1）对16*16的二维矩阵数据块进行转置，（2）[N,C,H,W]与[N,H,W,C]四维矩阵互相转换。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── transpose
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── transpose.asc           // Ascend C样例实现 & 调用样例
```

## 样例描述

- 样例功能：  
  Transpose样例实现矩阵转置功能，支持普通转置和增强转置两种场景:

  1、普通转置，支持16*16的二维矩阵数据块进行转置

  2、增强转置，通过transposeParams指定转置类型，支持16*16的二维矩阵数据块转置，支持[N,C,H,W]与[N,H,W,C]四维矩阵互相转换

  并且样例可通过编译参数 `SCENARIO_NUM` 来切换不同的场景，参数详见表2。


- 场景说明：
  <table border="2" align="center">
  <caption>表1：场景与样例规格对照表</caption>
  <tr>
    <th align="center">场景名称</th>
    <th align="center">输入shape</th>
    <th align="center">输出shape</th>
    <th align="center">数据类型</th>
    <th align="center">输入格式</th>
    <th align="center">输出格式</th>
    <th align="center">转置类型</th>
  </tr>
  <tr>
    <td align="center">普通转置</td>
    <td align="center">[16,16]</td>
    <td align="center">[16,16]</td>
    <td align="center">half</td>
    <td align="center">ND</td>
    <td align="center">ND</td>
    <td align="center">/</td>
  </tr>
  <tr>
    <td align="center">增强转置</td>
    <td align="center">[3,3,2,8]</td>
    <td align="center">[3,2,8,3]</td>
    <td align="center">half</td>
    <td align="center">NCHW</td>
    <td align="center">NHWC</td>
    <td align="center">TRANSPOSE_NCHW2NHWC</td>
  </tr>
  </table>

- 数据格式说明：

  卷积神经网络的特征图（Feature Map）通常用四维数组保存，即4D，4D格式解释如下：
    - N：Batch数量。
    - H：Height，特征图高度。
    - W：Width，特征图宽度。
    - C：Channels，特征图通道。
    
  由于数据只能线性存储，因此这四个维度有对应的顺序，不同深度学习框架会按照不同的顺序存储特征图数据，比如在TensorFlow中，排列顺序为[Batch, Height, Width, Channels]，即NHWC。

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
  SCENARIO_NUM=1      # 设置场景编号
  mkdir -p build && cd build;      # 创建并进入build目录
  cmake .. -DRUN_MODE=npu -DNPU_ARCH=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j;    # 编译工程
  python3 ../scripts/gen_data.py -scenario_num=$SCENARIO_NUM   # 生成测试输入数据
  ./demo                           # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py ./output/output.bin ./output/golden.bin  # 验证输出结果是否正确
  ```
  使用CPU调试或NPU仿真模式时，添加 `-DRUN_MODE=cpu` 或 `-DRUN_MODE=sim` 参数即可。
  示例如下：
  ```bash
 	cmake -DRUN_MODE=cpu -DNPU_ARCH=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j; # CPU调试模式
 	cmake -DRUN_MODE=sim -DNPU_ARCH=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j; # NPU仿真模式
 	```
 	若需详细了解CPU调试相关内容，请参考[03_cpudebug样例](../../../01_utilities/03_cpudebug)。

- 编译选项说明
 	 
 	| 选项 | 可选值 | 说明 |
 	|------|--------|------|
 	| `RUN_MODE` | `npu`（默认）、`cpu`、`sim` | 运行模式：NPU 运行、CPU调试、NPU仿真 |
 	| `NPU_ARCH` | `dav-2201`（默认）、`dav-3510` | NPU 架构：dav-2201 对应 Atlas A2/A3 系列、dav-3510 对应 Ascend 950PR/Ascend 950DT |
 	  `SCENARIO_NUM` | `1`（默认）、`2` | 场景编号：1（普通转置）、2（增强转置） |

  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```

  
