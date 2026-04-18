# 自定义算子工程编译、打包和部署样例

## 概述

本样例以简单自定义算子为基础，展示了其编译、打包成自定义算子包，并部署到CANN环境中的流程。

## 支持的产品

本样例支持如下产品型号：
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas 200I/500 A2 推理产品
- Atlas 推理系列产品

> 注意: 本样例中涉及多个算子示例，请以各个算子示例实际支持的产品型号为准。

## 目录结构介绍

```
├── CMakeLists.txt
├── framework
│   ├── CMakeLists.txt
│   ├── onnx_plugin
│   │   ├── CMakeLists.txt
│   │   └── leaky_relu_custom_plugin.cc
│   └── tf_plugin
│       ├── CMakeLists.txt
│       └── tensorflow_add_custom_plugin.cc
├── op_host
│   ├── CMakeLists.txt
│   ├── add_custom
│   │   └── add_custom_host.cpp
│   ├── add_custom_template
│   │   └── add_custom_template.cpp
│   ├── add_custom_tiling_sink
│   │   ├── add_custom_tiling_sink.cpp
│   │   ├── add_custom_tiling_sink_tiling.cpp
│   │   └── add_custom_tiling_sink_tiling.h
│   └── leaky_relu_custom
│       └── leaky_relu_custom_host.cpp
└── op_kernel
    ├── CMakeLists.txt
    ├── add_custom
    │   ├── add_custom_kernel.cpp
    │   └── add_custom_tiling.h
    ├── add_custom_template
    │   ├── add_custom_template.cpp
    │   ├── add_custom_template_tiling.h
    │   └── tiling_key_add_custom_template.h
    ├── add_custom_tiling_sink
    │   ├── add_custom_tiling_sink_kernel.cpp
    │   └── add_custom_tiling_sink_tiling_struct.h
    └── leaky_relu_custom
        ├── leaky_relu_custom_kernel.cpp
        └── leaky_relu_custom_tiling.h
```

## 样例描述

Add计算公式为：

```
z = x + y
```

AddCustomTilingSink、AddCustomTemplate与Add的核函数功能一致。其中，
  - AddCustomTemplate展示了Tiling模板编程，添加的模板参数包括输入的数据类型、shape等，根据模板参数，简化或统一样例的实现逻辑，开发者可以在模板参数中定义需要的信息，如输入输出的数据类型，其他扩展参数等；
  - AddCustomTilingSink用于展示Tiling下沉场景，通过`DEVICE_IMPL_OP_OPTILING`将Tiling函数同时注册到host和device上执行。

LeakyRelu计算公式为：

$$
y=
\begin{cases}
x, \quad x\geq 0\\
a*x, \quad x<0
\end{cases}
$$
其中a为scalar值。

## 样例规格描述

- Add
  <table border="2" align="center">
  <caption>表1：Add样例规格描述</caption>
  <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[8, 2048]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">[8, 2048]</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">样例输出</td><td align="center">z</td><td align="center">[8, 2048]</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom, add_custom_tiling_sink</td></tr>
  </table>

- AddCustomTemplate

  <table border="2" align="center">
  <caption>表2：AddCustomTemplate样例规格描述</caption>
  <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[8, 2048]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">[8, 2048]</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">样例输出</td><td align="center">z</td><td align="center">[8, 2048]</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_template_custom</td></tr>
  <tr><td rowspan="6" align="center">模板参数</td><td colspan="4" align="center">template&lt;typename D_T_X, typename D_T_Y, typename D_T_Z, int TILE_NUM, int IS_SPLIT&gt;</td>
      <tr><td>D_T_X</td><td colspan="1">typename</td><td colspan="2">数据类型(float16，float)</td></tr>
      <tr><td>D_T_Y</td><td colspan="1">typename</td><td colspan="2">数据类型(float16，float)</td></tr>
      <tr><td>D_T_Z</td><td colspan="1">typename</td><td colspan="2">数据类型(float16，float)</td></tr>
      <tr><td>TILE_NUM</td><td colspan="1">int</td><td colspan="2">切分数量</td></tr>
      <tr><td>IS_SPLIT</td><td colspan="1">int</td><td colspan="2">是否切分</td></tr>
  </tr>
  </table>

- LeakyRelu

  <table border="2" align="center">
  <caption>表3：LeakyRelu样例规格描述</caption>
  <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="4" align="center">LeakyRelu</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[8, 200, 1024]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">negative_slope</td><td align="center">0.0</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">样例输出</td><td align="center">y</td><td align="center">[8, 200, 1024]</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">leaky_relu_custom</td></tr>
  </table>

## 代码实现介绍

- Add</br>
  - kernel实现：</br>
    基于Ascend C提供的矢量计算接口`Add`实现。</br>

  - tiling实现：</br>
    TilingData参数设计，`AddCustomTilingData`参数本质上是和并行数据切分相关的参数，本示例样例使用了2个tiling参数：`totalLength`和`tileNum`。`totalLength`是指需要计算的数据量大小，`tileNum`是指每个核上总计算数据分块个数。</br>

- AddCustomTemplate</br>
  - kernel实现：</br>
    与Add一致。
  - Tiling模板设计：</br>
    本示例使用了5个模板参数，`D_T_X`、`D_T_Y`、`D_T_Z`分别是指输入x、输入y、输出z的数据类型，`TILE_NUM`是指每个核上总计算数据分块个数，`IS_SPLIT`是是否使能数据分块计算，`IS_SPLIT`为0时`TILE_NUM`无效。通过模板参数组合替代传统的TilingKey。

  - TilingData参数设计：</br>
    本示例样例使用了1个tiling参数，`totalLength`是指所有核需要计算的数据量总大小。</br>


- AddCustomTilingSink</br>
  - kernel实现：</br>
    功能与Add一致，kernel通过`KERNEL_TASK_TYPE_DEFAULT`接口将样例强制指定在AIC、AIV混合场景运行，满足Tiling下沉算子条件。将所有的Tiling函数逻辑单独在`add_custom_tiling_sink_tiling.cpp`中实现，并通过`DEVICE_IMPL_OP_OPTILING`接口注册下沉的Tiling函数。</br>

- LeakyRelu</br>
  - kernel实现：</br>
    基于高阶API接口`LeakyRelu`实现。</br>

  - tiling实现：</br>
    TilingData参数设计，`LeakyReluCustomTilingData`参数本质上是和并行数据切分相关的参数，本示例样例使用了3个tiling参数：`totalLength`、`tileNum`、`negativeSlope`。`totalLength`、`tileNum`与Add样例类似，`negativeSlope`表示LeakyRelu的负半轴斜率系数，作为计算参数传递给kernel侧。</br>


## 编译运行

在本样例根目录下执行如下步骤，编译、打包并部署自定义样例包。

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

- 编译、打包样例并部署

  ```bash
  mkdir -p build && cd build
  cmake .. && make -j binary package
  ./custom_opp_*.run
  ```

  执行结果如下，说明执行成功。

  ```log
  SUCCESS
  ```
