# ReduceComputation样例

## 概述

本样例基于ReduceMax/ReduceMin/ReduceSum接口实现归约计算，支持以下6种场景（scenario）：
    <table>
 	  	 	<tr>
 	  	 		<td>scenarioNum</td>
 	  	 		<td>场景</td>
          <td>描述</td>
 	  	 	</tr>
 	  	 	<tr>
 	  	 		<td>1</td>
 	  	 		<td>ReduceMax前n个数据计算</td>
          <td>从输入tensor的前n个数据中找出最大值及最大值对应的索引位置</td>
 	  	 	</tr>
 	  	 	<tr>
 	  	 		<td>2</td>
 	  	 		<td>ReduceMax高维切分计算</td>
          <td>从所有输入数据中找出最大值及最大值对应的索引位置，使用mask控制每次迭代内参与计算的元素</td>
 	  	 	</tr>
        <tr>
 	  	 		<td>3</td>
 	  	 		<td>ReduceMin前n个数据计算</td>
          <td>从输入tensor的前n个数据中找出最小值及最小值对应的索引位置</td>
 	  	 	</tr>
        <tr>
 	  	 		<td>4</td>
 	  	 		<td>ReduceMin高维切分计算</td>
          <td>从所有输入数据中找出最小值及最小值对应的索引位置，使用mask控制每次迭代内参与计算的元素</td>
 	  	 	</tr>
        <tr>
 	  	 		<td>5</td>
 	  	 		<td>ReduceSum前n个数据计算</td>
          <td>对输入tensor的前n个数据求和</td>
 	  	 	</tr>
        <tr>
 	  	 		<td>6</td>
 	  	 		<td>ReduceSum高维切分计算</td>
          <td>对所有输入数据求和，使用mask控制每次迭代内参与计算的元素</td>
 	  	 	</tr>
 	  	 </table>

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── reduce_computation
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── reduce_computation.asc  // Ascend C样例实现 & 调用样例
```

## 样例描述

- 样例功能：

  本样例基于ReduceMax/ReduceMin/ReduceSum接口实现归约计算，包含前n个数据计算接口和tensor高维切分计算接口。接口资料参考[ReduceMax](../../../../../../docs/api/context/ReduceMax.md)/[ReduceMin](../../../../../../docs/api/context/ReduceMin.md)/[ReduceSum](../../../../../../docs/api/context/ReduceSum.md)。

- 样例规格：

  不同场景的输入输出规格如下表所示：

  <table border="2" align="center">
  <tr>
    <th align="center">scenarioNum</th>
    <th align="center">样例场景</th>
    <th align="center">输入name</th>
    <th align="center">输入shape</th>
    <th align="center">输入data type</th>
    <th align="center">输出name</th>
    <th align="center">输出shape</th>
    <th align="center">输出data type</th>
  </tr>
  <tr>
    <td align="center">1</td>
    <td align="center">ReduceMax前n个数据计算</td>
    <td align="center">x</td>
    <td align="center">[1, 288]</td>
    <td align="center">half</td>
    <td align="center">y</td>
    <td align="center">[1, 16]</td>
    <td align="center">half</td>
  </tr>
  <tr>
    <td align="center">2</td>
    <td align="center">ReduceMax高维切分计算</td>
    <td align="center">x</td>
    <td align="center">[1, 512]</td>
    <td align="center">half</td>
    <td align="center">y</td>
    <td align="center">[1, 16]</td>
    <td align="center">half</td>
  </tr>
  <tr>
    <td align="center">3</td>
    <td align="center">ReduceMin前n个数据计算</td>
    <td align="center">x</td>
    <td align="center">[1, 288]</td>
    <td align="center">half</td>
    <td align="center">y</td>
    <td align="center">[1, 16]</td>
    <td align="center">half</td>
  </tr>
  <tr>
    <td align="center">4</td>
    <td align="center">ReduceMin高维切分计算</td>
    <td align="center">x</td>
    <td align="center">[1, 512]</td>
    <td align="center">half</td>
    <td align="center">y</td>
    <td align="center">[1, 16]</td>
    <td align="center">half</td>
  </tr>
  <tr>
    <td align="center">5</td>
    <td align="center">ReduceSum前n个数据计算</td>
    <td align="center">x</td>
    <td align="center">[1, 288]</td>
    <td align="center">half</td>
    <td align="center">y</td>
    <td align="center">[1, 16]</td>
    <td align="center">half</td>
  </tr>
  <tr>
    <td align="center">6</td>
    <td align="center">ReduceSum高维切分计算</td>
    <td align="center">x</td>
    <td align="center">[8320]</td>
    <td align="center">half</td>
    <td align="center">y</td>
    <td align="center">[1, 16]</td>
    <td align="center">half</td>
  </tr>
  </table>

- 样例实现：

  - Kernel实现
    - 调用DataCopy基础API将数据从GM（Global Memory）搬运到UB（Unified Buffer），并将归约计算后的数据搬出到GM（Global Memory）。
    - 调用ReduceMax/ReduceMin/ReduceSum接口完成归约计算。
    - 在ReduceSum前n个数据计算场景，调用[GetReduceRepeatSumSpr](../../../../../../docs/api/context/GetReduceRepeatSumSpr(ISASI).md)获取计算结果。

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
  python3 ../scripts/verify_result.py -scenarioNum=$SCENARIO output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功：
  ```bash
  test pass!
  ```
