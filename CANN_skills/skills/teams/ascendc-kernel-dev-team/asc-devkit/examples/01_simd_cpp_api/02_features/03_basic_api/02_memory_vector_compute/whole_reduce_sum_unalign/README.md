# whole_reduce_sum算子直调样例

## 概述

本样例介绍非对齐whole_reduce_sum算子的核函数直调方法，采用核函数<<<>>>调用，有效降低调度开销，实现高效的算子执行。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── whole_reduce_sum
│   ├── scripts
│   │   ├── gen_data.py                // 输入数据和真值数据生成脚本
│   │   └── verify_result.py           // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt                 // 编译工程文件
│   ├── data_utils.h                   // 数据读入写出函数
│   └── whole_reduce_sum_custom.asc    // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  本非对齐WholeReduceSum算子对二维Tensor输入作行归约求和。其python代码表示如下：
  ```python
  y = np.sum(x, axis=1)
  ```

  本算子的输入x的shape为[13, 123]，数据类型为float16，每行的数据（246B）不满足32B对齐约束。输出y的shape为[13]，数据类型为float16，长度为26B，也不满足32B对齐。
- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">WholeReduceSum</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">13 * 123</td><td align="center">float16</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">13</td><td align="center">float16</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">whole_reduce_sum_custom</td></tr>
  </table>
- 算子实现：  
  非对齐WholeReduceSum算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor x非对齐搬运到Local Memory，存储在xLocal中，Compute任务负责对xLocal执行按行规约求和操作，计算结果存储在yLocal中，CopyOut任务负责将输出数据从yLocal非对齐搬运至Global Memory上的输出Tensor y中。

  本样例的输入x的shape为[13, 123]，数据类型为float16，每行的数据（246B）不满足32B对齐约束。输出y的shape为[13]，数据类型为float16，长度为26B，也不满足32B对齐。由于输入Tensor单行数据不满足32B对齐约束，我们把x从GM搬入到xLocal时应该使用DataCopyPad接口进行非对齐搬运。**注意到输出y也不满足32B对齐，我们申请y的Global Memory Buffer时应该向上32B对齐，避免搬出时访问非法内存。**
      
  我们在UB上计算行和时，应注意每行只应计算前cols个数据，因此需要通过mask等参数控制WholeReduceSum高阶API的行为。
      
  将yLocal从UB搬出到GM，由于只需要搬出reducesBytes，需要用DataCopyPad进行非对齐搬出。
    
  TilingData参数设计，TilingData参数本质上是和并行数据切分相关的参数，本样例算子使用了3个tiling参数：totalLength，rows，cols 。totalLength是指需要计算的数据量大小，rows是指二维输入Tensor的行数，cols则是指每行的数据个数。通过将totalLength，rows，cols传递到kernel侧，就可以实现将输入数据按行切分，然后规约求和。tiling实现代码中通过上下文获取输入输出的shape信息，并对应设置TilingData。
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