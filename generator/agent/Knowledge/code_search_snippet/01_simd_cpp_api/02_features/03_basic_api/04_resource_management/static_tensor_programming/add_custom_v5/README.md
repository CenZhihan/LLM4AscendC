# AddCustomV5样例

## 概述

使用静态Tensor编程方法进行add算子的编程，实现SetFlag/WaitFlag同步指令的循环内外依赖。

## 支持的产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── add_custom_v5
│   ├── scripts
│   │   ├── gen_data.py             // 输入数据和真值数据生成脚本
│   │   └── verify_result.py        // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt              // 编译工程文件
│   ├── data_utils.h                // 数据读入写出函数
│   └── add_custom_v5.asc           // Ascend C算子实现 & 调用样例
```

## 算子描述

算子固定shape为[72, 1024], 通过9次循环实现输入数据累加9次。

计算公式为：

```python
y = 9 * x
```

- x：输入，形状为\[72, 1024]，数据类型为float；
- y：输出，形状为\[72, 1024]，数据类型为float；

## 算子规格描述

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
</tr>
<tr><td rowspan="1" align="center"></td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td rowspan="1" align="center">算子输入</td><td align="center">x</td><td align="center">72 * 1024</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">72 * 1024</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom_v5</td></tr>
</table>

- 算子实现：

  数学表达式为：

  ```
  y = 9 * x
  ```

  计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用计算接口完成输入参数累加，得到最终结果，再搬出到外部存储上。

  该算子的实现流程：将Global Memory上的输入Tensor xGm搬运到Local Memory，存储在xLocal中；完成9次循环，每次循环完成一次矢量加法：yLocal = yLocal + xLocal，循环结束后，yLocal 中存储了9倍xLocal的结果；最后将输出数据从yLocal搬运至Global Memory上的输出Tensor yGm中。

## 编译运行

在本样例根目录下执行如下步骤，编译并执行算子。
- 配置环境变量  
  请根据当前环境上CANN开发套件包的[安装方式](../../../../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
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
  mkdir -p build && cd build;      # 创建并进入build目录
  cmake ..;make -j;                # 编译工程
  python3 ../scripts/gen_data.py   # 生成测试输入数据
  ./demo                           # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```
