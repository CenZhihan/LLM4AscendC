# GroupMatmul算子直调样例

## 概述

本样例介绍QuantGroupMatmul算子的高性能实现，支持分组量化矩阵乘与Gelu激活计算，使用<<<>>>内核调用符来完成算子核函数在NPU侧运行验证的基础流程。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── 06_grouped_matmul
│   ├── scripts
│   │   ├── gen_data.py                    // 输入数据和真值数据生成脚本
│   │   └── verify_result.py               // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt                     // 编译工程文件
│   ├── data_utils.h                       // 数据读入写出函数
│   └── quant_group_matmul_custom.asc      // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  算子实现了分组的pertoken量化matmul计算，分组轴为m轴，并对结果进行激活函数Gelu计算。

  QuantGroupMatmul的计算公式为：

  ```python
  offset = 0
  for i in range(g):
      mmOut = x[offset:offset + group[i]] * weight[i] + bias[i]  # Cube计算
      y[offset:offset + group[i]] = Gelu(mmOut * scale[i] * pertokenScale[offset:offset + group[i]])  # vector计算
      offset += group[i]
  ```

  - x：左矩阵，形状为\[m, k]，数据类型为int8；
  - weight：右矩阵，形状为\[g, k, n]，数据类型为int8；
  - bias：矩阵乘偏置，形状为\[g, n]，数据类型为int32，对第i次矩阵乘结果的每一行都采用bias[i]进行偏置；
  - group：记录每组m的大小，数据类型为int64；
  - scale：右矩阵的量化参数，形状为\[g, n]，数据类型为float，用于矩阵乘结果的反量化，对第i次矩阵乘结果采用scale[i]进行反量化；
  - pertokenScale：左矩阵的量化参数，形状为\[m]，数据类型为float，用于矩阵乘结果的反量化，采用与x行相同的索引范围进行反量化；
  - y：输出，存放矩阵乘结果的矩阵，形状为\[m, n]，数据类型为float16；

- 算子规格：

  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">QuantGroupMatmul</td></tr>
  </tr>
  <tr><td rowspan="7" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">1024 * 1024</td><td align="center">int8</td><td align="center">ND</td></tr>
  <tr><td align="center">weight</td><td align="center">8 * 1024 * 8192</td><td align="center">int8</td><td align="center">NZ</td></tr>
  <tr><td align="center">bias</td><td align="center">8 * 8192</td><td align="center">int32</td><td align="center">ND</td></tr>
  <tr><td align="center">group</td><td align="center">8</td><td align="center">int64</td><td align="center">ND</td></tr>
  <tr><td align="center">scale</td><td align="center">8 * 8192</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">pretokenScale</td><td align="center">1024</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">1024 * 8192</td><td align="center">float16</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">quant_group_matmul_custom</td></tr>
  </table>

- 算子实现：  
  本样例中实现的是pertoken量化的QuantGroupMatmul算子。

  - Kernel实现
  
    QuantGroupMatmul算子的计算为：

    ```python
    offset = 0
    for i in range(g):
        mmOut = x[offset:offset + group[i]] * weight[i] + bias[i]  # Cube计算
        y[offset:offset + group[i]] = Gelu(mmOut * scale[i] * pertokenScale[offset:offset + group[i]])  # vector计算
        offset += group[i]
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