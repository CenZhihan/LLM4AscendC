# mmad_with_sparse样例
## 概述
本样例介绍基础API MmadWithSparse调用样例。

## 支持的产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍
```
├── mmad_with_sparse
│   ├── scripts
│   │   ├── gen_data.py             // 输入数据和真值数据生成脚本
│   │   └── verify_result.py        // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt              // 编译工程文件
│   ├── data_utils.h                // 数据读入写出函数
│   └── mmad_with_sparse.asc        // Ascend C算子实现 & 调用样例
```

## 算子描述
- 算子功能：
  本样例中实现的是[m, n, k]固定为[16, 16, 64]的Matmul算子，其中A和B矩阵为稠密矩阵，使用Ascend C基础Api MmadWithSparse接口实现。Matmul算子的数学表达式为：
  ```
  C = A * B
  ```
- 算子规格：
  <table>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">16 * 64</td><td align="center">float16</td><td align="center">NZ</td></tr>
  <tr><td align="center">b</td><td align="center">32 * 16</td><td align="center">float16</td><td align="center">NZ</td></tr>
  <tr><td align="center">idx</td><td align="center">8 * 16</td><td align="center">float</td><td align="center">NZ</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">16 * 16</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">mmad_with_sparse_custom</td></tr>
  </table>

- 算子实现：
  计算逻辑是：Ascend C提供的矩阵乘计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储并进行分形转换，然后使用计算接口完成两个输入参数矩阵乘运算，得到最终结果，再搬出到外部存储上。  

  矩阵乘加操作中，传入的左矩阵A为稀疏矩阵，右矩阵B为稠密矩阵。对于矩阵A，在MmadWithSparse计算时完成稠密化；对于矩阵B，在计算执行前的输入数据准备时自行完成稠密化，所以输入本接口的B矩阵为稠密矩阵。B稠密矩阵需要通过调用LoadDataWithSparse载入，同时加载索引矩阵，索引矩阵在矩阵B稠密化的过程中生成，再用于A矩阵的稠密化。

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