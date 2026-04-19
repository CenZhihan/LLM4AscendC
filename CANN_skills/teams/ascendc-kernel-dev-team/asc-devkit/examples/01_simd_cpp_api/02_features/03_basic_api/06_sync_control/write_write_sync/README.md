# write_write_sync样例

## 概述
本样例演示了SIMD场景下，基于RegBase编程范式下写操作与写操作之间依赖场景下的同步。

## 支持的产品
- Ascend 950PR/Ascend 950DT

## 目录结构介绍
```
├── read_write_sync
│   ├── scripts
│   │   ├── gen_data.py                // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt                 // 编译工程文件
│   ├── data_utils.h                   // 数据读入写出函数
│   └── read_write_sync.asc            // AscendC算子实现 & 调用样例
```

## 算子描述
- 算子功能：  
  算子输入一个长度为1024，数据类型为float的向量A，，对该向量进行以下操作，
  1. 将向量A下标为0到63的位置填充1
  2. 将下标为64到127的位置填充2
  通过写写同步使得操作1在前，操作2在后。
- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="3" align="center">AIV算子</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">1024</td><td align="center">float</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">1024</td><td align="center">float</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">write_write_sync</td></tr>
  </table>
- 算子实现：  
  主要包括以下步骤：
  1. 初始化参数
  2. 将数据从GM搬运到UB
  3. 调用ComputeVF进行计算，将向量A下标为0到63的位置填充为1，下标为64到127的位置填充为2，通过写写同步使得填充为1的操作在前，填充为2的操作在后。结果写回UB
  4. 将结果从UB搬运回GM
  
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
  mkdir -p build && cd build;                                               # 创建并进入build目录
  cmake ..;make -j;                                                         # 编译工程
  python3 ../scripts/gen_data.py                                            # 生成测试输入数据
  ./demo                                                                    # 执行编译生成的可执行程序，执行样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```