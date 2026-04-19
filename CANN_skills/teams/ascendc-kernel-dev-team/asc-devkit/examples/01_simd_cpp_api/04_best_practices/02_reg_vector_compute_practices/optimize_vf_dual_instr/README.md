# optimize_vf_dual_instr样例

## 概述
本样例演示了SIMD场景下，基于RegBase编程范式下vf指令双发优化的样例，通过合理拆分VF循环，适当搬出中间结果到UB，减少数据依赖。

## 支持的产品
- Ascend 950PR/Ascend 950DT

## 目录结构介绍
```
├── optimize_vf_dual_instr
│   ├── scripts
│   │   ├── gen_data.py                    // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt                     // 编译工程文件
│   ├── data_utils.h                       // 数据读入写出函数
│   └── optimize_vf_dual_instr.asc         // AscendC算子实现 & 调用样例
```

## 算子描述
- 算子功能：  
  算子实现两个长度为2048的向量相加，然后对结果向量进行加60操作，过程中通过合理拆分VF循环使能指令双发。

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="3" align="center">AIV算子</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">2048</td><td align="center">float</td></tr>
   <tr><td align="center">y</td><td align="center">2048</td><td align="center">float</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">2048</td><td align="center">float</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">optimize_vf_dual_instr</td></tr>
  </table>
- 算子实现：  
  算子实现两个长度为2048的向量相加，然后对结果向量加60操作，过程中通过合理拆分VF循环使能指令双发。
  
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
  mkdir -p build && cd build;                                               # 创建并进入build目录
  cmake ..;make -j;                                                         # 编译工程
  python3 ../scripts/gen_data.py                                            # 生成测试输入数据
  ./demo                                                                    # 执行编译生成的可执行程序，执行样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```