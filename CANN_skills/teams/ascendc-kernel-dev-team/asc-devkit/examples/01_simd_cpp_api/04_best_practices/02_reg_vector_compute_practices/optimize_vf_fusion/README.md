# optimize_vf_fusion样例

## 概述
本样例演示了SIMD场景下，基于RegBase编程范式，通过VF融合优化算子代码实现。样例中定义了两个vf函数，其中，
DivVF函数实现对输入向量中的每个数取倒数操作，AddVF函数实现对输入向量的加一操作，
DivVF和AddVF会被编译器融合为一个VF函数。

## 支持的产品
- Ascend 950PR/Ascend 950DT

## 目录结构介绍
```
├── optimize_vf_fusion
│   ├── scripts
│   │   ├── gen_data.py                // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt                 // 编译工程文件
│   ├── data_utils.h                   // 数据读入写出函数
│   └── optimize_vf_fusion.asc         // AscendC算子实现 & 调用样例
```

## 算子描述
- 算子功能：  
  输入向量长度为1024，算子对输入向量中的每个数进行取倒数并且加一，编译器通过VF融合将算子代码中的多个VF融合为一个VF。
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
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">optimize_vf_fusion</td></tr>
  </table>
- 算子实现：  
  输入向量长度为1024，算子对输入向量中的每个数进行取倒数并且加一，编译器通过VF融合将算子代码中的多个VF融合为一个VF。
  
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