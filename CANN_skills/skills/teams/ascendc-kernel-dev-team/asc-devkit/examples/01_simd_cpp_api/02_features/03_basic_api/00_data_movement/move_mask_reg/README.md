# move_mask_reg样例

## 概述
本样例演示了SIMD场景下，基于RegBase编程范式下数据从UB到MaskReg之间的搬入搬出。

## 支持的产品
- Ascend 950PR/Ascend 950DT

## 目录结构介绍
```
├── move_mask_reg
│   ├── scripts
│   │   ├── gen_data.py                // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt                 // 编译工程文件
│   ├── data_utils.h                   // 数据读入写出函数
│   └── move_mask_reg.asc              // AscendC算子实现 & 调用样例
```

## 算子描述
- 算子功能：  
  算子输入一个数据类型为uint8_t，长度为1024的向量，取前向量的前256bit作为掩码进行Duplicate计算，随后将掩码寄存器的比特位全部置为1，将掩码寄存器的256bit值保存到UB中
- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="3" align="center">AIV算子</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">1024</td><td align="center">uint8_t</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">1024</td><td align="center">uint8_t</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">move_mask_reg</td></tr>
  </table>
- 算子实现：  
  主要包括以下步骤：
  1. 初始化参数
  2. 将数据从GM搬运到UB
  3. 调用CopyVF，调用LoadAlign将UB的前256bit(32个uint8_t)数据拷入mask，
  本样例中，前32个uint8_t数据分别为1,0,...,0，芯片每次从低位读取数字，这32个数字最终填充到MaskReg中为
  b'1000...000，然后调用Duplicate指令，使用该掩码将只填充RegTensor第一个数为1，随后将RegTensor内的结果保存到UB。
  随后，算子中将MaskReg中的比特位全部置为1，将该数据输出到UB中，对应32个uint8_t数，每个数的每位比特位都为1，因此每个数的值都是255(0xFFFF..FF)
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