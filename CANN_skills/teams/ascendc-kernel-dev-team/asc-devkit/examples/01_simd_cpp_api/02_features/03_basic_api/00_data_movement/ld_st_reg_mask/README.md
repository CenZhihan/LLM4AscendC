# ld_st_reg_mask样例

## 概述
本样例基于RegBase编程范式实现UB(Unified Buffer)对MaskReg(掩码寄存器)的搬入搬出，以及使用mask进行掩码搬出的操作。样例使用LoadAlign，StoreAlign，CreateMask，Duplicate接口。

## 支持的产品
- Ascend 950PR/Ascend 950DT

## 目录结构介绍
```
├── ld_st_reg_mask
│   ├── scripts
│   │   ├── gen_data.py                // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt                 // 编译工程文件
│   ├── data_utils.h                   // 数据读入写出函数
|   |── README.md                      // 样例介绍
│   └── ld_st_reg_mask.asc             // AscendC样例实现 & 调用样例
```

## 样例描述
- 样例功能：  
  样例输入一个数据类型为uint8_t，数据量为1024的向量，取前向量的前256bit作为掩码进行Duplicate计算，随后将掩码寄存器的比特位全部置为1，将掩码寄存器的256bit值保存到UB中
- 样例规格：
  <table>
  <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="3" align="center">AIV样例</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 1024]</td><td align="center">uint8_t</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">样例输出</td><td align="center">y</td><td align="center">[1, 1024]</td><td align="center">uint8_t</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">ld_st_reg_mask</td></tr>
  </table>
- 样例实现：  
<p align="center">
  <img src="figures/ld_st_reg_mask.png" width="1000">
</p>

  1. 在CopyVF函数里，调用Load接口从UB搬运256bit(32*uint8_t)数据到MaskReg，以实现mask的动态设置。本样例中，32个uint8_t数据设置为1,0,...,1，即第一个数和最后一个数为1(b'00000001)，因为芯片每次从低位读取数字，所以这32个数字最终填充到MaskReg中为b'1000...1...000。
  2. 调用Duplicate指令进行数据填充，MaskReg可以指示计算过程中参与计算的元素，由step1可知MaskReg中第1个bit和第249个bit为1，使用该掩码将只填充RegTensor第1个数和第249个数为2。
  3. 使用Store接口将RegTensor内的结果保存到UB。
  4. 将MaskReg中的bit位全部置为1，通过Store接口将MaskReg中的数据保存到UB(地址=step3中的保存地址+256B)，实现MaskReg数据存储在UB上的功能。对应32个uint8_t数，每个数的每位比特位都为1，因此每个数的值都是255(0xFFFF..FF)。
  
  - 调用实现  
    使用内核调用符<<<>>>调用核函数。
    
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
  mkdir -p build && cd build;                                               # 创建并进入build目录
  cmake -DNPU_ARCH=dav-3510 ..;make -j;                                     # 编译工程（默认npu模式）
  python3 ../scripts/gen_data.py                                            # 生成测试输入数据
  ./demo                                                                    # 执行编译生成的可执行程序，执行样例
  ```

  使用 CPU调试 或 NPU仿真 模式时，添加 `-DRUN_MODE=cpu` 或 `-DRUN_MODE=sim` 参数即可。

  示例如下：
  ```bash
  cmake -DRUN_MODE=cpu -DNPU_ARCH=dav-3510 ..;make -j; # cpu调试模式
  cmake -DRUN_MODE=sim -DNPU_ARCH=dav-3510 ..;make -j; # NPU仿真模式
  ```

  > **注意：** 切换编译模式前需清理 cmake 缓存，可在 build 目录下执行 `rm CMakeCache.txt` 后重新 cmake。

- 编译选项说明

  | 选项 | 可选值 | 说明 |
  |------|--------|------|
  | `RUN_MODE` | `npu`（默认）、`cpu`、`sim` | 运行模式：NPU 运行、CPU调试、NPU仿真 |
  | `NPU_ARCH` | `dav-3510`（默认） | NPU 架构：dav-3510 对应 Ascend 950PR/Ascend 950DT |

- 执行结果

  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```