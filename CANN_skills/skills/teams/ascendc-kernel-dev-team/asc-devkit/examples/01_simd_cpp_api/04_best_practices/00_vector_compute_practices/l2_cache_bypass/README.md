# L2 CacheMode算子直调样例
## 概述

本样例介绍了设置L2 CacheMode的方法以及其影响场景，并提供核函数直调方法。

## 支持的产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍
```
├── 03_l2_cache_bypass
│   ├── scripts
│   │   ├── gen_data.py                    // 输入数据和真值数据生成脚本
│   │   └── verify_result.py               // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt                     // 编译工程文件
│   ├── data_utils.h                       // 数据读入写出函数
│   └── l2_cache_bypass.asc                // Ascend C算子实现 & 调用样例
```

## 算子描述  
- 算子功能：  
  Add算子实现了两个Shape不相同的Tensor相加，返回相加结果的功能。

  对应的数学表达式为：

  ```python
  z = x + y
  ```

  - x：输入，形状为\[5120, 5120]，数据类型为float；
  - y：输入，形状为\[5120, 15360]，数据类型为float；
  - z：输出，形状为\[5120, 15360]，数据类型为float；

- 算子规格：

    <table>
    <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
    </tr>
    <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
    <tr><td align="center">x</td><td align="center">5120 * 5120</td><td align="center">float</td><td align="center">ND</td></tr>
    <tr><td align="center">y</td><td align="center">5120 * 15360</td><td align="center">float</td><td align="center">ND</td></tr>
    </tr>
    </tr>
    <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">5120 * 15360</td><td align="center">float</td><td align="center">ND</td></tr>
    </tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom_v1/add_custom_v2</td></tr>
    </table>


- 算子实现：

  - kernel实现    

    本样例主要介绍数据搬运中设置合理CacheMode对搬运效率的影响，在Global Memory的数据访问中，如果数据只需要访问一次，后续不需要重复读取，那么这种场景下可以设置Global Memory的CacheMode为CACHE_MODE_DISABLED，在这种模式下数据访问将不经过L2 Cache，避免影响需要重复访问的数据，从而提升数据访问效率。

    本样例中共有2个实现版本：
    add_custom_v1：基础版本，从列方向切分，每个核计算5120×128的数据量，共有40个核参与计算。
    add_custom_v2：在add_custom_v1基础上，设置y和z的CacheMode为CACHE_MODE_DISABLED，避免替换已进入Cache的x数据，影响搬运效率。


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
  python3 ../scripts/verify_result.py output/output_z_1.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  python3 ../scripts/verify_result.py output/output_z_2.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```