# InitDetermineComputeWorkspace样例

## 概述

本样例模拟8个核进行数据处理，使用确定性计算接口保证核间运行顺序，进行原子累加，具体步骤如下：
首先调用InitDetermineComputeWorkspace初始化GM共享内存的值；然后调用WaitPreBlock通过读GM地址中的值，确认是否需要继续等待；接着调用SetAtomicAdd开启原子累加；最后，调用NotifyNextBlock通过写GM地址，通知下一个核当前核的操作已完成，下一个核可以进行操作。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── init_determine_compute_workspace
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── init_determine_compute_workspace.asc      // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  初始化GM共享内存的值，完成初始化后才可以调用WaitPreBlock和NotifyNextBlock。

- 算子规格：  
  <table>
    <tr>
      <td rowspan="3" align="center">算子输入</td>
      <td align="center">name</td>
      <td align="center">shape</td>
      <td align="center">data type</td>
      <td align="center">format</td>
    </tr>
    <tr>
      <td align="center">x</td>
      <td align="center">256</td>
      <td align="center">float</td>
      <td align="center">ND</td>

    </tr>
    <tr>
      <td align="center">workspace</td>
      <td align="center">256</td>
      <td align="center">int32_t</td>
      <td align="center">ND</td>
    </tr>
    <tr>
      <td align="center">算子输出</td>
      <td align="center">z</td><td align="center">256</td>
      <td align="center">float</td>
      <td align="center">ND</td>
    </tr>
    <tr>
      <td align="center">核函数名</td>
      <td colspan="4" align="center">kernel_init_determine_compute_workspace</td>
    </tr>  
  </table>

- 算子实现：  
  初始化GM共享内存的值，完成初始化后才可以调用WaitPreBlock和NotifyNextBlock。

  - Kernel实现

    计算逻辑是：在Process中使用InitDetermineComputeWorkspace基础API接口完成workspace初始化，然后调用WaitPreBlock确认是否需要继续等待；接着调用SetAtomicAdd开启原子累加，通过DataCopy将数据从dstLocal累加至GM；然后调用DisableDmaAtomic关闭原子累加；最后，调用NotifyNextBlock通过写GM地址，通知下一个核当前核的操作已完成，下一个核可以进行操作

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
  test pass
  ```
