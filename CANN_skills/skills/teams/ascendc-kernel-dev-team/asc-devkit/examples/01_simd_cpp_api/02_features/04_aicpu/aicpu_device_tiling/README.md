# AI CPU算子Tiling下沉样例介绍
## 概述

本样例介绍使用AI CPU算子进行tiling下沉计算，使用<<<>>>内核调用符来完成算子核函数在NPU侧运行验证的基础流程。

## 支持的产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍
```
├── 02_aicpu_device_tiling
│   ├── CMakeLists.txt                     // 编译工程文件
│   ├── aicore_kernel.asc                  // AI Core算子实现
│   ├── kernel_args.h                      // tiling结构体头文件
│   ├── main.asc                           // AI CPU算子与AI Core算子调用
│   └── aicpu_tiling.aicpu                 // AI CPU算子实现
```

## 算子描述
- main.asc中内AI CPU算子与AI Core算子均使用内核调用符<<<...>>>进行调用，AI CPU算子将tiling计算的结果传给AI Core算子。
- AI CPU算子与AI Core算子在不同stream上进行launch，样例中分别为aicpu_stream与aicore_stream，event用于记录stream上已下发的任务。使用aclrtRecordEvent在指定stream中记录event，使用aclrtStreamWaitEvent阻塞指定的stream，直到指定的event完成。

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
  ./demo                        # 执行编译生成的可执行程序，执行样例
  ```
  执行结果如下，说明执行成功。
  ```bash
  MyAicpuKernel inited
  MyAicpuKernel inited type 1 mode 2 len 4 end!
  Hello World: int mode 2 len 4 m 10.
  Hello World: int mode 2 len 4 m 10.
  Hello World: int mode 2 len 4 m 10.
  ```