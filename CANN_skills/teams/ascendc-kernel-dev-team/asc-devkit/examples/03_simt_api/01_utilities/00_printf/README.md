# 基于gather算子的SIMT printf功能实现样例

## 概述

本样例演示在SIMT编程下使用```printf()```接口实现上板打印进行功能调试的方法。

## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构

```
├── 00_printf
│   ├── CMakeLists.txt         # cmake编译文件
│   ├── printf.asc             # Ascend C算子实现加printf打印的调用样例
|   └── README.md
```

## 算子描述

- 算子功能:

  本样例详细展示了在SIMT实现函数中使用```printf()```接口的实践方式，实现对算子执行过程中打印每个线程的变量信息。


- 算子实现:
  ```cpp
  __global__ void  simt_printf(float* input, uint32_t in_shape)
  {
      // Calculate global thread ID
      int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (threadIdx.x < 3) {
      printf("[SIMT %s] thread index[%u], input data shape: %u\n", "print 1", idx, in_shape);
      printf("[SIMT %s] input addr: %p value[%u]: %f\n", "print 2",  input, idx, input[idx]);
      }
  }
  ```

## 编译运行

在本样例根目录下执行如下步骤，编译并执行算子。

- 配置环境变量
  请根据当前环境上CANN开发套件包的[安装方式](../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
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
  cmake ..; make -j;            # 编译工程
  ./demo                        # 执行样例
  ```
  执行后有如下打印信息，说明打印功能正常。
  ```
  [SIMT print 1] thread index[34], input data shape: 128
  [SIMT print 1] thread index[2], input data shape: 128
  [SIMT print 1] thread index[33], input data shape: 128
  [SIMT print 1] thread index[1], input data shape: 128
  [SIMT print 1] thread index[32], input data shape: 128
  [SIMT print 1] thread index[0], input data shape: 128
  [SIMT print 2] input addr: 0x120000016000 value[2]: 3.118000
  [SIMT print 2] input addr: 0x120000016000 value[1]: 2.118000
  [SIMT print 2] input addr: 0x120000016000 value[0]: 1.118000
  [SIMT print 2] input addr: 0x120000016000 value[34]: 35.118000
  [SIMT print 2] input addr: 0x120000016000 value[33]: 34.118000
  [SIMT print 2] input addr: 0x120000016000 value[32]: 33.118000
  ```
