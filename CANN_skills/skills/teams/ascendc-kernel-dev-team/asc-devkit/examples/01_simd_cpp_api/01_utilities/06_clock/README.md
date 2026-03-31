# 基于gather算子的SIMT时间戳打点功能实现样例

## 概述

本样例基于gather算子，演示在```__simt_vf__```中使用```clock()```接口实现时间戳打点的方法，精准记录算子执行前后的时间戳并统计执行耗时，为算子性能分析提供可靠的时间数据支撑。

## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构

```
├── 04_clock
│   ├── CMakeLists.txt         # cmake编译文件
│   ├── gather.asc             # Ascend C算子实现gather调用样例
|   └── README.md
```

## 算子描述

- 算子功能:

  本样例详细展示了在```__simt_vf__```函数中使用```clock()```接口的实践方式，通过在算子核心逻辑执行前后分别调用clock()接口记录时钟周期数，计算并输出算子执行的耗时周期，实现对算子执行效率的精准量化。

  样例使用gather算子进行```clock()```的功能的说明，具体功能描述可参考[gather算子详情](../../02_features/03_simt/simt_gather/README.md)章节。

- 算子实现:
  ```c++
  __simt_vf__ __launch_bounds__(THREAD_COUNT) inline void simt_gather(
      __gm__ float* input,
      __gm__ uint32_t* index,
      __gm__ float* gather_output,
      uint32_t in_shape0,
      uint32_t in_shape1,
      uint32_t index_total_length)
  {
      uint64_t t1 = clock();
      ...
      uint64_t t2 = clock();
      printf("simt_vf execute cycle : %lu\n", t2 - t1);
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
  执行结果如下，说明时间打点功能和精度对比成功。
  ```
  simt_vf execute cycle : 22289
  ...
  [Success] Case accuracy is verification passed.
  ```
