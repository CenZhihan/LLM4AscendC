# PipeBarrier样例

## 概述
本样例基于PipeBarrier实现核内同步，适用于以下场景：阻塞相同流水，具有数据依赖的相同流水之间需要插入此同步。本样例，使用静态Tensor编程方式，所有核内同步操作均需要用户自行处理。
## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── pipe_barrier
│   ├── CMakeLists.txt              // 编译工程文件
│   └── pipe_barrier.asc           // Ascend C算子实现 & 调用样例
```

## 算子描述


注意事项：调用PipeBarrier<PIPE_S>()会引发硬件错误，Scalar流水之间的同步请使用[DataSyncBarrier](../data_sync_barrier/)。


PIPE_MTE2/PIPE_MTE3在搬运地址有重叠的情况下需要开发者插入同步。例如，搬运的目的地址Unified Buffer存在重叠，两条搬运指令之间需要调用PipeBarrier<PIPE_MTE2>()添加MTE2搬入流水的同步。

如下所示，静态Tensor编程方式创建src1和src2两个长度为256的LocalTensor时，存在一段长度为128的地址重叠。


    static constexpr uint32_t src1Addr = 256 * sizeof(float); // src1起始地址，长度为256
    static constexpr uint32_t src2Addr = (256 + 128) * sizeof(float); // src2起始地址，长度为256
    
样例预期的输出是dst = scr0 + scr1，因此为了保证搬入的src2的数据不会覆盖src1在重叠区间的值，必须保证先搬入src2，后搬入src1，此时需要在两条数据搬入指令之间需要调用PipeBarrier<PIPE_MTE2>()添加MTE2搬入流水的同步。

    AscendC::DataCopy(src0Local, src0Global, srcDataSize);
    AscendC::DataCopy(src2Local, src2Global, srcDataSize);
    AscendC::PipeBarrier<PIPE_MTE2>();
    AscendC::DataCopy(src1Local, src1Global, srcDataSize);
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
  mkdir -p build && cd build;      # 创建并进入build目录
  cmake ..;make -j;                # 编译工程
  ./demo                           # 执行编译生成的可执行程序，执行样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```
