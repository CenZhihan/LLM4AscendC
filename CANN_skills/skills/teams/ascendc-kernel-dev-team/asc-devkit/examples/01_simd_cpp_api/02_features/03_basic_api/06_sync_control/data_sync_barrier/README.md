# DataSyncBarrier样例

## 概述

本样例介绍DataSyncBarrier的调用，该接口功能：阻塞后续的指令执行，直到所有之前的内存访问指令（需要等待的内存位置可通过参数控制）执行结束，适用于核内标量单元流水（PIPE_S）的同步。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── data_sync_barrier
│   ├── CMakeLists.txt                   // 编译工程文件
│   └── data_sync_barrier.asc      // Ascend C算子实现 & 调用样例
```

## 算子描述

核内单流水同步指令包含[PipeBarrier](../pipe_barrier/)和DataSyncBarrier，其中调用PipeBarrier<PIPE_S>()会引发硬件错误。因此当需要做标量流水之间的同步需要使用DataSyncBarrier。

下面介绍 DataSyncBarrier 的一种使用场景：
系统中有两个 AIV 核（核 0 与核 1），变量 x、y 初始值均为 1。核 0 通过标量流水接口[WriteGmByPassDCache](../../16_scalar_computation/write_gm_by_pass_dcache/)向GM依次写入 x=7、y=6；核 1 持续轮询变量 y，直到检测到 y=6 后，再从 GM 中读取变量 x 并打印。

预期行为：当 y=6 写入 GM 时，x=7 必须已完成写入 GM，因此核 1 最终打印的 x 值应为 7。

但由于标量流水不能保证指令间的写序一致性，若不加同步，可能出现 y 已更新而 x 尚未写入 GM 的情况，导致核 1 读到错误的 x 值。
此时需要插入同步指令 DataSyncBarrier，以确保两次写操作顺序执行。

如下所示，核 0 通过同步指令完成对GM中张量srcGlobal第 1 个元素与第 0 个元素的顺序写入。调用DataSyncBarrier后，将阻塞后续WriteGmByPassDCache指令的执行，直至此前所有WriteGmByPassDCache类型的 GM(对应DDR) 访问指令全部完成。

        if (blockIdx == 0) {
            AscendC::WriteGmByPassDCache<T>(reinterpret_cast<__gm__ T *>(srcGm)+1, value+1);
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>(); // 三种取值，ALL(包含DDR+UB)、DDR、UB
            AscendC::WriteGmByPassDCache<T>(reinterpret_cast<__gm__ T *>(srcGm), value); // value = 6
        }

在核 1 中，一直循环读取GM中输入tensor srcGlobal的第0个元素的取值value，当value的取值与核 0 向srcGlobal写入的值相等时，此时可以认为GM中srcGlobal第 1 个元素也已经被核 0 写入。

        if (blockIdx == 1) {
            while(true){
                __gm__ T *addr = const_cast<__gm__ T *>(srcGlobal.GetPhyAddr());
                T value = AscendC::ReadGmByPassDCache<T>(addr);
                if(value == 6){
                    __gm__ T *addr = const_cast<__gm__ T *>(srcGlobal.GetPhyAddr());
                    T valueNew = AscendC::ReadGmByPassDCache<T>(addr+1);
                    AscendC::printf("GM上addr+1位置的值：%u\n",valueNew);
                    AscendC::WriteGmByPassDCache<T>(reinterpret_cast<__gm__ T *>(dstGm), 2 * valueNew);
                    return;
                }
            }
        }

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
  ./demo                        # 执行编译生成的可执行程序，执行样例
  ```

  执行结果如下，说明精度对比成功。

  ```bash
  test pass!
  ```
