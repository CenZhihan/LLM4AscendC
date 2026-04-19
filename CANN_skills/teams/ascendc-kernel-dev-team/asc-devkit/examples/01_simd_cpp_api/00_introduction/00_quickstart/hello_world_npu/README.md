# HelloWorld样例

## 概述

本样例通过使用<<<>>>内核调用符来完成样例核函数在NPU侧运行验证的基础流程，核函数内通过printf打印输出结果。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── hello_world_npu
│   ├── CMakeLists.txt      // 编译工程文件
│   └── hello_world.asc     // Ascend C样例实现 & 调用样例
```

## 编译运行

在本样例根目录下执行如下步骤，编译并执行样例。
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
  mkdir -p build && cd build;
  cmake -DNPU_ARCH=dav-2201 ..;make -j; # 默认npu模式
  ./demo
  ```

  使用 CPU调试 或 NPU仿真 模式时，添加 `-DRUN_MODE=cpu` 或 `-DRUN_MODE=sim` 参数即可。

  示例如下：
  ```bash
  cmake -DRUN_MODE=cpu -DNPU_ARCH=dav-2201 ..;make -j; # cpu调试模式
  cmake -DRUN_MODE=sim -DNPU_ARCH=dav-2201 ..;make -j; # NPU仿真模式
  ```

  若需详细了解CPU调试相关内容，请参考[03_cpudebug样例](../../../01_utilities/03_cpudebug)。

  > **注意：** 切换编译模式前需清理 cmake 缓存，可在 build 目录下执行 `rm CMakeCache.txt` 后重新 cmake。

- 编译选项说明

  | 选项 | 可选值 | 说明 |
  |------|--------|------|
  | `RUN_MODE` | `npu`（默认）、`cpu`、`sim` | 运行模式：NPU 运行、CPU调试、NPU仿真 |
  | `NPU_ARCH` | `dav-2201`（默认）、`dav-3510` | NPU 架构，dav-2201 对应 Atlas A2/A3 系列，dav-3510 对应 Ascend 950PR/Ascend 950DT |

- 执行结果

  执行结果如下，说明执行成功。
  ```bash
  [Block (0/8)]: Hello World!!!
  [Block (1/8)]: Hello World!!!
  [Block (2/8)]: Hello World!!!
  [Block (3/8)]: Hello World!!!
  [Block (4/8)]: Hello World!!!
  [Block (5/8)]: Hello World!!!
  [Block (6/8)]: Hello World!!!
  [Block (7/8)]: Hello World!!!
  ```