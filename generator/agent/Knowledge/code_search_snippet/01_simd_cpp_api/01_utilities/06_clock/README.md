# 基于gather样例的SIMT时间戳打点功能实现样例

## 概述

本样例基于gather计算，演示在```__simt_vf__```核函数中使用```clock()```接口实现时间戳打点方法，用于记录样例执行前后的时间戳并统计执行耗时。

## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构

```
├── 04_clock
│   ├── CMakeLists.txt         # cmake编译文件
│   ├── gather.asc             # Ascend C样例实现gather调用样例
|   └── README.md
```

## 样例描述

- 样例功能:

  gather计算可参考[gather样例详情](../../02_features/03_simt/simt_gather/README.md)章节。

## 编译运行

在本样例根目录下执行如下步骤，编译并执行样例。

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
