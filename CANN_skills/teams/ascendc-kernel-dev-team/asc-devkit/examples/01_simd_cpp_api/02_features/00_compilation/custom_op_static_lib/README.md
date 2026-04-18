# 自定义算子静态库

## 概述

本样例以`AddCustom`为例，展示如何编译、打包并链接自定义算子静态库，通过aclnn的方式执行算子。

## 支持的产品

本样例支持如下产品型号：
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── custom_op_static_lib
│   ├── app                       // 链接静态库并执行算子
│   │   ├── CMakeLists.txt
│   │   └── main.cpp
│   ├── op                        // 编译打包生成自定义算子静态库
│   │   ├── add_custom_host.cpp
│   │   ├── add_custom_kernel.cpp
│   │   ├── add_custom_tiling.h
│   │   └── CMakeLists.txt
│   └── CMakeLists.txt
```

## 代码实现介绍

本样例中`op`目录负责编译打包生成自定义算子静态库，生成的静态库路径为`./build/customize-install/lib/lib${package_name}.a`；`app`目录通过`find_package(${package_name})`引入静态库，编译`main.cpp`生成`execute_add_op`，调用`aclnnAddCustom`并校验结果。

AddCustom算子介绍等内容可参考[算子描述](../custom_op/README.md)。

## 编译运行

- 配置环境变量

  请根据当前环境上CANN开发套件包的[安装方式](../../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。需确保`ASCEND_HOME_PATH`指向CANN安装根路径，以便头文件与库路径生效。
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

  在本样例根目录下执行如下步骤，运行该样例。

  ```bash
  mkdir -p build; cd build
  cmake .. && make -j
  ./execute_add_op
  ```

  执行结果如下，说明执行成功。

  ```log
  test pass
  ```
