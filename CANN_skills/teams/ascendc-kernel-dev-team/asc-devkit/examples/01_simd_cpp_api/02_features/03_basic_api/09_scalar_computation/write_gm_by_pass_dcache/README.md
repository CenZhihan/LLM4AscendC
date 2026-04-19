# WriteGmByPassDcache样例

## 概述

本样例介绍基础api WriteGmByPassDcache的调用，该api功能：不经过DCache向GM地址上写数据。

## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构介绍

```
├── write_gm_by_pass_dcache
│   ├── CMakeLists.txt                   // 编译工程文件
│   └── write_gm_by_pass_dcache.asc      // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  WriteGmByPassDcache算子实现了不经过DCache向GM地址上写数据的功能。

- 算子规格：  
  <table>
  <tr><td rowspan="2" align="center">算子输出</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">z</td><td align="center">1024</td><td align="center">int32_t</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">kernel_write_gm_by_pass_dcache</td></tr>
  </table>

- 算子实现
  本样例中实现的是WriteGmByPassDcache算子。

  - Kernel实现  
    WriteGmByPassDcache算子将int32_t类型的标量2直接写入dstGlobal。

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
  ./demo                        # 执行编译生成的可执行程序，执行样例
  ```

  执行结果如下，说明精度对比成功。

  ```bash
  test pass!
  ```
