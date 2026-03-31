# SIMT编程模式实现Add算子样例

## 概述

样例基于Ascend C SIMT编程方式实现Add算子，实现两个输入张量逐元素相加得到输出张量的功能，展示SIMT编程的基本流程。

## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构

```
├── 01_add
│   ├── add.asc             # SIMT实现add调用样例
|   └── README.md
```

## 算子描述

- 算子功能：  
  本算子实现了形状为48 * 256的两个张量x，y相加得到算子输出z。第i个元素的计算公式为：
  
  ```
  z[i] = x[i] + y[i]
  ```

- 算子规格：  
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">add</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">48 * 256</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">48 * 256</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">48 * 256</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>
  </table>

- 数据切分：  
  * 核数：48核
  * 每核线程数：256线程
  * 单线程处理：1个元素
  * 总处理能力：48×256=12288

- 算子实现：  
  算子的实现流程为从输入x（Global Memory上的指针）中获取指定索引的数据。基于上述数据切分，首先计算线程应处理数据的索引，然后通过加法运算符计算得到输出值。

- 调用实现：  
  使用内核调用符<<<>>>调用核函数。

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
  CANN_PATH=$(printenv ASCEND_HOME_PATH)  # 获取CANN包安装路径
  OUTPUT="demo"               #用户自定义编译生成的二进制文件名称

  bisheng \
  -x dpp --cce-aicore-arch=dav-c310-vec \
  -std=c++17 \
   add.asc \
  -I${CANN_PATH}/include \
  -I${CANN_PATH}/include/ascendc/host_api \
  -I${CANN_PATH}/compiler/ascendc/include/highlevel_api \
  -I${CANN_PATH}/compiler/tikcpp/tikcfw \
  -I${CANN_PATH}/compiler/tikcpp/tikcfw/lib \
  -I${CANN_PATH}/compiler/tikcpp/tikcfw/lib/matmul \
  -I${CANN_PATH}/compiler/tikcpp/tikcfw/impl \
  -I${CANN_PATH}/compiler/tikcpp/tikcfw/interface \
  -L${CANN_PATH}/lib64 \
  -lascendc_runtime \
  -lascendcl \
  -lruntime \
  -lregister \
  -lerror_manager \
  -lprofapi \
  -lascendalog \
  -lmmpa \
  -lascend_dump \
  -ltiling_api \
  -lplatform \
  -ldl \
  -lc_sec \
  -lstdc++ \
  -o ${OUTPUT}

  ./${OUTPUT}                        # 执行样例

  ```
  执行结果如下，说明精度对比成功。
  ```
  [Success] Case accuracy is verification passed.
  ```
