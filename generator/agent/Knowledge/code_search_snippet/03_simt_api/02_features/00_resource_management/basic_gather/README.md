# SIMT编程模式实现Gather算子样例

## 概述

样例基于Ascend C SIMT编程方式实现简单场景（固定shape的）Gather算子，从输入张量中采集指定的m行数据，展示简化场景离散内存访问类算子的开发方法。

## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构

```
├── basic_gather
│   ├── gather.asc             # SIMT实现gather调用样例
|   └── README.md
```

## 算子描述

- 算子功能：  
  gather算子实现了从形状为100000 * 128的二维向量中获取指定索引的12288行数据的功能。算子输出output第i行数据计算公式为：
  
  ```
  output[i] = input[index[i]]
  ```

- 算子规格：  
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">gather</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">input</td><td align="center">100000 * 128</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">index</td><td align="center">12288</td><td align="center">uint32_t</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">output</td><td align="center">12288 * 128</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">gather_custom</td></tr>
  </table>

- 数据切分：  
  * 核数：48核
  * 每核线程数：256线程
  * 单线程处理：1行（128列）
  * 总处理能力：48×256=12288行（覆盖索引长度）

- 算子实现：  
  gather算子的实现流程为从输入input（Global Memory）中获取指定索引的数据。基于上述数据切分，首先计算线程应处理数据的索引，然后通过赋值操作将一行数据存储到Global Memory上。

- 调用实现：  
  使用内核调用符<<<>>>调用核函数。

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
  cmake ..; make -j;            # 编译工程
  ./gather                      # 执行样例
  ```
  执行结果如下，说明精度对比成功。
  ```
  [Success] Case accuracy is verification passed.
  ```
