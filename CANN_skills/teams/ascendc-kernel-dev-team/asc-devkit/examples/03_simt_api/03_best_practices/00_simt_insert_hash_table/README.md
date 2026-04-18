# SIMT实现InsertHashTable算子样例

## 概述

本样例介绍InsertHashTable算子，展示基于SIMT实现的大规模线程并发访问Global Memory内存的算子实践。

## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构

```
├── 00_simt_insert_hash_table
│   ├── CMakeLists.txt          # cmake编译文件
│   ├── insert_hash_table.asc   # Ascend C算子实现 & 调用样例
|   └── README.md
```

## 背景知识

Hash Table是一种高效的数据结构，它通过一个哈希函数将“键”（key）映射到一个固定大小的数组（桶数组）的特定位置上，从而实现快速地查找、插入和删除操作。

- 哈希函数：一个函数，它接受一个键作为输入，计算出一个整数（哈希值），该整数通过某种运算被映射为数组的一个索引。
- 桶数组：一个数组，每个元素（称为“桶”，Bucket）用于存储键值对（key-value pair）

向hash table 中插入一个键值对的基本流程是：

- 使用哈希函数计算出key的哈希值
- 将哈希值映射为桶数组的索引index（通常是对哈希表容量取余）
- 将键值对存储到数组的bucket[index]位置

哈希冲突：由于hash table的容量（桶数组大小）远小于哈希函数的输出范围，不同的key可能会被映射到同一个索引上。

常见的哈希冲突解决方法是开放寻址法：当发生冲突时，算法会按照线性探查寻找下一个“空”桶来存储数据。

由于每个键值对的存储位置由计算出的哈希值决定，这些值通常是随机且离散的，加上哈希冲突的存在，导致在实际写入数据前需要进行多个条件判断，因此，哈希表不适合基于SIMD编程模型来实现。相比之下，SIMT编程中每个线程都能独立处理分支判断，并支持离散访存，因此在实现哈希表方面更具优势。

## 算法分析

SIMT能够通过大量线程的并发执行实现高效处理大批数据，但同时也带来了两个问题：多线程写冲突和核间数据同步问题。

### 多线程写冲突问题

在多线程操作同一块内存时，资源冲突问题在所难免。当两个线程插入的key产生哈希冲突时，多个线程会尝试向桶数组的同一位置写入数据，因此需要确保仅有一个线程能够获得桶的写入权限。程序实现时，在Bucket结构体中添加一个标记位“flag”，用于标记当前桶的写入权限，线程通过原子指令asc_atomic_cas()修改flag来获取桶的写入权限。

### 核间数据同步问题

当发生哈希冲突时，线程需要判断当前桶中存储的key是否与待插入的key相同，这要求当前线程能够读取其他线程写入的数据，并确保数据的完整性。程序实现时，在Bucket结构体中添加一个标记位“state”，用于标识key值的写入状态。写线程中在写入key后将state标记位置1，并在二者之间调用asc_threadfence()接口，确保当state被设置为1时，key的写入操作已经完成。读线程中，通过while循环轮询state值，直到state被设置为1，然后再读取key值。

```C++
struct Bucket {
    int64_t key;            // 键
    uint32_t state;         // key值写入状态标记
    uint32_t flag;          // 原子操作标志位
    float value[32];        // 值
};
```

## 算子描述

- 算子功能：  
  InsertHashTable算子实现将N个键值对插入容量为Z的哈希表中，其中key为int64_t类型数，value为长度M的float类型的Tensor。

- 算子规格：  
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">insert_hash_table</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">keys</td><td align="center">1, N</td><td align="center">int64_t</td><td align="center">ND</td></tr>
  <tr><td align="center">values</td><td align="center">N, M</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">table_addr</td><td align="center">1, Z</td><td align="center">Bucket</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">insert_hash_table</td></tr>
  </table>

- 算子实现：  
  InsertHashTable算子的实现流程为，每个warp处理一个键值对，warp中线程0负责根据key的哈希值寻找可用桶，线程0-31负责将value存入桶中。线程0寻找可用桶时，通过开放寻址法解决哈希冲突问题，并使用asc_atomic_cas()接口解决多线程冲突问题，使用asc_threadfence()接口解决核间数据同步问题。

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
  mkdir -p build && cd build;   # 创建并进入build目录
  cmake ..; make -j;            # 编译工程
  ./demo                        # 执行样例
  ```
  执行结果如下，说明精度对比成功。
  ```
  [Success] find all key-value in hash table.
  ```