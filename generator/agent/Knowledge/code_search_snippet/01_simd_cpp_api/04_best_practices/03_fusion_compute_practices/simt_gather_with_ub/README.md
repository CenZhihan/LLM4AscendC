# SIMD与SIMT混合编程使用UB提高内存访问效率

## 概述

样例基于SIMD与SIMT混合编程方式实现gather算子，从长度为8192的一维向量中获取指定索引的65536个数据，将输入数据预先搬运到UB，展示SIMD与SIMT混合编程模式下使用UB提升离散内存访问效率的性能优化方式。

## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构

```
├── 15_simt_gather_with_ub
│   ├── CMakeLists.txt         # cmake编译文件
│   ├── gather_v1.asc          # Ascend C直接访问GM实现gather算子样例
│   ├── gather_v2.asc          # Ascend C使用UB实现gather算子样例
|   └── README.md
```

## 算子描述

- 算子功能：  
  gather算子实现了从长度为8192的一维向量中获取指定索引的65536个数据的功能。算子输出output第i个数据计算公式为：
  
  ```
  output[i] = input[index[i]]
  ```

- 算子规格：  
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">gather</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">input</td><td align="center">8192</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">index</td><td align="center">65536</td><td align="center">uint32_t</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">output</td><td align="center">65536</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">gather_kernel</td></tr>
  </table>

- 线程层次结构：
  * 线程块数：64
  * 单个线程块中线程数：1024

- 算子实现：  
  simt_gather负责从输入input中获取指定索引的数据。首先计算线程应处理数据的索引，然后通过赋值操作将数据存储到Global Memory上。  
  在v1版本中，SIMT核函数直接从Global Memory读取数据；在v2版本中，预先将input数据搬运到UB上，SIMT核函数从UB中读取input，完成gather操作。
  ```
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  ...
  uint32_t gather_idx = index[idx];
  ...
  output[idx] = input[gather_idx];
  ```

- 性能收益：  
  使用msprof工具采集上板性能数据，v1版本算子平均运行时间约为4.56us，v2版本算子平均运行时间约为3.57us，性能提升约21.71%。

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
  v1版本执行方式：
  ```bash
  mkdir -p build && cd build;              # 创建并进入build目录
  cmake -DGATHER_VERSION=v1 ..; make -j;   # 编译工程
  ./gather                                 # 执行样例
  ```
  v2版本执行方式：
  ```bash
  mkdir -p build && cd build;              # 创建并进入build目录
  cmake -DGATHER_VERSION=v2 ..; make -j;   # 编译工程
  ./gather                                 # 执行样例
  ```
  执行结果如下，说明精度对比成功。
  ```
  [Success] Case accuracy is verification passed.
  ```

## 性能调优

算子调优工具支持上板调优和仿真调优两种模式，可分别获取算子在实际硬件环境和仿真环境的性能数据，用于定位性能瓶颈、优化算子实现。

### 上板调优

基于编译生成的可执行文件，直接在NPU硬件上采集算子性能数据，包含算子基本信息和内存负载分析等。

**操作步骤**

**1.执行调优命令**

基于编译得到的gather文件，运行算子调优工具。
```bash
msprof op ./gather
```

**2.查看性能数据**

在当前目录下会生成OPPPROF_为前缀的文件夹，目录结构及文件说明如下：
```bash 
OPPROF_202xxxxx_XXXXXX
├── dump                             # 原始性能数据（无需关注）
├── OpBasicInfo.csv                  # 算子基础数据
├── ArithmeticUtilization.csv        # cube及vector类型指令的cycle占比数据
├──  ResourceConflictRatio.csv       # 资源冲突占比数据
├── ... (开启的aic-metrics)
└──  visualize_data.bin              # 算子可视化文件（可通过MindStudio Insight加载，直观查看算子性能）
```
可通过MindStudio Insight工具打开`visualize_data.bin`文件可视化查看性能数据。

### 仿真调优

在无NPU硬件环境的场景下，通过编译仿真算子可执行文件，结合仿真器获取算子仿真性能数据，包含指令流水图等。

**操作步骤**

**1.仿真算子编译**

```bash
mkdir -p build && cd build
# 替换${SOC_VERSION}为实际NPU型号，可通过npu-smi info命令进行查询，如Ascend950PR_9599。
# 替换${GATHER_VERSION}为算子版本，值为v1或v2。
cmake -DGATHER_VERSION=${GATHER_VERSION} -DRUN_MODE=sim -DSOC_VERSION=${SOC_VERSION} ..
make -j
```

**2.配置运行时依赖**

添加运行时依赖库路径（需替换{SOC_VERSION}为实际NPU型号）：
```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib/:$LD_LIBRARY_PATH
```

**3.执行仿真调优命令**

```bash
msprof op simulator ./gather
```

**4.查看仿真性能数据**

当前目录下会生成OPPROF_前缀的文件夹，目录结构如下：
```bash
OPPROF_202xxxxx_XXXXXX
├── dump                                    # 原始性能数据，无需关注
└── simulation                              # 仿真性能数据分析结果
    ├── core0.veccore0                      # 算子block级子核
        ├── core0.veccore1_code_exe.csv     # 代码行耗时
        ├── core0.veccore1_instr_exe.csv    # 程序代码指令详细信息
        └── trace.json                      # 算子block级子核流水图
    ├── ...
    ├── visualize_data.bin                  # 算子可视化文件（可通过MindStudio Insight加载，直观查看算子性能）
    └── trace.json                          # 算子所有核的流水图
```
可通过MindStudio Insight工具打开`visualize_data.bin`文件可视化查看性能数据。

**补充说明**
更多性能指标的详细说明及调优方案，可参考《算子开发工具》手册。