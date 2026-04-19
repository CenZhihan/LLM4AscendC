
# Mmad unitFlag特性样例

## 概述

本样例介绍是否使能unitFlag对于Mmad指令执行矩阵乘法性能的影响。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── mmad_unitflag
│   ├── scripts
│   │   ├── gen_data.py             // 输入数据和真值数据生成脚本
│   │   └── verify_result.py        // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt              // 编译工程文件
│   ├── data_utils.h                // 数据读入写出函数
│   └── mmad_unitflag_disable.asc                    // 不使能unitFlag
│   └── mmad_unitflag_enable.asc                    // 使能unitFlag
```

## 算子描述

unitFlag是一种Mmad指令和Fixpipe指令细粒度的并行，使能该功能后，硬件每计算完一个分形，计算结果就会被搬出，该功能不适用于在L0C Buffer累加的场景。取值说明如下：
0：保留值；
2：使能unitFlag，硬件执行完指令之后，不会关闭unitFlag功能；
3：使能unitFlag，硬件执行完指令之后，会将unitFlag功能关闭。
使能该功能时，Mmad指令的unitFlag在最后1个分形设置为3、其余分形计算设置为2即可。

下面以一个示例说明使能unitFlag前后。假设有A矩阵shape为[128, 512],B矩阵shape为[512, 256],执行A、B矩阵乘法时，需要沿着K轴进行迭代循环，假设每次迭代K长度为64，则需要迭代8次，此时8次Mmad对应1次Fixpipe操作。此时，前7次Mmad的unitFlag都设置为2，保证后续Mmad可以写入L0C；最后1次Mmad的unitFlag都设置为3，保证Fixpipe可以读取L0C；当Mmad使能unitFlag时，Fixpipe也必须使能unitFlag，Fixpipe的unitFlag设置为3，保证后续Mmad可以顺利写入L0C。

另外，需要注意当使能unitFlag时，L0C上的LocalTensor不能用[TQue](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0137.html)获取，需要改用[TBuf](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0161.html)。

## 性能调优

算子调优工具支持上板调优和仿真调优两种模式，可分别获取算子在实际硬件/仿真环境下的性能数据，用于定位性能瓶颈、优化算子实现。

### 上板调优

基于编译生成的可执行文件，直接在NPU硬件上采集算子性能数据，数据为算子预热后的真实运行指标。

**操作步骤**

**1.执行调优命令**

基于编译得到的demo文件，运行[算子调优工具](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/optool/atlasopdev_16_0082.html)。
```bash
msprof op ./demo
```
**2.查看性能数据**

在当前目录下会生成OPPPROF_为前缀的文件夹，目录结构及文件说明如下：
```bash 
OPPROF_2025xxxx_XXXXXX
├── dump                             # 原始性能数据（无需关注）
├── OpBasicInfo.csv                  # 算子基础数据
├── ArithmeticUtilization.csv        # cube及vector类型指令的cycle占比数据
├──  ResourceConflictRatio.csv       # 资源冲突占比数据
├── ... (开启的aic-metrics)
└──  visualize_data.bin               # 算子可视化文件（可通过MindStudio Insight加载，直观查看算子性能）
```
可直接查看.csv文件获取算子block级性能数据或通过MindStudio Insight工具打开`visualize_data.bin`文件可视化查看性能数据。

### 仿真调优

在无NPU硬件环境的场景下，通过编译仿真版算子可执行文件，结合模拟器获取算子仿真性能数据，适用于开发阶段的快速调优。

**操作步骤**

**1.仿真算子编译**

```bash
# 创建并进入build目录
mkdir -p build && cd build
# 替换${SOC_VERSION}为实际NPU型号，可通过npu-smi info命令进行查询，如Ascend910_957c。
cmake -DRUN_MODE=sim -DSOC_VERSION=${SOC_VERSION} ..
# 多线程编译
make -j$(nproc)     
```
**2.配置运行时依赖**

添加运行时依赖库路径（需替换{SOC_VERSION}为实际NPU型号）：
```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib/:$LD_LIBRARY_PATH  
```
**3.执行仿真调优命令**

```bash
msprof op simulator ./demo
```
**4.查看仿真性能数据**

当前目录下会生成OPPROF_前缀的文件夹，目录结构如下：
```bash
OPPROF_2025xxxx_XXXXXX
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
**补充说明**
更多性能指标的详细说明及调优方案，可参考《算子开发工具》手册。

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
  cmake ..;make -j;                # 编译工程，注意CMakeLists中根据是否要使能unitflag在以下两个.asc文件中切换
  python3 ../scripts/gen_data.py   # 生成测试输入数据
  ./demo                           # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```