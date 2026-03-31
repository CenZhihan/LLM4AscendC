# PairReduceSum样例
## 概述
本样例在归约场景下，基于[PairReduceSum](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0085.html)对数据（a1, a2, a3, a4, a5, a6...）的相邻两个元素求和为（a1+a2, a3+a4, a5+a6, ......）。PairReduceSum接口适用于需要成对元素求和的场景，如信号处理、特征聚合等。
## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
## 目录结构介绍
```
├── pair_reduce_sum
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
   │   └── pair_reduce_sum.asc      // Ascend C样例实现 & 调用样例
```

## 样例描述
- 样例功能：
   PairReduceSumCustom样例对每个pair内所有元素求和。
- 样例规格：
   <table border="2" align="center">
   <caption>表1：PairReduceSum样例规格</caption>
   <tr>
   <td rowspan="1" align="center">样例类型(OpType)</td>
   <td colspan="4" align="center">PairReduceSum</td></tr>
   <tr><td rowspan="2" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
   <tr><td align="center">x</td><td align="center">[128]</td><td align="center">half</td><td align="center">ND</td></tr>
   <tr><td rowspan="2" align="center">样例输出</td></tr>
   <tr><td align="center">y</td><td align="center">[64]</td><td align="center">half</td><td align="center">ND</td></tr>

   <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">pair_reduce_sum_custom</td></tr>
   </table>

- Kernel实现

   计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用PairReduceSum基础API接口完成计算，得到最终结果，再搬出到外部存储上。


- 调用实现
   使用内核调用符<<<>>>调用核函数。

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
  cmake ..;make -j;             # 编译工程
  python3 ../scripts/gen_data.py   # 生成测试输入数据
  ./demo                        # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```