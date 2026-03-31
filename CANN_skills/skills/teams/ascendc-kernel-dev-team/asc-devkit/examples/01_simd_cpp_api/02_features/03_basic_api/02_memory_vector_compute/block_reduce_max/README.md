# BlockReduceMax样例
## 概述
本样例在归约场景下，基于[BlockReduceMax](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0082.html)实现最大值归约功能，对每个datablock内所有元素求最大值。

## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
## 目录结构介绍
```
├── block_reduce_max
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── block_reduce_max.asc      // Ascend C样例实现 & 调用样例
```

## 样例描述
- 样例功能：  
  该样例是对每个datablock内所有元素求最大值。
 - 样例规格：
   <table border="2" align="center">
   <caption>表1：BlockReduceMax样例规格</caption>
   <tr>
   <td rowspan="1" align="center">样例类型(OpType)</td>
   <td colspan="4" align="center">BlockReduceMax</td></tr>
   <tr><td rowspan="3" align="center">样例输入</td></tr>
   <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
   <tr><td align="center">x</td><td align="center">[128]</td><td align="center">half</td><td align="center">ND</td></tr>
   <tr><td rowspan="2" align="center">样例输出</td></tr>
   <tr><td align="center">y</td><td align="center">[64]</td><td align="center">half</td><td align="center">ND</td></tr>
   <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">block_reduce_max_custom</td></tr>
   </table>
- 样例实现：  	 
   本样例实现的是固定shape为输入x[128]，输出y[64]。

- Kernel实现

  计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用BlockReduceMax接口完成计算，得到最终结果，再搬出到外部存储上。

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