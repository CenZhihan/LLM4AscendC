# Addn算子直调样例

## 概述

本样例演示基于动态Tensor编程模型的AddN算子实现，该实现采用ListTensorDesc结构处理多输入参数，结合TQue内存管理机制实现数据搬运与计算任务的协同调度，特别适用于以下场景：  
1.多输入参数动态处理：支持模型中多个输入张量的动态组合运算（如多分支网络结构）。  
2.内存流水线优化：通过TQue队列管理实现数据搬运与计算的流水线并行，降低内存访问延迟。  
3.多核并行计算：适配AI处理器的多核架构，支持大规模张量运算的高效分发。  

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── add_dynamic        
│   ├── CMakeLists.txt          // 编译工程文件
│   └── addn.asc                // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  

  AddN算子实现了两个数据相加，返回相加结果的功能，其中核函数的输入参数为动态输入，动态输入参数包含两个入参，x和y。对应的数学表达式为：  
  ```
  z = x + y
  ```
- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">AddN</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x（动态输入参数srcList[0]）</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y（动态输入参数srcList[1]）</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">addn_custom</td></tr>
  </table>
- 算子实现：  

  动态输入特性是指，核函数的入参采用ListTensorDesc的结构存储输入数据信息。  
  构造TensorList数据结构，示例如下。
  ```cpp
  constexpr uint32_t SHAPE_DIM = 2;
    struct TensorDesc {
      uint32_t dim{SHAPE_DIM};
      uint32_t index;
      uint64_t shape[SHAPE_DIM] = {8, 2048};
    };

  constexpr uint32_t TENSOR_DESC_NUM = 2;
    struct ListTensorDesc {
      uint64_t ptrOffset;
      TensorDesc tensorDesc[TENSOR_DESC_NUM];
      uintptr_t dataPtr[TENSOR_DESC_NUM];
    } inputDesc;
  ```
  将申请分配的Tensor入参组合成ListTensorDesc的数据结构，示例如下。
  ```cpp
  inputDesc = {(1 + (1 + SHAPE_DIM) * TENSOR_DESC_NUM) * sizeof(uint64_t),
              {xDesc, yDesc},
              {(uintptr_t)xDevice, (uintptr_t)yDevice}};
  ``` 
  按照传入的数据格式，解析出对应的各入参，示例如下。

  ```cpp
  uint64_t buf[SHAPE_DIM] = {0};
  AscendC::TensorDesc<int32_t> tensorDesc;
  tensorDesc.SetShapeAddr(buf);
  listTensorDesc.GetDesc(tensorDesc, 0);
  uint64_t totalLength = tensorDesc.GetShape(0) * tensorDesc.GetShape(1);
  __gm__ uint8_t *x = listTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
  __gm__ uint8_t *y = listTensorDesc.GetDataPtr<__gm__ uint8_t>(1);
  ```
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
  ./demo                       # 执行编译生成的可执行程序，执行样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  [Success] Case accuracy is verification passed.
  ```