# SIMT与SIMD混合编程实现floor_mod算子样例

## 概述

本样例以实现floor_mod功能的算子为例，展示了SIMT和SIMD混合编程的算子开发方式。算子中基于SIMD的datacopy完成数据搬入搬出，float场景通过SIMD完成计算逻辑，而int32_t场景则是基于SIMT完成计算功能，提升整型场景性能。


## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构

```
├── 14_simt_and_simd_floor_mod
│   ├── CMakeLists.txt         # cmake编译文件
│   ├── floor_mod.asc    # Ascend C算子实现 & 调用样例
|   └── README.md
```

## 算子描述

- 算子功能：  
  输入self和other两个tensor，将self每个元素除以other的对应元素得到的余数，结果与除数other同符号，并且绝对值小于other的绝对值。算子输出output第i个数据计算公式为：

  ```
  output[i] = self[i] - floor(self[i]/other[i]) * other[i]
  ```

- 算子规格：  
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">floor_mod</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">self</td><td align="center">6400</td><td align="center">float/int32_t</td><td align="center">ND</td></tr>
  <tr><td align="center">other</td><td align="center">6400</td><td align="center">float/int32_t</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">output</td><td align="center">6400</td><td align="center">float/int32_t</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">floor_mod</td></tr>
  </table>

- 基础知识：  
  SIMD编程提供基于寄存器（Regbase）的编程API，可以直接操作Vector Core中的寄存器。单次API处理的数据量上限为寄存器大小，通过AscendC::GetVecLen函数获取该值。在计算过程中，需要多次调用微指令API完成单核数据的处理。在SIMT编程中，可以直接读取和使用Global Memory上的数据。在Vector Core中，SIMT单元和SIMD单元共享片上存储，因此可以使用片上存储完成SIMT和SIMD的混合编程。  
  本例中，float场景使用SIMD方式完成算子功能，而在整型场景，通过SIMD完成数据搬入搬出，SIMT方式实现计算功能。

- 数据切分：  
  本例中算子输入总元素个数为6400，每个核处理数据量为1024，因此核数为7。
  - SIMT场景，设置每个核实际启动的线程数THREAD_COUNT为1024，每个线程负责处理一个元素。每个线程处理的数据位置是通过threadIdx进行索引，后续按照总线程数进行步进索引。
    ```cpp
    uint32_t index = threadIdx.x;
    ```
  - SIMD场景，受限于微指令计算的Reg大小受限，单次只能处理256B数据，因此需要分多次循环处理1024个单核数据量。

- 算子实现：  
  本算子的实现流程主要分为3个步骤：CopyIn， Compute， CupyOut。CopyIn和CopyOut遵循一般的SIMD算子开发方式，这里不再赘述。
  Compute中通过数据类型判断不同的处理方式。

  floor_mod_simt负责整型场景的若干个元素的计算。
  ```cpp
    uint32_t index = threadIdx.x;
    auto rem = self[index] % other[index];
    bool signs_differ = ((rem < 0) != (other[index] < 0));
    if (signs_differ && (rem != 0)) {
        out[index] = rem + other[index];
    } else {
        out[index] = rem;
    }
  ```

  floor_mod_simd负责float场景的计算。
  ```cpp
  for (uint16_t j = 0; j < loopTimes; j++) {
        preg = AscendC::Reg::UpdateMask<T>(sregMask);
        AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_NORM>(fmodResValue, fmodResAddr + VL_T * j);
        AscendC::Reg::Compare<T, AscendC::CMPMODE::NE>(negValue, fmodResValue, zeroValue, preg);

        AscendC::Reg::And(fmodSignValue, (AscendC::Reg::RegTensor<uint32_t>&)fmodResValue, signValue, preg);
        AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_NORM>(inputX2Value, otherAddr + VL_T * j);
        AscendC::Reg::Add(addValue, fmodResValue, inputX2Value, preg);
        AscendC::Reg::And(inputX2signValue, (AscendC::Reg::RegTensor<uint32_t>&)inputX2Value, signValue, preg);
        AscendC::Reg::Compare<uint32_t, AscendC::CMPMODE::NE>(signNegValue, fmodSignValue, inputX2signValue, preg);

        AscendC::Reg::MaskAnd(resMaskValue, signNegValue, negValue, preg);
        AscendC::Reg::Select(resValue, addValue, fmodResValue, resMaskValue);
        AscendC::Reg::DataCopy<T, AscendC::Reg::StoreDist::DIST_NORM>(dstAddr + VL_T * j, resValue, preg);
    }
  ```

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
  ./demo                        # 执行样例
  ```
  执行结果如下，说明精度对比成功。
  ```
  [Success] Case accuracy is verification passed.
  ```
  当前样例代码默认执行整形场景功能，可通过修改main()函数调用`process_float()`的方法执行float场景的功能。
  ```
  int32_t main()
  {
      // process_float:  test float function
      // process_int:  test int32_t function
      return process_float();
  }
  ```