# move_successive_align样例

## 概述
本样例演示了SIMD场景下，基于RegBase编程范式的连续对齐搬运算子的核函数直调方法，算子支持单核运行。

## 支持的产品
- Ascend 950PR/Ascend 950DT

## 目录结构介绍
```
├── move_successive_align
│   ├── CMakeLists.txt                            // 编译工程文件
│   └── move_successive_align.asc                 // AscendC算子实现 & 调用样例
```

## 算子描述
- 算子功能：  
  在源操作数和目的操作数地址均32B对齐的情况下，搬运1024个float类型数据。
- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="3" align="center">AIV算子</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">1024</td><td align="center">float</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">1024</td><td align="center">float</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">move_successive_align_kernel</td></tr>
  </table>
- 算子实现：  
  主要包括以下步骤：
  1. 初始化参数
  2. 将数据从GM搬运到UB
  3. 当mode=0，调用CopyVF搬运1024个float数，写回UB;
     当mode=1，调用CopyWithPostModeVF，开启post mode，搬运1024个float数，写回UB
  4. 将数据从UB搬运回GM
  
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
  mkdir -p build && cd build;                                               # 创建并进入build目录
  cmake ..;make -j;                                                         # 编译工程
  ./demo                                                                    # 执行编译生成的可执行程序，执行样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```