# int4b_t数据类型下Matmul兼容性样例
## 概述
int4b_t矩阵乘计算的端到端用例。
## 支持的产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
## 目录结构介绍
```
├── matmul_s4_910B
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── matmul_s4.asc    // AscendC算子实现 & 调用样例
```
## 算子描述
- 算子功能：  
  实现L0A Buffer和L0B Buffer初始化值的端到端用例
- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="7" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">16 * 32</td><td align="center">int4b_t</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">32 * 64</td><td align="center">int4b_t</td><td align="center">ND</td></tr>
  </tr>
    <tr><td align="center">bias</td><td align="center">64</td><td align="center">int32_t</td><td align="center">ND</td></tr>
  </tr>
    <tr><td align="center">tiling</td><td align="center">64</td><td align="center">int32_t</td><td align="center">ND</td></tr>
  </tr>
    <tr><td align="center">quant</td><td align="center">64</td><td align="center">uint64_t</td><td align="center">ND</td></tr>
  </tr>
    </tr>
    <tr><td align="center">workspace</td><td align="center">16777</td><td align="center">uint64_t</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">16 * 64</td><td align="center">int32_t</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">matmul_s4</td></tr>
  </table>
- 算子实现：  
  实现int4b_t数据类型的矩阵计算。

  - 调用实现  
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
  mkdir -p build && cd build;                                               # 创建并进入build目录
  cmake ..;make -j;                                                         # 编译工程
  python3 ../scripts/gen_data.py                                            # 生成测试输入数据
  ./demo                                                                    # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```