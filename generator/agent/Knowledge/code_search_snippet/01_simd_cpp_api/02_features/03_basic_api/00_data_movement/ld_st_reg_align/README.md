# ld_st_reg_align样例

## 概述
本样例基于RegBase编程范式实现UB(Unified Buffer)对RegTensor(Reg矢量计算基本单元)的连续和非连续的对齐数据搬运操作，该样例使用LoadAlign，StoreAlign接口，以及POST_MODE_UPDATE和DATA_BLOCK_COPY模式的使能。

## 支持的产品
- Ascend 950PR/Ascend 950DT

## 目录结构介绍
```
├── ld_st_reg_align
│   ├── CMakeLists.txt                      // 编译工程文件
│   └── README.md                           // 样例介绍
│   └── ld_st_reg_align.asc                 // AscendC样例实现 & 调用样例
```

## 样例描述
- 样例功能：
  输入一个数据类型为float，数据量为1024的向量，在源操作数和目的操作数地址均32B对齐的情况下，搬运1024个数据。
- 样例规格：
  <table>
  <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="3" align="center">AIV样例</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 1024]</td><td align="center">float</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">样例输出</td><td align="center">y</td><td align="center">[1, 1024]</td><td align="center">float</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">ld_st_reg_align_kernel</td></tr>
  </table>
- 样例实现：  
  实现以下搬运模式：
   
   <table>
  <tr>
    <td align="center">mode</td>
    <td align="center">调用函数</td>
    <td align="center">搬运方式</td>
    <td align="center">Post Mode</td>
  </tr>
  <tr>
    <td align="center">0</td>
    <td align="center">CopySucceVF</td>
    <td align="center">连续搬运</td>
    <td align="center">关闭</td>
  </tr>
  <tr>
    <td align="center">1</td>
    <td align="center">CopySucceWithPostModeVF</td>
    <td align="center">连续搬运</td>
    <td align="center">开启</td>
  </tr>
  <tr>
    <td align="center">2</td>
    <td align="center">CopyUnSucceVF</td>
    <td align="center">非连续搬运</td>
    <td align="center">关闭</td>
  </tr>
  <tr>
    <td align="center">3</td>
    <td align="center">CopyUnSucceWithPostModeVF</td>
    <td align="center">非连续搬运</td>
    <td align="center">开启</td>
  </tr>
</table>

  
  - 调用实现  
    使用内核调用符<<<>>>调用核函数。
    
## 编译运行
在本样例根目录下执行如下步骤，编译并执行样例。
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
  cmake -DNPU_ARCH=dav-3510 ..;make -j;                                     # 编译工程
  ./demo                                                                    # 执行编译生成的可执行程序，执行样例
  ```

  使用 CPU调试 或 NPU仿真 模式时，添加 `-DRUN_MODE=cpu` 或 `-DRUN_MODE=sim` 参数即可。

  示例如下：
  ```bash
  cmake -DRUN_MODE=cpu -DNPU_ARCH=dav-3510 ..;make -j; # cpu调试模式
  cmake -DRUN_MODE=sim -DNPU_ARCH=dav-3510 ..;make -j; # NPU仿真模式
  ```

  > **注意：** 切换编译模式前需清理 cmake 缓存，可在 build 目录下执行 `rm CMakeCache.txt` 后重新 cmake。

- 编译选项说明

  | 选项 | 可选值 | 说明 |
  |------|--------|------|
  | `RUN_MODE` | `npu`（默认）、`cpu`、`sim` | 运行模式：NPU 运行、CPU调试、NPU仿真 |
  | `NPU_ARCH` | `dav-3510`（默认） | NPU 架构：dav-3510 对应 Ascend 950PR/Ascend 950DT |

- 执行结果

  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```