# Region Proposal排序流水线示例

## 概述

本样例基于ProposalConcat、RpSort16、MrgSort4和ProposalExtract四个接口实现完整排序流水线。首先通过ProposalConcat将连续score值合入Region Proposal格式，再经RpSort16对每组16个Region Proposal按score域降序排列，然后通过MrgSort4将4组已排序的Region Proposal合并为1组，最后通过ProposalExtract从合并后的Region Proposal中提取score域，得到全局降序排列的连续score值。

## 支持的产品

- Atlas 推理系列产品AI Core

## 目录结构介绍

```
├── region_proposal_sort
│   ├── scripts
│   │   ├── gen_data.py                    // 输入数据和真值数据生成脚本
│   │   └── verify_result.py              // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt                    // 编译工程文件
│   ├── data_utils.h                      // 数据读入写出函数
│   └── region_proposal_sort.asc          // Ascend C样例实现 & 调用样例
```

## 样例描述

- 样例功能：
  将64个score值通过完整流水线排序后输出。一个Region Proposal由8个half元素组成，其中第5个元素（offset=4）为score域。

- 样例规格：

<table border="2">
<caption>表1：Region Proposal排序样例规格</caption>
<tr><td rowspan="2" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[1, 64]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">样例输出</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">y</td><td align="center">[1, 64]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">region_proposal_sort_custom</td></tr>
</table>

- 样例实现：
  本样例实现的是Region Proposal排序完整流水线：
  1. **ProposalConcat**：将连续的16个score值合入16个Region Proposal的score域位置（offset=4），共4组，每组1次repeat
  2. **RpSort16**：对每组16个Region Proposal按score域从大到小排序，共4组
  3. **MrgSort4**：将4组已排序的Region Proposal合并为1组，结果按score降序排列
  4. **ProposalExtract**：从合并后的64个Region Proposal中提取score域，得到64个降序排列的连续score值

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
  mkdir -p build && cd build;      # 创建并进入build目录
  cmake ..;make -j;             # 编译工程
  python3 ../scripts/gen_data.py   # 生成测试输入数据
  ./demo                           # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py ./output/output.bin ./output/golden.bin  # 验证输出结果是否正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```
