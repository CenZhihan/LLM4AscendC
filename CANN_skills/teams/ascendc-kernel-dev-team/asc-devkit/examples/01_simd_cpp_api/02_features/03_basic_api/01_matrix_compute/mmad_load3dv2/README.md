
# 矩阵乘法中LoadData3DV2数据搬运示例

## 概述

本样例介绍LoadData3DV2指令在矩阵乘法中的使用场景和方法。LoadData3DV2可以将二维的A、B矩阵从L1搬运到L0A/L0B，其中 A 和 B 分别表示矩阵乘法的左右输入矩阵。LoadData3DV2指令参数配置及执行指令前后各个矩阵数据排布变化，均配合示意图进行了说明。

## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── mmad_load3dv2
│   ├── scripts
│   │   ├── gen_data.py             // 输入数据和真值数据生成脚本
│   │   └── verify_result.py        // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt              // 编译工程文件
│   ├── data_utils.h                // 数据读入写出函数
│   └── mmad_load3dv2.asc                    // Ascend C算子实现 & 调用样例
```

## 算子描述

LoadData3DV2指令以下简称load3dv2，该指令对于二维矩阵的转置能力和支持的数据类型都与目的地址所处的存储位置有关，具体来说：

（1）目的地址位于L0A上时，支持数据类型为：uint8_t/int8_t/half/bfloat16_t/uint32_t/int32_t/float/int4b_t；
    目的地址位于L0B上，支持数据类型为：half/bfloat16_t/uint32_t/int32_t/float。

（2）目的地址位于L0A上时，enTranspose能够决定是否启用转置功能；
    目的地址位于L0B上，默认启用转置功能（enTranspose=false时，依然会启用转置功能）。

由于本样例暂不支持输入数据类型为int4b_t，因此本样例展示了以下五种load3dv2在矩阵乘法的使用：
    <title>Load3Dv2接口场景对照表</title>
    <style>
        /* 基础表格样式，保证美观且易读 */
        table {
            border-collapse: collapse; /* 合并边框 */
            width: 80%; /* 表格宽度，可根据需要调整 */
            margin: 20px auto; /* 居中显示 */
            font-family: Arial, sans-serif; /* 通用字体 */
        }
        th, td {
            border: 1px solid #333; /* 边框样式 */
            padding: 12px; /* 单元格内边距 */
            text-align: center; /* 文字居中 */
        }
        th {
            background-color: #f0f0f0; /* 表头背景色 */
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9; /* 偶数行背景色，提升可读性 */
        }
    </style>
</head>
<body>
    <h3>Load3Dv2接口场景对照表</h3>
    <table>
        <thead>
            <tr>
                <th>scenarioNum</th>
                <th>输入数据类型</th>
                <th>A矩阵转置</th>
                <th>B矩阵转置</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>1</td>
                <td>half</td>
                <td>不转置</td>
                <td>不转置</td>
                <td>-</td>
            </tr>
            <tr>
                <td>2</td>
                <td>half</td>
                <td>转置</td>
                <td>不转置</td>
                <td>-</td>
            </tr>
            <tr>
                <td>3</td>
                <td>float</td>
                <td>不转置</td>
                <td>不转置</td>
                <td>-</td>
            </tr>
            <tr>
                <td>4</td>
                <td>float</td>
                <td>转置</td>
                <td>不转置</td>
                <td>-</td>
            </tr>
            <tr>
                <td>5</td>
                <td>int8_t</td>
                <td>不转置</td>
                <td>转置</td>
                <td>-</td>
            </tr>
        </tbody>
    </table>
    <p style="text-align: center; color: #666; font-size: 14px;">
        注：输入数据类型为b8时且目的地址位于L0B时，不支持load3dv2指令，因此当scenarioNum=5时，SplitB中调用的是load2d指令。
    </p>

  本样例中scenarioNum=3和4分别与样例[mmad_s8_f16_f32_with_A_B_transpose_option](./mmad_s8_f16_f32_with_A_B_transpose_option/README.md)中scenarioNum=10和12场景一致，因此load3dv2指令具体的参数配置和示意图可以参考该样例readme的2.3.1和3.3.1两个小节。

  由于输入数据类型不同，对于load3dv2指令配置参数的影响不大，因此本样例中其余场景可以参考scenarioNum=3和4。

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
  SCENARIO=4 M=30 K=40 N=70
  mkdir -p build && cd build;      # 创建并进入build目录
  cmake .. -DSCENARIO_NUM=$SCENARIO -DM_SIZE=$M -DK_SIZE=$K -DN_SIZE=$N;make -j;    # 编译工程
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO -m=$M -k=$K -n=$N   # 生成测试输入数据
  ./demo                           # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py -scenarioNum=$SCENARIO output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```