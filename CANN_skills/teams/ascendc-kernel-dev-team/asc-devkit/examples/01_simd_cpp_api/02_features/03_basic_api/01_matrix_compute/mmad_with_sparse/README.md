# mmad_with_sparse样例
## 概述
本样例介绍基础API MmadWithSparse调用样例。

## 支持的产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍
```
├── mmad_with_sparse
│   ├── scripts
│   │   ├── gen_data.py             // 输入数据和真值数据生成脚本
│   │   └── verify_result.py        // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt              // 编译工程文件
│   ├── data_utils.h                // 数据读入写出函数
│   └── mmad_with_sparse.asc        // Ascend C算子实现 & 调用样例
```

## 算子描述
- 算子功能：
  本样例中实现的是[m, n, k]固定为[128, 128, 64]的稀疏矩阵乘算子，使用Ascend C基础Api MmadWithSparse接口实现。4选2 sparseMatmul是一种特殊的矩阵乘，要求一个连续的4个权重或激活值的组中，最多只有2个值为非零，其余2个强制为零，算子的数学表达式为：
  ```
  C = A * B
  ```
  其中矩阵B为稠密化后的矩阵，原始B矩阵每4个元素中至少包含2个零元素，通过4选2稠密化策略进行压缩存储，并且B矩阵必须转置，即为[N,K]输入。A、B矩阵数据类型只支持int8_t，index矩阵矩阵数据类型为uint8_t
- 算子规格：
  <table>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">128 * 128</td><td align="center">int8</td><td align="center">NZ</td></tr>
  <tr><td align="center">b</td><td align="center">64 * 128</td><td align="center">int8</td><td align="center">NZ</td></tr>
  <tr><td align="center">idx</td><td align="center">128 * 8</td><td align="center">uint8</td><td align="center">ZN</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">128 * 128</td><td align="center">int32</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">mmad_with_sparse_custom</td></tr>
  </table>

- 算子实现：
  算子实现流程分为以下步骤：
  - **CopyIn**: 将Global Memory上的输入数据搬运到Local Memory L1中，并把加载索引矩阵到L1中，其中Index分型与B矩阵分型必须为Zn分型，即GM上的B矩阵必须为转置，Index离线生成必须为Zn排布
  - **SplitB**: 使用LoadDataWithSparse将B矩阵和索引矩阵从L1搬运到L0B与内置索引buffer中
  - **SplitA**: 将A矩阵从L1搬运到L0A中
  - **Compute**: 使用MmadWithSparse完成稀疏矩阵乘运算，计算结果存储在Local Memory L0C中
  - **CopyOut**: 将输出数据从L0C搬运至Global Memory上的输出GM中

## 编译运行
在本样例根目录下执行如下步骤，编译并执行算子。

### 1. 配置环境变量  
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

### 2. 样例执行
```bash
mkdir -p build && cd build;      # 创建并进入build目录
cmake ..;make -j;                # 编译工程
python3 ../scripts/gen_data.py   # 生成测试输入数据
./demo                           # 执行编译生成的可执行程序，执行样例
python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
```

### 3. 执行结果
执行成功后，输出如下信息，说明精度对比成功。
```bash
test pass!
```

## 数据生成与验证说明

### gen_data.py 脚本功能
1. **构造稀疏矩阵B**: 生成指定形状的稀疏矩阵，每行的每4个元素块中至少包含2个零元素
2. **稠密化处理**: 将稀疏矩阵B通过4选2策略稠密化，生成稠密矩阵dense_B
3. **索引矩阵生成**: 
   - 生成index_matrix：记录每个块中被选中元素的相对位置（用于NPU计算）
   - 生成index_mask_matrix：记录被选中元素的绝对索引（用于golden计算）
4. **生成golden数据**: 使用稠密化后的矩阵和索引矩阵计算稀疏矩阵乘的真值
5. **数据格式转换**: 
   - 将索引矩阵从uint8转换为uint2格式
   - 对输入矩阵进行ND到NZ的分格转置
