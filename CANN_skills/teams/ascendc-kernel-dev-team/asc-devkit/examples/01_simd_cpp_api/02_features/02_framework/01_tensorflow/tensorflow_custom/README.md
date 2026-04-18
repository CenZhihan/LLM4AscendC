# 自定义算子工程+TensorFlow自定义算子样例

## 概述

本样例展示如何将Ascend C自定义算子Add计算映射到TensorFlow自定义算子，并通过TensorFlow调用Ascend C算子。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── tensorflow_custom
│   ├── CMakeLists.txt
│   ├── custom_assign_add_custom.cc
│   └── run_add_custom_tf.py
```

## 代码实现介绍

完成算子工程创建后，会在算子工程目录下生成framework/tf_plugin目录，用于存放TensorFlow框架适配插件实现文件。

本样例以自定义算子Add为例，将该算子映射到TensorFlow自定义算子上，核心流程如下：
1. 通过 `tf.load_op_library` 加载自定义算子库文件，获取自定义算子接口 `add_custom`。
2. 构造输入数据，使用 `tf.compat.v1.placeholder` 定义输入张量，分别计算 `tf.math.add` 与 `add_custom` 的结果。
3. 配置 `ConfigProto`，启用 `NpuOptimizer`，并关闭重映射与内存优化，确保算子按预期执行。
4. 调用`np.allclose`对比标准TensorFlow加法算子和自定义算子的结果，验证计算正确性。

## 编译运行

  在本样例根目录下执行如下步骤，编译并执行算子。
  - 编译、打包和部署自定义算子工程

    运行此样例前，需先进入[自定义算子工程样例](../../00_compilation/custom_op/)目录完成编译、打包和部署。

    > [!NOTE]注意
    > 需适配插件代码，路径为： `examples/01_simd_cpp_api/02_features/00_compilation/custom_op/framework/tf_plugin/tensorflow_add_custom_plugin.cc`，需修改插件代码中的TensorFlow调用算子名称OriginOpType为"AddCustom"，如下所示：
    >
    > ```cc
    > REGISTER_CUSTOM_OP("AddCustom")
    >   .FrameworkType(TENSORFLOW)      // type: TENSORFLOW
    >   .OriginOpType("AddCustom")      // name in tf module
    >   .ParseParamsByOperatorFn(AutoMappingByOpFn);
    > ```

  - 安装TensorFlow插件包

    请参考[《TensorFlow 2.6.5模型迁移》](https://www.hiascend.com/document/redirect/canncommercial-tfmigr26)中的“安装框架插件包”章节，获取详细的安装指南和步骤。

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
    mkdir -p build; cd build
    cmake .. && make -j
    python3 ../run_add_custom_tf.py
    ```

    执行结果如下，说明执行成功。

    ```log
    test pass
    ```
