# type_traits类型特性直调样例
## 概述
本样例通过Ascend C编程语言实现了type_traits类型特性用例，类型特性接口包括is_void/remove_const/add_const等。使用<<<>>>内核调用符来完成算子核函数在NPU侧运行验证的基础流程，给出了对应的端到端实现。

## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构
```
├── type_traits
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── type_traits.asc              // Ascend C算子实现 & 调用样例
```

## 功能描述
- type_traits类型特性接口，接口包括is_void/is_integral/is_floating_point/is_array/is_pointer/is_reference/is_const/remove_const/remove_volatile/remove_cv/remove_reference/remove_pointer/add_const/add_volatile/add_cv/add_pointer/add_lvalue_reference/add_rvalue_reference，可用于在程序编译时对类型的判断、增加、删除、转换等处理。

- 算子实现：  
  - kernel实现   
    Ascend C调用各接口，打印接口返回值和ascendc_assert判断。   

  - 调用实现  
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
  cmake ..;make -j;             # 编译工程
  ./demo                        # 执行编译生成的可执行程序，执行样例
  ```
  最终执行结果如下，说明执行成功。
  ```bash
  test pass!
  ```