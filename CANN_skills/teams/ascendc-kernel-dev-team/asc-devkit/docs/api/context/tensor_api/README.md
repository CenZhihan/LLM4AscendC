# Tensor API

Tensor API是一套基于Ascend C的C++模板抽象库，专为定义和操作层次化多维数据布局而设计。Tensor API提供**Layout**和**Tensor**对象，将数据类型、形状、存储空间和内存布局紧凑地封装在一起，并为用户自动处理复杂的索引计算，让开发者能以直观、逻辑化的方式访问数据，而无需手动计算底层地址。

Tensor API的核心抽象是Layout（层次化多维布局），它可以与数据指针组合来表示张量。这种布局表达方式足够强大，能够表示计算任务中所需的各种数据排列。在将复杂的硬件细节封装在底层的同时，通过模板参数暴露足够的控制点，让高级用户能够对硬件行为进行精细调优。基于Layout所描述的数据组织方式，Atom（原子操作）​ 进一步封装了底层硬件指令，它将操作类型和特征参数组合在一起，成为执行最小单元的计算或数据搬运操作的基础。

Tensor API当前仅支持Ascend 950PR/Ascend 950DT。

## Tensor API架构

<img class="eddx" id="image292993463310" src="figure/tensor_api_arch.png">

### 接口调用

调用上层接口，组装Atom，传入待搬运或计算的张量：

- **Copy**：通用数据搬运算法
- **Mad**：矩阵乘加算法

### Atom抽象

将操作类型和特征参数组合，映射到昇腾处理器的硬件指令：

- **Atom**：原子操作对象
  - `CopyAtom`：数据搬运原子操作；
  - `MmadAtom`：矩阵乘加原子操作。
- **Traits**：特征配置，包含Operation和自定义Trait
  - `CopyTraits`：数据搬运特征配置；
  - `MmadTraits`：矩阵乘加特征配置。
- **Operation**：操作类型，指定数据通路
  - 数据搬运：`CopyGM2L1`、`CopyL12L0`、`CopyL0C2Out`等；
  - 矩阵计算：`MmadOperation`等。

### Arch层接口

封装硬件指令的底层接口，对参数进行处理后调用硬件指令，用户也可以直接调用：

- **DataCopy**: GM到L1 Buffer，L1 Buffer到BiasTable Buffer/Fixpipe Buffer的数据搬运指令；
- **LoadData**: L1 Buffer到L0A Buffer/L0B Buffer的数据加载指令；
- **Fixpipe**: L0C Buffer到GM/UB的数据输出指令；
- **Mmad**: 矩阵乘加计算指令。

## 参考资源

### 文档

#### 数据结构

- [Layout](Layout和层次化表述法.md)：张量的内存排布描述，由Shape（形状）和Stride（步长）组成。支持层次化表达，可描述多重分形格式。
- [Tensor](struct/tensor/LocalTensor.md)：包含指向张量位置的指针，也包含一个用于访问其元素的Layout。

#### 接口

- [Copy](data_move/Copy.md)：数据搬运算法，封装多级存储间的数据传输。包括DataCopy（GM→L1）、LoadData（L1→L0A/L0B）、FixPipe（L0C→OUT）等通路。
- [Mad](cube_compute/Mad.md)：矩阵乘加算法，封装MMAD指令。通过Traits配置数据类型、计算块大小等参数。

完整数据结构和接口列表请参考[Tensor API列表](tensor_api_list.md)。

### 测试和样例

本仓库提供[测试用例](../../../../tests/api/tensor_api) 以及[示例代码](../../../../examples/01_simd_cpp_api/02_features/05_tensor_api)，方便快速了解功能用法并进行验证。
