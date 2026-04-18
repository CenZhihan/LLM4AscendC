# Stride

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

Stride用于定义张量的步长，描述张量在各维度上相邻元素在内存中的间隔，间隔的单位为元素。

## 类型定义

Stride是一个模板别名，用于表示张量的步长：

```cpp
template <typename... Strides>
using Stride = Std::tuple<Strides...>;
```

其中：
- `Strides...`是可变参数模板，表示各维度的步长
- 实际类型为`Std::tuple<Strides...>`

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|----------|------|
| Strides... | 输入 | 各维度的步长，类型为size_t等整数类型或者Std::Int类型。 |

## API映射关系

Stride通常通过[MakeStride](./layout/MakeStride.md)函数创建。

## 约束说明

- Stride的维度数量必须与对应的Shape维度数量一致。
- 各维度的步长值必须为正整数。
- 支持的数据类型包括：size_t、int等整数类型或者Std::Int类型。

## 调用示例

```cpp
// 使用整数类型创建一个3维张量的Stride
auto stride = AscendC::Te::MakeStride(1, 100, 200);

// 获取各维度的步长
auto stride0 = AscendC::Std::get<0>(stride); // stride0 = 1
auto stride1 = AscendC::Std::get<1>(stride); // stride1 = 100
auto stride2 = AscendC::Std::get<2>(stride); // stride2 = 200

// 使用Std::Int创建一个3维张量的Stride
auto strideInt = AscendC::Te::MakeStride(AscendC::Std::Int<1>{}, AscendC::Std::Int<100>{}, AscendC::Std::Int<200>{});

// 获取各维度的步长
auto strideInt0 = AscendC::Std::get<0>(strideInt); // strideInt0 = 1
auto strideInt1 = AscendC::Std::get<1>(strideInt); // strideInt1 = 100
auto strideInt2 = AscendC::Std::get<2>(strideInt); // strideInt2 = 200
```