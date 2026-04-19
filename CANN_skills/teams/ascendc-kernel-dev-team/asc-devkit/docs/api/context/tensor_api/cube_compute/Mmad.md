# Mmad

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
| Ascend 950PR/Ascend 950DT|√|

## 功能说明

完成矩阵乘加（C += A * B或C = A * B + Bias）操作。

## 函数原型

- 不支持偏置的矩阵乘加接口
  ```cpp
  template <const MmadTrait& trait = DEFAULT_MMAD_TRAIT, typename T, typename U, typename S, typename Params>
  Mmad(const T& dst, const U& fm, const S& filter, const Params& params)
  ```

- 支持偏置的矩阵乘加接口
  ```cpp
  template <const MmadTrait& trait = DEFAULT_MMAD_TRAIT, typename T, typename U, typename S, typename V, typename Params>
  Mmad(const T& dst, const U& fm, const S& filter, const V& bias, const Params& params)
  ```

## 参数说明

**表 1** 模板参数说明

| 参数名 | 描述 |
| :--- | :--- |
| trait | 矩阵乘加特征，类型为MmadTrait：<br/>struct MmadTrait {<br/>&emsp;int32_t fmOffset = 0;<br/>&emsp;bool kDirectionAlign = false;<br/>&emsp;bool cmatrixSource = false;<br/>&emsp;bool disableGemv = false;<br/>&emsp;MmadType mmadType = MmadType::NORMAL;<br/>};<br/>默认值为`DEFAULT_MMAD_TRAIT`。 |
| T | 计算结果矩阵的数据类型。 |
| U | 左矩阵的数据类型。 |
| S | 右矩阵的数据类型。 |
| V | 偏置矩阵的数据类型。 |
| Params | 运行时参数类型。 |

**表2** MmadTrait参数说明：

|参数名|含义|
|--|--|
|fmOffset|特征图偏移。|
|kDirectionAlign|K方向是否需要对齐。|
|cmatrixSource|配置C矩阵初始值是否来源于BiasTable Buffer。|
|disableGemv|是否禁用GEMV优化。|
|mmadType|MmadType是一个枚举类型，用于标识矩阵乘的类型。默认值为MmadType::NORMAL，若涉及MX数据类型计算，需要设置为MmadType::MX。|

**表3** 参数说明

| 参数名 | 说明 |
|--------|------|
| dst | 计算结果矩阵C，通过MakeTensor构造的[LocalTensor类型](../struct/tensor/LocalTensor.md)，存储位置支持L0C，数据格式仅支持NZ。 |
| fm | 左矩阵A，通过MakeTensor构造的LocalTensor类型，存储位置支持L0A，数据格式仅支持NZ。 |
| filter |  右矩阵B，通过MakeTensor构造的LocalTensor类型，存储位置支持L0B，数据格式仅支持ZN。 |
| bias |  可选，偏置向量，通过MakeTensor构造的LocalTensor类型，存储位置支持L0C和BiasTable Buffer，数据格式仅支持ND。带偏置时传入。 |
| params | MmadParams，矩阵乘加参数结构体。 |

**表4** MmadParams参数说明

| 参数名  | 说明 |
|------|------|
| m | 左矩阵Height，取值范围：m∈[0, 4095] 。默认值为0。 |
| n | 右矩阵Width，取值范围：n∈[0, 4095] 。默认值为0。 |
| k | 左矩阵Width、右矩阵Height，取值范围：k∈[0, 4095] 。默认值为0。 |
| unitFlag | unitFlag是一种Mmad指令和Fixpipe指令细粒度的并行，使能该功能后，硬件每计算完一个分形，计算结果就会被搬出，该功能不适用于在L0C Buffer累加的场景。取值说明如下：<br>0：保留值；<br>2：使能unitFlag，硬件执行完指令之后，不会关闭unitFlag功能；<br>3：使能unitFlag，硬件执行完指令之后，会将unitFlag功能关闭。|
| cmatrixInitVal | 配置C矩阵初始值是否为0。默认值false。<br>true: C矩阵初始值为0；<br>false: C矩阵初始值通过模板参数MmadTrait中的cmatrixSource参数进行配置。|

## 数据类型支持

**表 5**  dst、fm、filter、bias支持的精度类型组合

| 左矩阵 (fm type) | 右矩阵 (filter type) | bias type | 结果矩阵 (dst type) |
| :--- | :--- | :--- | :--- |
| int8_t | int8_t | int32_t | int32_t |
| half | half | float | float |
| float | float | float | float |
| bfloat16_t | bfloat16_t | float | float |
| fp8_e4m3fn_t | fp8_e4m3fn_t | float | float |
| fp8_e4m3fn_t | fp8_e5m2_t | float | float |
| fp8_e5m2_t | fp8_e4m3fn_t | float | float |
| fp8_e5m2_t | fp8_e5m2_t | float | float |
| hifloat8_t | hifloat8_t | float | float |

## 返回值说明

无

## 流水类型

PIPE_M

## 约束说明

- 地址重叠约束：源操作数与目的操作数位于不同物理位置，不存在地址重叠场景。
- 地址对齐约束：源操作数起始地址要求512字节对齐。目的操作数起始地址要求满足64字节对齐。
- 参数组合约束：无约束。
- 环境影响约束：无约束。
- 特殊数据类型约束：无约束。
- 性能约束：无约束。
- 异常和边界值处理：当m、n、k中的任意一个值为0时，该指令不会被执行。

## 调用示例

```cpp
using namespace AscendC::Te;

constexpr int32_t L0_MAX = 128 * 128 * 2;
constexpr int32_t L0C_MAX = 128 * 128;

size_t m = 128;
size_t n = 64;
size_t k = 64;

auto l0ALayout = MakeNzLayout<half>(m, k);
auto l0BLayout = MakeZnLayout<half>(k, n);
auto l0CLayout = MakeL0CLayout(m, n);
auto biasLayout = MakeNDCLayout<float>(1, n);

__ca__ half l0AAddr[L0_MAX];
__cb__ half l0BAddr[L0_MAX];
__cc__ float l0CAddr[L0C_MAX];
__biasbuf__ float biasAddr[L0C_MAX];

auto l0APtr = MakeL0AmemPtr(l0AAddr);
auto l0BPtr = MakeL0BmemPtr(l0BAddr);
auto l0CPtr = MakeL0CmemPtr(l0CAddr);
auto biasPtr = MakeBiasmemPtr(biasAddr);

auto l0ATensor = MakeTensor(l0APtr, l0ALayout);
auto l0BTensor = MakeTensor(l0BPtr, l0BLayout);
auto l0CTensor = MakeTensor(l0CPtr, l0CLayout);
auto biasTensor = MakeTensor(biasPtr, biasLayout);

MmadParams para;
para.m = m;
para.n = n;
para.k = k;
para.cmatrixInitVal = true;


// 下面接口为3种调用方式的示例
// 调用方式1
Mad(MmadAtom<MmadTraits<MmadOperation, MmadTraitDefault>>{}, l0CTensor, l0ATensor, l0BTensor, para);
// 调用方式2
MmadAtom<MmadTraits<MmadOperation, MmadTraitDefault>>{}.Call(l0CTensor, l0ATensor, l0BTensor, para);
// 调用方式3
auto atomMad = MakeMad(MmadOperation{}, MmadTraitDefault{});
atomMad.Call(l0CTensor, l0ATensor, l0BTensor, para);

// 带Bias的调用方式
Mad(MmadAtom<MmadTraits<MmadOperation, MmadTraitDefault>>{}, l0CTensor, l0ATensor, l0BTensor, biasTensor, para);

```

完整样例请参考[TensorAPI样例代码](../../../../../examples/01_simd_cpp_api/02_features/05_tensor_api/matmul_quant_relu/matmul_quant_relu.asc)。
