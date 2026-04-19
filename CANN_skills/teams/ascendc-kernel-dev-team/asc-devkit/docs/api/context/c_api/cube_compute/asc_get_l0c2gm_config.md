# asc_get_l0c2gm_config

## 产品支持情况

| 产品     | 是否支持 |
| :----------- |:----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

数据搬运过程中进行随路量化时，通过调用该接口获取量化流程中的矢量量化参数。

## 函数原型

```cpp
__aicore__ inline uint64_t asc_get_l0c2gm_relu()
__aicore__ inline uint64_t asc_get_l0c2gm_unitflag()
__aicore__ inline uint64_t asc_get_l0c2gm_prequant()
```

## 参数说明

无

## 返回值说明

| 函数名 | 返回值描述 |
| :----- | :-------- |
| asc_get_l0c2gm_relu | ReLU操作前矢量的起始地址。|
| asc_get_l0c2gm_unitflag | unit_flag设置。unit_flag是一种矩阵计算指令和矩阵搬运指令细粒度的并行，使能该功能后，硬件每计算完一个分形，计算结果就会被搬出，该功能不适用于L0C Buffer累加的场景。 |
| asc_get_l0c2gm_prequant | 量化操作前矢量的起始地址。|

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
uint64_t relu_value = asc_get_l0c2gm_relu();
uint64_t uintflag_value = asc_get_l0c2gm_unitflag();
uint64_t prequant_value = asc_get_l0c2gm_prequant();
```
