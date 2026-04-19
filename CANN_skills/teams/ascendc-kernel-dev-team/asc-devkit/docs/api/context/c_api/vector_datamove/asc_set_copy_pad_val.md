# asc_set_copy_pad_val

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <cann-filter npu_type="950"><term>Ascend 950PR/Ascend 950DT</term>  | √ </cann-filter>|

## 功能说明

和asc_copy_gm2ub_align或asc_copy_ub2gm_align接口配合使用，设置连续搬运数据块左右两侧需要填补的数据值。
<cann-filter npu_type="950">
对于Ascend 950PR/Ascend 950DT产品，该接口仅对asc_copy_gm2ub_align接口有效。
</cann-filter>

## 函数原型

```cpp
__aicore__ inline void asc_set_copy_pad_val(int8_t pad_value)
__aicore__ inline void asc_set_copy_pad_val(uint8_t pad_value)
__aicore__ inline void asc_set_copy_pad_val(int16_t pad_value)
__aicore__ inline void asc_set_copy_pad_val(uint16_t pad_value)
__aicore__ inline void asc_set_copy_pad_val(half pad_value)
__aicore__ inline void asc_set_copy_pad_val(bfloat16_t pad_value)
__aicore__ inline void asc_set_copy_pad_val(int32_t pad_value)
__aicore__ inline void asc_set_copy_pad_val(uint32_t pad_value)
__aicore__ inline void asc_set_copy_pad_val(float pad_value)
```

<cann-filter npu_type="950">

```cpp
// 如下接口仅在Ascend 950PR/Ascend 950DT版本上支持：
__aicore__ inline void asc_set_copy_pad_val(fp4x2_e2m1_t pad_value)
__aicore__ inline void asc_set_copy_pad_val(fp4x2_e1m2_t pad_value)
__aicore__ inline void asc_set_copy_pad_val(fp8_e8m0_t pad_value)
__aicore__ inline void asc_set_copy_pad_val(fp8_e5m2_t pad_value)
__aicore__ inline void asc_set_copy_pad_val(fp8_e4m3fn_t pad_value)
__aicore__ inline void asc_set_copy_pad_val(hifloat8_t pad_value)
```

</cann-filter>

## 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| pad_val     | 输入     | 需要填补的数据值。|

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
// 源操作数非对齐，需要填补数据
asc_set_copy_pad_val(0);
asc_copy_gm2ub_align(dst, src, 2, 48 * sizeof(int8_t), 0, 0, true, 0, 0, 0);
```
