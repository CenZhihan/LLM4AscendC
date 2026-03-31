# dump

## 产品支持情况

| 产品          | 是否支持|
| :-----------------------| :-----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |   √   |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |   √   |

## 功能说明

将对应内存上的数据打印出来,同时支持打印自定义的附加信息（仅支持uint32_t类型的信息）。比如打印当前行号等。

## 函数原型

GM内存上的数据打印
  ```cpp
  template<typename T>
  asc_dump_gm(__gm__ T* input, uint32_t desc, uint32_t dump_size)
  ```

  UB内存上的数据打印
  ```cpp
  template<typename T>
  asc_dump_ubuf(__ubuf__ T* input, uint32_t desc, uint32_t dump_size)
  ```
## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :----| :-----| :-----|
| input | 输入 | 所需打印的内存块的起始地址。 |
| desc  | 输入 | 自定义信息。|
| dump_size | 输入 | 所需要打印的元素数量。 |

## 返回值说明

无

## 流水类型

无

## 约束说明

- 暂时仅支持GM、UB上的内存打印。
- 使用该接口时，在每个核上dump的数据总量不能超过1M，请开发者自行控制打印的内容数据量，超出则不会打印。
- 在计算数据量时，若dump的总长度未对齐，需要考虑padding数据的影响。当进行非对齐dump时，如果实际dump的元素长度不满足32字节对齐，
  系统会自动在其末尾补充一定数量的padding数据，以满足对其要求。
- 如果要正常使用dump功能，需要在kernel核函数外定义如下宏定义:"#define ASCENDC_DUMP 1"

## 调用示例

```cpp
__gm__ half* src;
uint32_t desc = 0;
uint32_t dump_size = 32;
asc_dump_gm<half>(src, desc, dump_size);
```