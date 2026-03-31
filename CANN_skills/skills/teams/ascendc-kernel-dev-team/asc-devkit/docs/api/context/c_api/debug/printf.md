# printf

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

该接口提供NPU域调试场景下的格式化输出功能。

在算子kernel侧实现代码中需要输出日志信息的地方调用printf接口打印相关内容。样例如下：

```
#include "c_api/asc_simd.h"
printf("fmt string %d\n", 0x123);
```

> [!CAUTION]注意  
>printf接口打印功能会对算子实际运行的性能带来一定影响（具体性能影响取决于代码中printf的使用数量），通常在调测阶段使用。开发者可以通过设置ASCENDC_DUMP为1来开启打印功能。

## 函数原型

```
template <class... Args>
__aicore__ inline void printf(__gm__ const char* fmt, Args&&... args)
```

## 参数说明

|参数名|输入/输出|描述|
| ------------ | ------------ | ------------ |
|fmt|输入|格式控制字符串，包含两种类型的对象：普通字符和转换说明。<br><br><ul><li>普通字符将原样不动地打印输出。</li> <li>转换说明并不直接输出而是用于控制printf中参数的转换和打印。每个转换说明都由一个百分号字符（%）开始，以转换说明结束，从而说明输出数据的类型 。<br>支持的转换类型包括：<ul><li>%d / %i：输出十进制数，支持打印的数据类型：bool/int8_t/int16_t/int32_t/int64_t</li><li>%f：输出实数，支持打印的数据类型：float/half/bfloat16_t</li><li>%x：输出十六进制整数，支持打印的数据类型：int8_t/int16_t/int32_t/int64_t/uint8_t/uint16_t/uint32_t/uint64_t</li><li>%s：输出字符串</li><li>%u：输出unsigned类型数据，支持打印的数据类型：bool/uint8_t/uint16_t/uint32_t/uint64_t</li><li>%p：输出指针地址</li></ul></li></ul>**注意**<ul><li>上文列出的数据类型是NPU域调试支持的数据类型。</li><li>在转换类型为%x，即输出十六进制整数时，NPU域上的输出为64位。</li>|
|args|输入|附加参数，个数和类型可变的参数列表：根据不同的fmt字符串，函数可能需要一系列的附加参数，每个参数包含了一个要被插入的值，替换了fmt参数中指定的每个%标签。参数的个数应与%标签的个数相同。|

## 返回值说明

无

## 流水类型

无

## 约束说明

-   本接口不支持打印除换行符之外的其他转义字符。
-   如果开发者需要包含标准库头文件stdio.h和cstdio，请在c_api/asc_simd.h头文件之前包含，避免printf符号冲突。
-   该接口使用Dump功能，所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1M。请开发者自行控制待打印的内容数据量，超出则不会打印。

## 调用示例

```
#include "c_api/asc_simd.h"

// 整型打印：
printf("fmt string %d\n", 0x123);

// 浮点型打印：
float a = 3.14;
printf("fmt string %f\n", a);

// 指针打印：
int *b;
printf("TEST %p\n", b);
```

NPU模式下，程序运行时打印效果如下：

```
fmt string 291
fmt string 3.140000
TEST 0x12c08001a000
```

