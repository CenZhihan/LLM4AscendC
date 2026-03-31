# assert

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

在算子Kernel侧实现代码中需要增加断言的地方使用assert检查代码，并格式化输出一些调测信息。示例如下：

```
int assertFlag = 10;

assert(assertFlag == 10);
```

> [!CAUTION]注意  
>assert接口打印功能会对算子实际运行的性能带来一定影响（每一条assert，系统会额外增加一条逻辑判断，具体性能影响取决于代码中assert的使用数量），通常在调测阶段使用。开发者可以通过设置ASCENDC_DUMP为1来开启打印功能。

## 函数原型

```
assert(expr)
```

## 参数说明

|参数名|输入/输出|描述|
| ------------ | ------------ | ------------ |
|expr|输入|assert断言是否终止程序的条件。为true则程序继续执行，为false则终止程序。|

## 返回值说明

无

## 流水类型

无

## 约束说明

-   Kernel开发不要包含系统的assert.h，会导致宏定义冲突。
-   该接口不支持打印除换行符之外的其他转义字符。
-   该接口使用Dump功能，所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1M。请开发者自行控制待打印的内容数据量，超出则不会打印。

## 调用示例

```
int assertFlag = 10;
// 断言条件
assert(assertFlag != 10);
```

程序运行时会触发assert，打印效果如下：

```
[ASSERT] /home/.../add_custom.cpp:44: Assertion `assertFlag != 10' 
```

