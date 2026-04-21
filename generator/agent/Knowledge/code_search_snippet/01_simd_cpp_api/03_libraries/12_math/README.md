# Math API样例介绍

## 概述

本样例集介绍了Math API不同特性的典型用法，给出了对应的端到端实现。

## 算子开发样例

| 目录名称                                 |  功能描述                                             |
|--------------------------------------| --------------------------------------------------- |
| [acos](./acos)                       | 本样例演示了基于Acos高阶API的算子实现。样例按元素做反余弦函数计算 |
| [acosh](./acosh)                     | 本样例演示了基于Acosh高阶API的算子实现。样例按元素做双曲反余弦函数计算 |
| [addcdiv](./addcdiv)                 | 本样例通过Ascend C编程语言实现了Addcdiv算子，使用<<<>>>内核调用符来完成算子核函数在NPU侧运行验证的基础流程 |
| [asin](./asin)                       | 本样例演示了基于Asin高阶API的算子实现。样例按元素做反正弦函数计算 |
| [asinh](./asinh)                     | 本样例演示了基于Asinh高阶API的算子实现。样例按元素做反双曲正弦函数计算 |
| [atanh](./atanh)                     | 本样例演示了基于Atanh高阶API的算子实现。样例按元素做反双曲正切余弦函数计算 |
| [axpy_half_float](./axpy_half_float) | 本样例基于Axpy实现源操作数src中每个元素与标量求积后和目的操作数dst中的对应元素相加的功能。Axpy接口的源操作数和目的操作数的数据类型只能取三种组合：(half, half)、(float, float)、(half, float)。本样例中输入tensor和标量的数据类型为half，输出tensor数据类型为float。本样例通过Ascend C编程语言实现了Axpy算子，使用<<<>>>内核调用符来完成算子核函数在NPU侧运行验证的基础流程，给出了对应的端到端实现 |
| [bitwiseand](./bitwiseand)           | 本样例演示了基于BitwiseAnd高阶API的算子实现。样例逐比特对两个输入进行与操作 |
| [bitwisenot](./bitwisenot)           | 本样例演示了基于BitwiseNot高阶API的算子实现。样例逐比特对输入进行取反 |
| [bitwiseor](./bitwiseor)             | 本样例演示了基于BitwiseOr高阶API的算子实现。样例逐比特对两个输入进行或操作 |
| [bitwisexor](./bitwisexor)           | 本样例演示了基于BitwiseXor高阶API的算子实现。样例逐比特对两个输入进行异或操作 |
| [axpy_half_half](./axpy_half_half)   | 本样例基于Axpy实现源操作数src中每个元素与标量求积后和目的操作数dst中的对应元素相加的功能。Axpy接口的源操作数和目的操作数的数据类型只能取三种组合：(half, half)、(float, float)、(half, float)。本样例中输入tensor、标量、输出tensor数据类型均为half。本样例通过Ascend C编程语言实现了Axpy算子，使用<<<>>>内核调用符来完成算子核函数在NPU侧运行验证的基础流程，给出了对应的端到端实现 | 
| [cast](./cast)                       | 本样例基于Cast实现数据精度转换，根据源操作数和目的操作数Tensor的数据类型进行精度转换 |
| [cast_int4b_t](./cast_int4b_t)       | 本样例基于Cast实现int4b_t类型的数据精度转换，进行half类型源操作数和int4b_t类型目的操作数Tensor之间的精度转换 |
| [ceil](./ceil)                       | 本样例演示了基于Ceil高阶API的算子实现。样例获取大于或等于x的最小的整数值，即向正无穷取整操作 |
| [clamp](./clamp)                     | 本样例演示了基于Clamp高阶API的算子实现。将输入中非nan且超出[min, max]范围的值剪裁至max或min，若min>max，则所有非nan值均置为max |
| [clampmax](./clampmax)               | 本样例演示了基于ClampMax高阶API的算子实现。样例将srcTensor中大于scalar的数替换为scalar，小于等于scalar的数保持不变，作为dstTensor输出 |
| [clampmin](./clampmin)               | 本样例演示了基于ClampMin高阶API的算子实现。样例将srcTensor中小于scalar的数替换为scalar，大于等于scalar的数保持不变，作为dstTensor输出 |
| [cos](./cos)                         | 本样例介绍了调用Cos高阶API实现cos算子，并按照核函数直调的方式给出了对应的端到端实现 |
| [cosh](./cosh)                       | 本样例演示了基于Cosh高阶API的算子实现。样例按元素做双曲余弦函数计算 |
| [cumsum](./cumsum)                   | 本样例介绍了调用CumSum高阶API实现cumsum单算子，用于对输入张量按行或列进行累加和操作 |
| [erf](./erf)                         | 本样例演示了基于Erf高阶API的算子实现。样例按元素做误差函数计算 |
| [erfc](./erfc)                       | 本样例演示了基于Erfc高阶API的算子实现。样例返回输入x的互补误差函数结果，积分区间为x到无穷大 |
| [exp](./exp)                         | 本样例演示了基于Exp高阶API的算子实现。样例按元素取自然指数，用户可以选择是否使用泰勒展开公式进行计算 |
| [floor](./floor)                     | 本样例演示了基于Floor高阶API的算子实现。样例获取小于或等于x的最小的整数值，即向负无穷取整操作 |
| [fma](./fma)                         | 本样例演示了基于Fma高阶API的算子实现。样例按元素计算两个输入相乘后与第三个输入相加的结果 |
| [fmod](./fmod)                       | 本样例演示了基于Fmod高阶API的算子实现。样例按元素计算两个浮点数a，b相除后的余数 |
| [frac](./frac)                       | 本样例演示了基于Frac高阶API的算子实现。样例按元素做取小数计算 |
| [isfinite](./isfinite)               | 本样例演示了基于IsFinite高阶API的算子实现。样例按元素判断输入的浮点数是否非NAN、非INF，输出结果为浮点数或者布尔值 |
| [isinf](./isinf)                     | 本样例演示了基于IsInf高阶API的算子实现。样例按元素判断输入的浮点数是否为 $\pm$ inf，输出结果为浮点数或布尔值 |
| [isnan](./isnan)                     | 本样例演示了基于IsNan高阶API的算子实现。样例按元素判断输入的浮点数是否为nan，输出结果为浮点数或布尔值 |
| [leaky_relu](./leaky_relu)           | 本样例基于LeakyRelu实现激活函数，可用于对输入tensor按元素执行Leaky Relu（Leaky Rectified Liner Unit）操作 |
| [lgamma](./lgamma)                   | 本样例演示了基于Lgamma高阶API的算子实现。样例按元素计算x的gamma函数的绝对值并求自然对数 |
| [log](./log)                         | 本样例演示了基于Log高阶API的算子实现。样例按元素以e、2、10为底做对数运算 |
| [logicaland](./logicaland)           | 本样例演示了基于LogicalAnd高阶API的算子实现。样例按元素进行与操作，输入数据类型不是bool时，零被视为False，非零数据被视为True |
| [logicalands](./logicalands)         | 本样例演示了基于LogicalAnds高阶API的算子实现。样例对输入矢量内的每个元素和标量进行与操作 |
| [logicalnot](./logicalnot)           | 本样例演示了基于LogicalNot高阶API的算子实现。样例按元素进行取反操作，输入数据类型不是bool时，零被是被Flase，非零数据被视为True |
| [logicalor](./logicalor)             | 本样例演示了基于LogicalOr高阶API的算子实现。样例按元素进行或操作功能，输入数据类型不是bool时，零被视为False，非零数据被视为True |
| [logicalors](./logicalors)           | 本样例演示了基于LogicalOrs高阶API的算子实现。样例对输入矢量内的每个元素和标量进行或操作 |
| [logicalxor](./logicalxor)           | 本样例演示了基于LogicalXor高阶API的算子实现。样例按元素进行异或操作功能，输入数据类型不是bool时，零被视为False，非零数据被视为True |
| [power](./power)                     | 本样例演示了基于Power高阶API的算子实现。样例实现按元素做幂运算功能，支持三种功能：指数和底数分别为张量对张量、张量对标量、标量对张量的幂运算 |
| [rint](./rint)                       | 本样例演示了基于Rint高阶API的算子实现。样例获取与输入数据最接近的整数，若存在两个相同接近的整数，则获取其中的偶数 |
| [round](./round)                     | 本样例演示了基于Round高阶API的算子实现。样例将输入的元素四舍五入到最接近的整数 |
| [sign](./sign)                       | 本样例演示了基于Sign高阶API的算子实现。样例按元素执行Sign操作，返回输入数据的符号 |
| [sincos](./sincos)                   | 本样例演示了基于SinCos高阶API的算子实现。样例按元素进行正弦计算和余弦计算，分别获得正弦和余弦的结果 |
| [sinh](./sinh)                       | 本样例演示了基于Sinh高阶API的算子实现。样例按元素做双曲正弦函数计算 |
| [tan](./tan)                         | 本样例演示了基于Tan高阶API的算子实现。样例按元素做正切函数计算 |
| [trunc](./trunc)                     | 本样例演示了基于Trunc高阶API的算子实现。样例按元素做浮点数截断，即向零取整操作 |
| [where](./where)                     | 本样例演示了基于Where高阶API的算子实现。样例根据指定的条件，从两个源操作数中选择元素，生成目标操作数。两个源操作数均可以是LocalTensor或标量 |
| [xor](./xor)                         | 本样例演示了基于Xor高阶API的算子实现。样例按元素执行Xor运算 |