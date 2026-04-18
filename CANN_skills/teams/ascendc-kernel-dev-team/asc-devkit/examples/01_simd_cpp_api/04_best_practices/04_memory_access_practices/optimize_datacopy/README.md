# 13_optimize_datacopy样例

## 概述

基于搬运类API使用的优化样例，通过<<<>>>直调的实现方式，介绍了减少无效数据搬运、减少搬运指令数量等方法。

## 算子开发样例

|  目录名称  |  功能描述  |
| -------------------------------------------------- | ---------------------------------------------------- |
| [optimize_datacopy_loop_mode](./optimize_datacopy_loop_mode) | 在使用DataCopyPad接口时，使用loop模式减少DataCopyPad指令的条数。 |
| [optimize_datacopy_nddma](./optimize_datacopy_nddma) | 在进行非对齐数据搬运时，使用nddma搬运减少搬运指令的条数。 |
| [optimize_reduce_invalid_datacopy](./optimize_reduce_invalid_datacopy) | 在使用DataCopyPad接口时，通过设置Compact模式减少无效数据的搬运。 |
