# 系统变量访问类api样例介绍

## 概述

本路径下包含了与系统变量访问相关的多个API的样例。每个样例均基于Ascend C的<<<>>>直调方法，支持main函数和kernel函数在同一个cpp文件中实现。

## 算子开发样例

| 目录名称 | 功能描述 |
| ------- | -------- |
| [get_ctrl_spr](./get_ctrl_spr)     | 本样例基于GetCtrlSpr接口实现读取CTRL寄存器特定比特位值的功能。 |
| [reset_ctrl_spr](./reset_ctrl_spr) | 本样例基于ResetCtrlSpr接口实现重置CTRL寄存器特定比特位的功能。 |
| [set_ctrl_spr](./set_ctrl_spr)     | 本样例基于SetCtrlSpr接口实现设置CTRL寄存器特定比特位的功能。 |
