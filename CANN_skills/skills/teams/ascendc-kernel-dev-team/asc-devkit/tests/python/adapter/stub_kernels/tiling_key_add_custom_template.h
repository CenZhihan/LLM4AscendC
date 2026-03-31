/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TILING_KEY_ADD_CUSTOM_TEMPLATE_H
#define TILING_KEY_ADD_CUSTOM_TEMPLATE_H

#if defined  ASCENDC_TPL_KERNEL || defined ASCENDC_TPL_PRE
#include "utils/tiling/template_argument.h"
#else
#include "tiling_api.h"
#endif

//// data types
#define _TILIG_FP16 10
#define _TILIG_FP32 20
#define _TILIG_BF16 30

//// formats
#define _TILIG_ND 15
#define _TILIG_NZ 25


ASCENDC_TPL_ARGS_DECL(AddCustom, //每个tiling_key定义唯一标识
ASCENDC_TPL_DTYPE_DECL(X, _TILIG_FP16, _TILIG_BF16, _TILIG_FP32), // 默认8比特, 可指定位宽
ASCENDC_TPL_FORMAT_DECL(Y, _TILIG_ND, _TILIG_NZ), // 默认4比特, 可指定位宽
ASCENDC_TPL_UINT_DECL(Z, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_MIX, 2, 0, 2, 3, 5,  7, 6),  // 默认8比特, 可指定位宽
ASCENDC_TPL_BOOL_DECL(S, 0, 1) // 默认1比特, 不可指定位宽
);

 ASCENDC_TPL_SEL(  // 2 * 1 * 2 *1 = 4个tiling key
    ASCENDC_TPL_ARGS_SEL( //vector
      ASCENDC_TPL_DTYPE_SEL(X, _TILIG_FP16, _TILIG_BF16),
      ASCENDC_TPL_FORMAT_SEL(Y, _TILIG_ND),
      ASCENDC_TPL_UINT_SEL(Z, ASCENDC_TPL_UI_LIST, 4, 7),
      ASCENDC_TPL_BOOL_SEL(S, 0, 1)
      ),

    ASCENDC_TPL_ARGS_SEL(// 2 * 1 * 2 *1 = 4个tiling key
      ASCENDC_TPL_DTYPE_SEL(X, _TILIG_FP32),
      ASCENDC_TPL_FORMAT_SEL(Y, _TILIG_ND, _TILIG_NZ),
      ASCENDC_TPL_UINT_SEL(Z, ASCENDC_TPL_UI_LIST, 4, 6),
      ASCENDC_TPL_BOOL_SEL(S, 0, 1)),
);
#endif