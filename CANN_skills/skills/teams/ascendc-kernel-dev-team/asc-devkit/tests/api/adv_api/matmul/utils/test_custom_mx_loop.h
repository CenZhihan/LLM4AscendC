/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TEST_CUSTOM_MX_LOOP_H
#define TEST_CUSTOM_MX_LOOP_H

#include <cstdint>
#include "test_custom_loop.h"

namespace TestCustomModules {
template <typename IMPL, class A_TYPE, const auto &MM_CFG, typename = void>
class CustomMxLoop : public CustomLoop<IMPL, A_TYPE, MM_CFG>
{
public:
    __aicore__ inline int32_t GetTileShapeScaleM() const
    {
        // 64 is mocked M size
        return 64;
    }

    __aicore__ inline int32_t GetTileShapeScaleN() const
    {
        // 64 is mocked N size
        return 64;
    }

    __aicore__ inline int32_t GetScaleFactorM() const
    {
        return 1;
    }

    __aicore__ inline int32_t GetScaleFactorN() const
    {
        return 1;
    }

    __aicore__ inline bool IsScaleAML1FullLoad() const
    {
        return false;
    }

    __aicore__ inline int32_t GetOuterScaleMIdx() const
    {
        return 0;
    }

    __aicore__ inline int32_t GetOuterScaleNIdx() const
    {
        return 0;
    }

    __aicore__ inline int32_t GetNextOuterScaleMIdx() const
    {
        return 0;
    }

    __aicore__ inline int32_t GetNextOuterScaleNIdx() const
    {
        return 0;
    }
};

template <typename IMPL, typename TRANS_T, class A_TYPE, const auto& MM_CFG, typename = void>
class CustomMxKLoop : public CustomKLoop<IMPL, TRANS_T, A_TYPE, MM_CFG> {
public:
    __aicore__ inline int32_t GetScaleFactorKa() const
    {
        return 1;
    }

    __aicore__ inline int32_t GetScaleFactorKb() const
    {
        return 1;
    }

    __aicore__ inline int32_t GetNextOuterScaleKaIdx() const
    {
        return 0;
    }

    __aicore__ inline int32_t GetNextOuterScaleKbIdx() const
    {
        return 0;
    }

    __aicore__ inline int32_t GetOuterScaleKaIdx()
    {
        return 1;
    }

    __aicore__ inline int32_t GetOuterScaleKbIdx()
    {
        return 1;
    }

    __aicore__ inline int32_t GetTileShapeScaleKa() const
    {
        return 0;
    }

    __aicore__ inline int32_t GetTileShapeScaleKb() const
    {
        return 0;
    }

    __aicore__ inline bool IsAKL1FullLoad() const
    {
        return false;
    }

    __aicore__ inline bool IsScaleAKL1FullLoad() const
    {
        return false;
    }

    __aicore__ inline bool IsBKL1FullLoad() const
    {
        return false;
    }

    __aicore__ inline bool IsScaleBKL1FullLoad() const
    {
        return false;
    }

    __aicore__ inline int32_t GetStepInnerIdx() const
    {
        return 0;
    }
};
}
#endif