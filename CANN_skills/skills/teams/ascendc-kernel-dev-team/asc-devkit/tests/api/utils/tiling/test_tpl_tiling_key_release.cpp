/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <string>
#include "utils/tiling/template_argument.h"

namespace AscendC {
class TestTplTilingKeyRelease : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TestTplTilingKeyRelease, GetTilingKeySuccess)
{
    TilingDeclareParams tilingDecl {
        ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 20, 30}, "DECL"),
        ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15, 25}, "DECL"),
        ParamStruct("N", ASCENDC_TPL_UINT, 8, {2, 2, 0, 2, 3, 5, 7, 6}, "DECL"),
        ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "DECL")
    };
    auto tilingKey1 = AscendC::FastEncodeTilingKeyDirect(tilingDecl, { 10, 15, 7, 1});
    uint64_t golden1 = 17174282;
    EXPECT_EQ(tilingKey1, golden1);

    auto tilingKey2 = AscendC::FastEncodeTilingKeyDirect(tilingDecl, { 20, 25, 4, 0});
    uint64_t golden2 = 268564;
    EXPECT_EQ(tilingKey2, golden2);
}

} // namespace AscendC