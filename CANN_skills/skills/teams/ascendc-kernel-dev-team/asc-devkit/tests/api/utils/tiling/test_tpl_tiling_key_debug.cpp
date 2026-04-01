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

namespace AscendC
{
class TestTplTilingKeyDebug : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TestTplTilingKeyDebug, GetTilingKeySuccess)
{
    TilingDeclareParams tilingDecl {
        ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 20, 30}, "DECL"),
        ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15, 25}, "DECL"),
        ParamStruct("N", ASCENDC_TPL_UINT, 8, {2, 2, 0, 2, 3, 5, 7, 6}, "DECL"),
        ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "DECL")
    };
    TilingSelectParams tilingSel {
        {
            ParamStruct("kernel_type", ASCENDC_TPL_KERNEL_TYPE, 8, {4}, "SEL"),
            ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 30}, "SEL"),
            ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15}, "SEL"),
            ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            ParamStruct("N", ASCENDC_TPL_UINT, 8, {1, 4, 7}, "SEL")
        },
        {
            ParamStruct("kernel_type", ASCENDC_TPL_KERNEL_TYPE, 8, {4}, "SEL"),
            ParamStruct("N", ASCENDC_TPL_UINT, 8, {1, 4, 6}, "SEL"),
            ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {20}, "SEL"),
            ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15, 25}, "SEL")
        },
    };
    auto tilingKey1 = AscendC::EncodeTilingKey(tilingDecl, tilingSel, { 10, 15, 7, 1});
    uint64_t golden1 = 17174282;
    EXPECT_EQ(tilingKey1, golden1);

    auto tilingKey2 = AscendC::EncodeTilingKey(tilingDecl, tilingSel, { 20, 25, 4, 0 });
    uint64_t golden2 = 268564;
    EXPECT_EQ(tilingKey2, golden2);
}

TEST_F(TestTplTilingKeyDebug, GetTilingKeyFailed1)
{
    TilingDeclareParams tilingDeclareParams {
        ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 20, 30}, "DECL"),
        ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15, 25}, "DECL"),
        ParamStruct("N", ASCENDC_TPL_UINT, 8, {8, 2, 0, 2, 3, 5, 7, 6}, "DECL"),
        ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "DECL")
    };
    TilingSelectParams tilingSelectParams {
        {
            ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 30}, "SEL"),
            ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15}, "SEL"),
            ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            ParamStruct("N", ASCENDC_TPL_UINT, 8, {1, 4, 7}, "SEL")
        },
        {
            ParamStruct("NKKK", ASCENDC_TPL_UINT, 8, {1, 4, 6, 100}, "SEL"),
            ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {20}, "SEL"),
            ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15, 25}, "SEL")
        },
    };
    auto tilingKey = AscendC::EncodeTilingKey(tilingDeclareParams,
                                                      tilingSelectParams,
                                                      { 10, 15, 7, 1});
    EXPECT_EQ(tilingKey, INVALID_TILING_KEY);
}


TEST_F(TestTplTilingKeyDebug, GetTilingKeyFailed2)
{
    TilingDeclareParams tilingDeclareParams {
        ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 20, 30}, "DECL"),
        ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15, 25}, "DECL"),
        ParamStruct("N", ASCENDC_TPL_UINT, 8, {2, 0, 2, 3, 5, 7, 6}, "DECL"),
        ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "DECL")
    };
    TilingSelectParams tilingSelectParams {
        {
            ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 30}, "SEL"),
            ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15}, "SEL"),
            ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            ParamStruct("N", ASCENDC_TPL_UINT, 8, {1, 4, 7}, "SEL")
        },
        {
            ParamStruct("NKKK", ASCENDC_TPL_UINT, 8, {1, 4, 6, 100}, "SEL"),
            ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {20}, "SEL"),
            ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15, 25}, "SEL")
        },
    };
    auto tilingKey = AscendC::EncodeTilingKey(tilingDeclareParams,
                                                tilingSelectParams,
                                                { 10, 15, 7, 1});
    EXPECT_EQ(tilingKey, INVALID_TILING_KEY);
}


TEST_F(TestTplTilingKeyDebug, GetTilingKeyFailed3)
{
    TilingDeclareParams tilingDeclareParams {};
    TilingSelectParams tilingSelectParams {};
    
    auto tilingKey = AscendC::EncodeTilingKey(tilingDeclareParams,
                                                tilingSelectParams,
                                                { 10, 15, 7, 1});
    EXPECT_EQ(tilingKey, INVALID_TILING_KEY);
}


TEST_F(TestTplTilingKeyDebug, GetTilingKeyParamStructValParseFailed1)
{
    TilingDeclareParams tilingDeclareParams {
        ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 20, 30}, "DECL"),
        ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15, 25}, "DECL"),
        ParamStruct("N", ASCENDC_TPL_UINT, 8, {8}, "DECL"),
        ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "DECL")
    };
    TilingSelectParams tilingSelectParams {
        {
            ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 30}, "SEL"),
            ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15}, "SEL"),
            ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            ParamStruct("N", ASCENDC_TPL_UINT, 8, {1, 4, 7}, "SEL")
        },
    };
    auto tilingKey = AscendC::EncodeTilingKey(tilingDeclareParams,
                                                      tilingSelectParams,
                                                      { 10, 15, 7, 1});
    EXPECT_EQ(tilingKey, INVALID_TILING_KEY);
}

TEST_F(TestTplTilingKeyDebug, GetTilingKeyParamStructValParseFailed2)
{
    TilingDeclareParams tilingDeclareParams {
        ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 20, 30}, "DECL"),
        ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {}, "DECL"),
        ParamStruct("N", ASCENDC_TPL_UINT, 8, {8}, "DECL"),
        ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "DECL")
    };
    TilingSelectParams tilingSelectParams {
        {
            ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 30}, "SEL"),
            ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15}, "SEL"),
            ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            ParamStruct("N", ASCENDC_TPL_UINT, 8, {1, 4, 7}, "SEL")
        },
    };
    auto tilingKey = AscendC::EncodeTilingKey(tilingDeclareParams,
                                        tilingSelectParams,
                                        { 10, 15, 7, 1});
    EXPECT_EQ(tilingKey, INVALID_TILING_KEY);
}

TEST_F(TestTplTilingKeyDebug, GetTilingKeyParamStructValParseFailed3)
{
    TilingDeclareParams tilingDeclareParams {
        ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 20, 30}, "DECL"),
        ParamStruct("Y", 123321, 8, {15, 25}, "DECL"),
        ParamStruct("N", ASCENDC_TPL_UINT, 8, {8}, "DECL"),
        ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "DECL")
    };
    TilingSelectParams tilingSelectParams {
        {
            ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 30}, "SEL"),
            ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15}, "SEL"),
            ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            ParamStruct("N", ASCENDC_TPL_UINT, 8, {1, 4, 7}, "SEL")
        },
    };
    
    auto tilingKey = AscendC::EncodeTilingKey(tilingDeclareParams,
                                        tilingSelectParams,
                                        { 10, 15, 7, 1});
    EXPECT_EQ(tilingKey, INVALID_TILING_KEY);
}

TEST_F(TestTplTilingKeyDebug, GetTilingKeyParamStructValParseFailed4)
{
    TilingDeclareParams tilingDeclareParams {
            ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 20, 30}, "DECL"),
            ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15, 25}, "DECL"),
            ParamStruct("N", ASCENDC_TPL_UINT, 8, {2, 0, 2, 3, 5, 7, 6}, "DECL"),
            ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "DECL")
    };
    TilingSelectParams tilingSelectParams {
            {
                ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 30}, "SEL"),
                ParamStruct("Y", ASCENDC_TPL_DTYPE, 8, {15}, "SEL"),
                ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
                ParamStruct("N", ASCENDC_TPL_UINT, 8, {1, 4, 7}, "SEL")
            },
    };
    auto tilingKey = AscendC::EncodeTilingKey(tilingDeclareParams,
                                        tilingSelectParams,
                                        { 10, 15, 7, 1});
    EXPECT_EQ(tilingKey, INVALID_TILING_KEY);
}

TEST_F(TestTplTilingKeyDebug, GetTilingKeyParamStructValParseFailed5)
{
    TilingDeclareParams tilingDeclareParams {
        ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 20, 30}, "DECL"),
        ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15, 25}, "DECL"),
        ParamStruct("N", ASCENDC_TPL_UINT, 8, {2, 0, 2, 3, 5, 7, 6}, "DECL"),
        ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "DECL")
    };
    TilingSelectParams tilingSelectParams {
            {
                ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 30}, "SEL"),
                ParamStruct("Y", ASCENDC_TPL_DTYPE, 8, {15}, "SEL"),
                ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            },
    };
    EXPECT_EQ(AscendC::EncodeTilingKey(tilingDeclareParams, tilingSelectParams,
                                        { 10, 15, 1}), INVALID_TILING_KEY);
}


TEST_F(TestTplTilingKeyDebug, GetTilingKeyParamStructValParseFailed6)
{
    TilingDeclareParams tilingDeclareParams {
        ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 20, 30}, "DECL"),
        ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15, 25}, "DECL"),
        ParamStruct("N", ASCENDC_TPL_UINT, 8, {2, 0, 2, 3, 5, 7, 6}, "DECL"),
        ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "DECL")
    };
    TilingSelectParams tilingSelectParams {
            {
                ParamStruct("X", 234234, 8, {10, 30}, "SEL"),
                ParamStruct("Y", ASCENDC_TPL_DTYPE, 8, {15}, "SEL"),
                ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            },
    };
    EXPECT_EQ(AscendC::EncodeTilingKey(tilingDeclareParams, tilingSelectParams,
                                        { 10, 15, 1}), INVALID_TILING_KEY);
}

TEST_F(TestTplTilingKeyDebug, GetTilingKeyParamStructValParseFailed7)
{
    TilingDeclareParams tilingDeclareParams {
        ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 20, 30}, "DECL")};
    TilingSelectParams tilingSelectParams {{ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 30}, "SEL")}};
    EXPECT_EQ(AscendC::EncodeTilingKey(tilingDeclareParams, tilingSelectParams,
                                    { 10, 15, 1}), INVALID_TILING_KEY);
}


TEST_F(TestTplTilingKeyDebug, GetTilingKeyParamStructValFailed)
{
    TilingDeclareParams tilingDeclareParams {
        ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 20, 30}, "DECL"),
        ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15, 25}, "DECL"),
        ParamStruct("N", ASCENDC_TPL_UINT, 8, {2, 1}, "DECL"),
        ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "DECL")
    };
    TilingSelectParams tilingSelectParams {
        {
            ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 30}, "SEL"),
            ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15}, "SEL"),
            ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            ParamStruct("N", ASCENDC_TPL_UINT, 8, {1, 4, 7}, "SEL")
        },
        {
            ParamStruct("N", ASCENDC_TPL_UINT, 8, {1, 4, 6}, "SEL"),
            ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {20}, "SEL"),
            ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15, 25}, "SEL")
        },
    };
    auto tilingKey = AscendC::EncodeTilingKey(tilingDeclareParams,
                                                      tilingSelectParams,
                                                      { 10, 15, 7, 1});
    EXPECT_EQ(tilingKey, INVALID_TILING_KEY);
}

TEST_F(TestTplTilingKeyDebug, GetTilingKeyParamStructUniqueFailed)
{
    TilingDeclareParams tilingDeclareParams {
        ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 20, 30}, "DECL"),
        ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15, 25}, "DECL"),
        ParamStruct("N", ASCENDC_TPL_UINT, 8, {2, 2, 0, 2, 2, 5, 7, 6}, "DECL"),
        ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "DECL")
    };
    TilingSelectParams tilingSelectParams {
        {
            ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 30}, "SEL"),
            ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15}, "SEL"),
            ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            ParamStruct("N", ASCENDC_TPL_UINT, 8, {1, 4, 7}, "SEL")
        },
        {
            ParamStruct("N", ASCENDC_TPL_UINT, 8, {1, 4, 6}, "SEL"),
            ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {20}, "SEL"),
            ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15, 25}, "SEL")
        },
    };
    auto tilingKey = AscendC::EncodeTilingKey(tilingDeclareParams,
                                                      tilingSelectParams,
                                                      { 10, 15, 7, 1});
    EXPECT_EQ(tilingKey, INVALID_TILING_KEY);
}

TEST_F(TestTplTilingKeyDebug, GetTilingKeyParamStructMaxFailed)
{
    TilingDeclareParams tilingDeclareParams {
        ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 20, 30}, "DECL"),
        ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15, 25}, "DECL"),
        ParamStruct("N", ASCENDC_TPL_UINT, 8, {2, 2, 0, 2, 3, 5, 7, 60000000000000}, "DECL"),
        ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "DECL")
    };
    TilingSelectParams tilingSelectParams {
        {
            ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 30}, "SEL"),
            ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15}, "SEL"),
            ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            ParamStruct("N", ASCENDC_TPL_UINT, 8, {1, 4, 7}, "SEL")
        },
        {
            ParamStruct("N", ASCENDC_TPL_UINT, 8, {1, 4, 6}, "SEL"),
            ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {20}, "SEL"),
            ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15, 25}, "SEL")
        },
    };
    auto tilingKey = AscendC::EncodeTilingKey(tilingDeclareParams,
                                                      tilingSelectParams,
                                                      { 10, 15, 7, 1});
    EXPECT_EQ(tilingKey, INVALID_TILING_KEY);
}

TEST_F(TestTplTilingKeyDebug, GetTilingKeyParamNoTplFailed)
{
    TilingDeclareParams tilingDeclareParams {
        ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 20, 30}, "DECL"),
        ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15, 25}, "DECL"),
        ParamStruct("N", ASCENDC_TPL_UINT, 8, {2, 2, 0, 2, 3, 5, 7, 6}, "DECL"),
        ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "DECL")
    };
    TilingSelectParams tilingSelectParams {
        {
            ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {10, 40}, "SEL"),
            ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15}, "SEL"),
            ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            ParamStruct("N", ASCENDC_TPL_UINT, 8, {1, 4, 7}, "SEL")
        },
        {
            ParamStruct("N", ASCENDC_TPL_UINT, 8, {1, 4, 6}, "SEL"),
            ParamStruct("S", ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            ParamStruct("X", ASCENDC_TPL_DTYPE, 8, {20}, "SEL"),
            ParamStruct("Y", ASCENDC_TPL_FORMAT, 8, {15, 25}, "SEL")
        },
    };
    auto tilingKey = AscendC::EncodeTilingKey(tilingDeclareParams,
                                                      tilingSelectParams,
                                                      { 10, 15, 7, 1});
    EXPECT_EQ(tilingKey, INVALID_TILING_KEY);
}

TEST_F(TestTplTilingKeyDebug, GetTilingKeySelectParamFailed)
{
    TilingDeclareParams tilingDeclareParams {
        ParamStruct("X",  ASCENDC_TPL_DTYPE, 8, {10, 20, 30}, "DECL"),
        ParamStruct("Y",  ASCENDC_TPL_FORMAT, 8, {15, 25}, "DECL"),
        ParamStruct("N",  ASCENDC_TPL_UINT, 8, {2, 2, 0, 2, 3, 5, 7, 6}, "DECL"),
        ParamStruct("S",  ASCENDC_TPL_BOOL, 1, {0, 1}, "DECL")
    };
    TilingSelectParams tilingSelectParams {
        {
            ParamStruct("X",  ASCENDC_TPL_DTYPE, 8, {10, 30}, "SEL"),
            ParamStruct("Y",  ASCENDC_TPL_FORMAT, 8, {15}, "SEL"),
            ParamStruct("S",  ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            ParamStruct("N",  ASCENDC_TPL_UINT, 8, {1, 4, 7}, "SEL")
        },
        {
            ParamStruct("N",  ASCENDC_TPL_UINT, 8, {1, 4, 6}, "SEL"),
            ParamStruct("S",  ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            ParamStruct("X",  ASCENDC_TPL_DTYPE, 8, {20}, "SEL"),
            ParamStruct("Y",  ASCENDC_TPL_FORMAT, 8, {15, 25}, "SEL")
        },
    };
    auto tilingKey = AscendC::EncodeTilingKey(tilingDeclareParams,
                                                      tilingSelectParams,
                                                      { 10, 15, 7, 1111111});
    EXPECT_EQ(tilingKey, INVALID_TILING_KEY);
}

TEST_F(TestTplTilingKeyDebug, GetTilingKeyValidFailed)
{
    TilingDeclareParams tilingDeclareParams {
        ParamStruct("X",  ASCENDC_TPL_DTYPE, 8, {10, 20, 30}, "DECL"),
        ParamStruct("Y", 1000, 8, {15, 25}, "DECL"),
        ParamStruct("N",  ASCENDC_TPL_UINT, 8, {2, 2, 0, 2, 3, 5, 7, 6}, "DECL"),
        ParamStruct("S",  ASCENDC_TPL_BOOL, 1, {0, 1}, "DECL")
    };
    TilingSelectParams tilingSelectParams {
        {
            ParamStruct("X",  ASCENDC_TPL_DTYPE, 8, {10, 30}, "SEL"),
            ParamStruct("Y", 1000, 8, {15}, "SEL"),
            ParamStruct("S",  ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            ParamStruct("N",  ASCENDC_TPL_UINT, 8, {1, 4, 7}, "SEL")
        },
        {
            ParamStruct("N",  ASCENDC_TPL_UINT, 8, {1, 4, 6}, "SEL"),
            ParamStruct("S",  ASCENDC_TPL_BOOL, 1, {0, 1}, "SEL"),
            ParamStruct("X",  ASCENDC_TPL_DTYPE, 8, {20}, "SEL"),
            ParamStruct("Y", 1000, 8, {15, 25}, "SEL")
        },
    };
    auto tilingKey = AscendC::EncodeTilingKey(tilingDeclareParams,
                                                      tilingSelectParams,
                                                      { 10, 15, 7, 1});
    EXPECT_EQ(tilingKey, INVALID_TILING_KEY);
}

TEST_F(TestTplTilingKeyDebug, GetTilingKeyParamMaxRangeFailed)
{
    TilingDeclareParams tilingDeclareParams {
        ParamStruct("X",  ASCENDC_TPL_DTYPE, 41, {10, 20, 30}, "DECL"),
        ParamStruct("Y",  ASCENDC_TPL_FORMAT, 8, {15, 25}, "DECL"),
        ParamStruct("N",  ASCENDC_TPL_UINT, 8, {2, 2, 0, 2, 3, 5, 7, 6}, "DECL"),
        ParamStruct("S",  ASCENDC_TPL_BOOL, 8, {0, 1}, "DECL")
    };
    TilingSelectParams tilingSelectParams {
        {
            ParamStruct("X",  ASCENDC_TPL_DTYPE, 41, {10, 30}, "SEL"),
            ParamStruct("Y",  ASCENDC_TPL_FORMAT, 8, {15}, "SEL"),
            ParamStruct("S",  ASCENDC_TPL_BOOL, 8, {0, 1}, "SEL"),
            ParamStruct("N",  ASCENDC_TPL_UINT, 8, {1, 4, 7}, "SEL")
        },
        {
            ParamStruct("N",  ASCENDC_TPL_UINT, 8, {1, 4, 6}, "SEL"),
            ParamStruct("S",  ASCENDC_TPL_BOOL, 8, {0, 1}, "SEL"),
            ParamStruct("X",  ASCENDC_TPL_DTYPE, 41, {20}, "SEL"),
            ParamStruct("Y",  ASCENDC_TPL_FORMAT, 8, {15, 25}, "SEL")
        },
    };
    auto tilingKey = AscendC::EncodeTilingKey(tilingDeclareParams,
                                                      tilingSelectParams,
                                                      { 10, 15, 7, 1});
    EXPECT_EQ(tilingKey, INVALID_TILING_KEY);
}

} // namespace AscendC