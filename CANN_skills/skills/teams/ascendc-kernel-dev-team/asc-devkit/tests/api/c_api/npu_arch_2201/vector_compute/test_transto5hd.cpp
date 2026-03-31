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
#include "c_api/stub/cce_stub.h"
#include "c_api/asc_simd.h"

class TestTransdataTo5HDCAPI : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

namespace {   
void scatter_vnchwconv_b8_stub(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dst_stride, uint16_t src_stride,
    bool dst_high_half, bool src_high_half)
{
    EXPECT_EQ(ub_addr8_t::VA0, dst);
    EXPECT_EQ(ub_addr8_t::VA2, src);
    EXPECT_EQ(repeat, static_cast<uint8_t>(1));
    EXPECT_EQ(dst_stride, static_cast<uint8_t>(1));
    EXPECT_EQ(src_stride, static_cast<uint8_t>(1));
    EXPECT_EQ(dst_high_half, static_cast<bool>(0));
    EXPECT_EQ(src_high_half, static_cast<bool>(0));
}
}

TEST_F(TestTransdataTo5HDCAPI, c_api_set_va_reg_Succ)
{
    ub_addr8_t dst = ub_addr8_t::VA0;
    ub_addr8_t src = ub_addr8_t::VA2;
    uint8_t repeat = 1;
    uint16_t dst_stride = 1;
    uint16_t src_stride = 1;
    bool dst_high_half = false;
    bool src_high_half = false;
    MOCKER_CPP(scatter_vnchwconv_b8, void(ub_addr8_t, ub_addr8_t, uint8_t, uint16_t, uint16_t, bool, bool))
            .times(1)
            .will(invoke(scatter_vnchwconv_b8_stub));

    asc_transto5hd_b8(dst, src, repeat, dst_stride, src_stride, dst_high_half, src_high_half);
    GlobalMockObject::verify();
}