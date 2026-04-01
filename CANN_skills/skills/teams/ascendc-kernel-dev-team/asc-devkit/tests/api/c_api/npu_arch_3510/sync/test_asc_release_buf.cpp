/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

template <typename DTYPE>
__aicore__ inline void rls_buf_stub(pipe_t pipe, DTYPE buf_id, bool mode) {
    EXPECT_EQ(pipe, static_cast<pipe_t>(pipe_t::PIPE_S));
    EXPECT_EQ(buf_id, static_cast<DTYPE>(11));
    EXPECT_EQ(mode, static_cast<bool>(true));
}

class TEST_ASC_RELEASE_BUF : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

#define TEST_ASC_RELEASE_BUF(dtype)  \
                                                                                      \
TEST_F(TEST_ASC_RELEASE_BUF, TEST_ASC_RELEASE_BUF_##dtype)               \
{                                                                                     \
    MOCKER_CPP(rls_buf, void(pipe_t, dtype, bool))                    \
            .times(1)                                                                           \
            .will(invoke(rls_buf_stub<dtype>));                               \
                                                                                                \
    pipe_t pipe = static_cast<pipe_t>(pipe_t::PIPE_S);                                              \
    dtype buf_id = static_cast<dtype>(11);                                             \
    bool mode = static_cast<bool>(true);                                             \
                                                                                                    \
    asc_release_buf(pipe, buf_id, mode);         \
    GlobalMockObject::verify();                                                                 \
}

// ==========asc_release_buf==========
TEST_ASC_RELEASE_BUF(uint8_t);
TEST_ASC_RELEASE_BUF(uint64_t);