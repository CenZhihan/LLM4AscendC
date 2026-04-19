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
#include "kernel_operator.h"
#include "include/adv_api/matmul/tiling.h"
#include "impl/adv_api/detail/matmul/utils/matmul_utils.h"
#include "impl/adv_api/detail/matmul/resource/cube_in_buffer/cube_in_buffer_utils.h"

namespace AscendC {
namespace {
class TestCubeInBufferUtilsMx : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
    bool TestUseQue()
    {
        Impl::Detail::CubeInQueType<MatmulInputAType<A_TYPE, typename A_TYPE::T>>::QUE qid1;
        Impl::Detail::CubeInQueType<MatmulInputBType<B_TYPE, typename B_TYPE::T>>::QUE qid2;
        return true;
    }

private:
    using A_TYPE = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp8_e4m3fn_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;
    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
};
}

TEST_F(TestCubeInBufferUtilsMx, test_mx_cube_in_buffer_utils_que_mx) {
    EXPECT_TRUE(TestUseQue());
}
}  // namespace AscendC
