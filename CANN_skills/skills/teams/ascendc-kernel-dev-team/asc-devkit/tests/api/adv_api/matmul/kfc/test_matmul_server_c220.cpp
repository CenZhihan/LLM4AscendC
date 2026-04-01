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
#include "impl/adv_api/detail/matmul/kfc/matmul_server_impl.h"
#include "impl/adv_api/detail/matmul/kfc/matmul_server_impl_c220.h"
#include "kfc_fake_modules.h"
#include "../copy_cube_in/base_tiling_struct.h"

// UT not support multi thread, only aic or aiv thread, func that contain aic and aiv can not simulated
using namespace std;

namespace AscendC {
constexpr uint32_t NUM_FIFTYSIX = 56;
template <const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using Scheduler = CustomMatmulScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG>;
    using C1Buffer = Impl::Detail::C1Buffer<IMPL, A_TYPE, BIAS_TYPE, MM_CFG>;
};

class TestMatmulServerC220 : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using A_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;

    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, true>;

    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
};

TEST_F(TestMatmulServerC220, ProcessEnd) {
    MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM,
        MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mmServer;
    KfcMsg kfcMsg;
    uint8_t cGM[2048] = {0};
    kfcMsg.body.cAddr = (uint64_t)(cGM);
    auto funID = KFC_Enum::MMFUN_END;
    mmServer.Process(&kfcMsg, funID);
}
} // namespace AscendC
