/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file test_quant_processor_fixpipe.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "kernel_operator.h"
#include "include/adv_api/matmul/tiling.h"
#include "impl/adv_api/detail/matmul/utils/matmul_param.h"
#include "impl/adv_api/detail/matmul/policy/matmul_policy.h"
#define private public
#define protected public
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"
#include "impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_fixpipe.h"

using namespace std;
using namespace AscendC;

namespace {
const uint64_t NUM_SIXTEEN = 16;
template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using KLoop = Impl::Detail::MatmulQuantProcessor<IMPL, A_TYPE, C_TYPE, MM_CFG>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG, class MM_CB,
MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE(KLoop) {
    MATMUL_ALLOW_USING(KLoop);

public:
    MatmulImpl() {};
    TPipe pipe;

};

class TestMatmulQuantProcessorFixPipe : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using A_TYPE = MatmulType<AscendC::TPosition::TSCM, CubeFormat::ND, fp8_e5m2_t>;
    using B_TYPE = MatmulType<AscendC::TPosition::TSCM, CubeFormat::ND, int8_t>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM, void, CustomMatmulPolicy> mm;
};

TEST_F(TestMatmulQuantProcessorFixPipe, Quant_Processor_Fixpipe){
    mm.Init(NUM_SIXTEEN);
    const GlobalTensor<uint64_t> quantTensor1;
    mm.SetQuantVector(quantTensor1);
    LocalTensor<uint64_t> quantTensor2;
    mm.CopyQuantTensor(quantTensor2, NUM_SIXTEEN, NUM_SIXTEEN);
    mm.GetMatmulQuantMode();
    mm.IsQuantSenario();
    mm.SetQuantScalar(NUM_SIXTEEN);
    mm.GetQuantScalarValue();
    mm.UpdateQuantTensor(NUM_SIXTEEN);
    mm.IsPerChannelSenario();
    mm.FreeQuantTensor(quantTensor2);
    mm.Destroy();
}
} //namespace
