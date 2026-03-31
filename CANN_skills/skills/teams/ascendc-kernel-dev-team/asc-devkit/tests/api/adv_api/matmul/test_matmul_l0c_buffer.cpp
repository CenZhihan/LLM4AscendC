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
#include "impl/adv_api/detail/matmul/utils/matmul_param.h"
#include "impl/adv_api/detail/matmul/policy/matmul_policy.h"
#include "impl/adv_api/detail/matmul/resource/cube_out_buffer/cube_out_buffer.h"
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"

using namespace std;
using namespace AscendC;


namespace {

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG, class MM_CB,
MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE(CubeOutBuffer)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
{
    MATMUL_ALLOW_USING(CubeOutBuffer);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);
    using VAR_PARAMS =
        typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;

public:
    using CubeOutBuffer::Destroy;
    using CubeOutBuffer::Init;
    using CubeOutBuffer::AllocTensor;
    using CubeOutBuffer::GetTensor;
    using CubeOutBuffer::EnQue;
    using CubeOutBuffer::DeQue;
    using CubeOutBuffer::FreeTensor;
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    MATMUL_USE_MODULE(MatmulShapeTiling);

public:
    MatmulImpl() {
        InitVar();
    }

    VAR_PARAMS& GetVar() {
        return var;
    }

    void InitVar() {
        MATMUL_MODULE(MatmulShapeTiling)->SetTiling(&tiling);
        var.tpipe_ = &pipe;
        var.baseMN_ = 1024;
    }

private:
    TCubeTiling tiling;
    TPipe pipe;
    VAR_PARAMS var;
};
}

class test_matmul_l0c_buffer : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using L0cT = float;

    using A_TYPE = MatmulType<AscendC::TPosition::TSCM, CubeFormat::ND, half>;
    using B_TYPE = MatmulType<AscendC::TPosition::TSCM, CubeFormat::ND, half>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, L0cT>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM, void> enUnitFlagMM;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL, void> disUnitFlagMM;
};

TEST_F(test_matmul_l0c_buffer, enable_unit_flag) {

    enUnitFlagMM.Init(1024);
    auto co1Local = enUnitFlagMM.AllocTensor();
    auto co1Local1 = enUnitFlagMM.GetTensor();
    enUnitFlagMM.EnQue(co1Local1);
    enUnitFlagMM.DeQue();
    enUnitFlagMM.FreeTensor(co1Local1);
    enUnitFlagMM.Destroy();
}

TEST_F(test_matmul_l0c_buffer, disable_unit_flag) {

    disUnitFlagMM.Init(1024);
    auto co1Local = disUnitFlagMM.AllocTensor();
    auto co1Local1 = disUnitFlagMM.GetTensor();
    disUnitFlagMM.EnQue(co1Local1);
    disUnitFlagMM.DeQue();
    disUnitFlagMM.FreeTensor(co1Local1);
    disUnitFlagMM.Destroy();
}
