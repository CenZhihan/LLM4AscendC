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
 * \file test_matmul_iterator.cpp
 * \brief matmul iterator ut for norm
 */

#include <gtest/gtest.h>
#include "kernel_operator.h"
#include "include/adv_api/matmul/tiling.h"
#include "impl/adv_api/detail/matmul/utils/matmul_call_back.h"
#include "impl/adv_api/detail/matmul/utils/matmul_module.h"
#include "impl/adv_api/detail/matmul/policy/matmul_policy.h"
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"

using namespace std;
using namespace AscendC;


using A_TYPE = MatmulType<AscendC::TPosition::TSCM, CubeFormat::ND, half>;
using B_TYPE = MatmulType<AscendC::TPosition::TSCM, CubeFormat::ND, half>;
using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

template <typename IMPL, class INPUT_TYPE, const auto& MM_CFG>
class CustomCopyCubeIn {
public:
    void Reset() {
        clearedCount++;
    }

public:
    uint32_t clearedCount {0};
};

namespace {
template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using CopyCubeInA = CustomCopyCubeIn<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = CustomCopyCubeIn<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>, MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl :
    MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling),
    MATMUL_IMPORT_MODULE(CopyCubeInA),
    MATMUL_IMPORT_MODULE(CopyCubeInB),
    MATMUL_IMPORT_MODULE_PRIVATE(MLoop),
    MATMUL_IMPORT_MODULE_PRIVATE(NLoop),
    MATMUL_IMPORT_MODULE(Scheduler)
{
    using VAR_PARAMS =
        typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;

    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);
    MATMUL_ALLOW_USING(CopyCubeInA);
    MATMUL_ALLOW_USING(CopyCubeInB);
    MATMUL_ALLOW_USING_PRIVATE(MLoop);
    MATMUL_ALLOW_USING_PRIVATE(NLoop);
    MATMUL_ALLOW_USING(Scheduler);

public:
    using Scheduler::MoveNext;
    using Scheduler::Reset;
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);

public:
    MatmulImpl() {
        InitVar();
    }

    void SetTiling(IterateOrder order, int32_t stepM, uint32_t stepN, int32_t baseM, int32_t baseN, int32_t singleM, int32_t singleN) {
        tiling.iterateOrder = static_cast<int32_t>(order);
        tiling.stepM = stepM;
        tiling.stepN = stepN;
        tiling.baseM = baseM;
        tiling.baseN = baseN;
        MATMUL_MODULE(MLoop)->Init(singleM);
        MATMUL_MODULE(NLoop)->Init(singleN);

        this->Reset();
    }

    VAR_PARAMS& GetVar() {
        return var;
    }

    void InitVar() {
        MATMUL_MODULE(MatmulShapeTiling)->SetTiling(&tiling);
        var.tpipe_ = &pipe;
    }

    uint32_t GetMLoopInnerIter() {
        return MATMUL_MODULE(MLoop)->GetInnerIter();
    }

    uint32_t GetNLoopInnerIter() {
        return MATMUL_MODULE(NLoop)->GetInnerIter();
    }

    uint32_t GetMLoopInnerIdx() {
        return MATMUL_MODULE(MLoop)->GetInnerIdx();
    }

    uint32_t GetNLoopInnerIdx() {
        return MATMUL_MODULE(NLoop)->GetInnerIdx();
    }

private:
    TCubeTiling tiling;
    TPipe pipe;
    VAR_PARAMS var;
};
}

class test_matmul_iterator : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>,
               CustomMatmulPolicy>
        mm;
};

TEST_F(test_matmul_iterator, first_iter_order_M) {
    mm.SetTiling(IterateOrder::ORDER_M, 4, 2, 128, 128, 512, 256);
    
    bool isFinished = mm.MoveNext();

    ASSERT_TRUE(isFinished);
    ASSERT_EQ(mm.GetMLoopInnerIter(), 4);
    ASSERT_EQ(mm.GetMLoopInnerIdx(), 0);
    ASSERT_EQ(mm.GetNLoopInnerIdx(), 0);
}

TEST_F(test_matmul_iterator, first_iter_order_N) {
    mm.SetTiling(IterateOrder::ORDER_N, 4, 2, 128, 128, 512, 256);
    
    bool isFinished = mm.MoveNext();

    ASSERT_TRUE(isFinished);

    ASSERT_EQ(mm.GetNLoopInnerIter(), 2);
    ASSERT_EQ(mm.GetNLoopInnerIdx(), 0);
}

TEST_F(test_matmul_iterator, order_M_iter_four_times) {
    mm.SetTiling(IterateOrder::ORDER_M, 4, 2, 128, 128, 512, 256);

    int32_t cnt = 0;
    while(mm.MoveNext()) {
        cnt++;
    }

    ASSERT_EQ(cnt, 8);
}

TEST_F(test_matmul_iterator, order_N_iter_four_times) {
    mm.SetTiling(IterateOrder::ORDER_N, 4, 2, 128, 128, 512, 256);

    int32_t cnt = 0;
    while(mm.MoveNext()) {
        cnt++;
    }

    ASSERT_EQ(cnt, 8);
}


TEST_F(test_matmul_iterator, order_M_iter_twice) {
    mm.SetTiling(IterateOrder::ORDER_M, 4, 2, 128, 128, 128, 256);

    auto isFinished = mm.MoveNext();
    ASSERT_EQ(mm.GetNLoopInnerIdx(), 0);
    isFinished = mm.MoveNext();
    ASSERT_EQ(mm.GetNLoopInnerIdx(), 1);
    ASSERT_TRUE(isFinished);
    isFinished = mm.MoveNext();
    ASSERT_FALSE(isFinished);
    ASSERT_EQ(mm.GetMLoopInnerIdx(), 0);
}

TEST_F(test_matmul_iterator, order_N_iter_twice) {
    mm.SetTiling(IterateOrder::ORDER_N, 4, 2, 128, 128, 256, 128);

    auto isFinished = mm.MoveNext();
    ASSERT_EQ(mm.GetMLoopInnerIdx(), 0);
    isFinished = mm.MoveNext();
    ASSERT_EQ(mm.GetMLoopInnerIdx(), 1);
    ASSERT_TRUE(isFinished);
    isFinished = mm.MoveNext();
    ASSERT_FALSE(isFinished);
    ASSERT_EQ(mm.GetNLoopInnerIdx(), 0);
}

// test when n-dimension is finished in OrderM case
TEST_F(test_matmul_iterator, order_M_n_is_finished) {
    mm.SetTiling(IterateOrder::ORDER_M, 4, 2, 128, 128, 256, 256);

    // first iter
    auto isFinished = mm.MoveNext();
    ASSERT_EQ(mm.GetMLoopInnerIdx(), 0);
    // n-dimension is finished
    isFinished = mm.MoveNext();
    ASSERT_EQ(mm.GetNLoopInnerIdx(), 1);
    ASSERT_TRUE(isFinished);
    (void)mm.MoveNext();
    ASSERT_EQ(mm.GetNLoopInnerIdx(), 0);
    ASSERT_EQ(mm.GetMLoopInnerIdx(), 1);
}

// test when m-dimension is finished in OrderN case
TEST_F(test_matmul_iterator, order_N_m_is_finished) {
    mm.SetTiling(IterateOrder::ORDER_N, 4, 2, 128, 128, 256, 256);
    // first iter
    auto isFinished = mm.MoveNext();
    ASSERT_EQ(mm.GetNLoopInnerIdx(), 0);
    // n-dimension is finished
    isFinished = mm.MoveNext();
    ASSERT_EQ(mm.GetMLoopInnerIdx(), 1);
    ASSERT_TRUE(isFinished);
    (void)mm.MoveNext();
    ASSERT_EQ(mm.GetMLoopInnerIdx(), 0);
    ASSERT_EQ(mm.GetNLoopInnerIdx(), 1);
}
