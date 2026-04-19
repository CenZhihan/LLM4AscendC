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

using namespace std;
using namespace AscendC;

using Reg::CreateMask;
using Reg::MaskReg;
using Reg::RegTensor;
using Reg::UpdateMask;

template <typename T, int Mode, int Count, int TraitNum> class KernelMicroDataCopyB64 {
public:
    __aicore__ inline KernelMicroDataCopyB64() {}
    __aicore__ inline void Init(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t totalNum)
    {
        const int alginSize = 32 / sizeof(T);
        dstSize = (totalNum + alginSize - 1) / alginSize * alginSize;
        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm), dstSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstGm), dstSize);
        pipe.InitBuffer(inQueue, 1, dstSize * sizeof(T));
        pipe.InitBuffer(outQueue, 1, dstSize * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<T> srcLocal = inQueue.AllocTensor<T>();
        DataCopy(srcLocal, srcGlobal, dstSize);
        inQueue.EnQue<T>(srcLocal);
    }

    __aicore__ inline void ComputeMode0(__ubuf__ T *dst, __ubuf__ T *src)
    {
        // Pat { ALL, VL1, VL2, VL3, VL4, VL8, VL16, VL32, VL64, VL128, M3, M4, H, Q, ALLF = 15 }
        if constexpr (TraitNum == 1) {
            RegTensor<T, Reg::RegTraitNumOne> vreg0;
            uint32_t sregLower = 32;
            MaskReg preg =
                CreateMask<T, static_cast<Reg::MaskPattern>(Mode), Reg::RegTraitNumOne>();
            for (uint16_t i = 0; i < (uint16_t)1; ++i) {
                DataCopy(vreg0, src + i * sregLower);
                DataCopy(dst + i * sregLower, vreg0, preg);
            }
        } else {
            RegTensor<T, Reg::RegTraitNumTwo> vreg0;
            uint32_t sregLower = 64;
            MaskReg preg =
                CreateMask<T, static_cast<Reg::MaskPattern>(Mode), Reg::RegTraitNumTwo>();
            for (uint16_t i = 0; i < (uint16_t)1; ++i) {
                DataCopy(vreg0, src + i * sregLower);
                DataCopy(dst + i * sregLower, vreg0, preg);
            }
        }
    }

    __aicore__ inline void ComputeMode1(__ubuf__ T *dst, __ubuf__ T *src)
    {
        uint32_t sreg = (uint32_t)Count;
        if constexpr (TraitNum == 1) {
            RegTensor<T, Reg::RegTraitNumOne> vreg0;
            uint32_t sregLower = 32;
            uint16_t repeatTimes = CeilDivision(dstSize, sregLower);
            MaskReg preg;
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                preg = UpdateMask<T, Reg::RegTraitNumOne>(sreg);
                DataCopy(vreg0, src + i * sregLower);
                DataCopy(dst + i * sregLower, vreg0, preg);
            }
        } else {
            RegTensor<T, Reg::RegTraitNumTwo> vreg0;
            uint32_t sregLower = 64;
            uint16_t repeatTimes = CeilDivision(dstSize, sregLower);
            MaskReg preg;
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                preg = UpdateMask<T, Reg::RegTraitNumTwo>(sreg);
                DataCopy(vreg0, src + i * sregLower);
                DataCopy(dst + i * sregLower, vreg0, preg);
            }
        }
    }

    __aicore__ inline void ComputeMode2(__ubuf__ T *dst, __ubuf__ T *src)
    {
        uint32_t sreg = (uint32_t)Count;
        if constexpr (TraitNum == 1) {
            RegTensor<T, Reg::RegTraitNumOne> vreg0;
            uint32_t sregLower = 32;
            uint16_t repeatTimes = CeilDivision(dstSize, sregLower);
            MaskReg preg;
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                preg = UpdateMask<T, Reg::RegTraitNumOne>(sreg);
                DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(vreg0, src, sregLower);
                DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(dst, vreg0, sregLower, preg);
            }
        } else {
            RegTensor<T, Reg::RegTraitNumTwo> vreg0;
            uint32_t sregLower = 64;
            uint16_t repeatTimes = CeilDivision(dstSize, sregLower);
            MaskReg preg;
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                preg = UpdateMask<T, Reg::RegTraitNumTwo>(sreg);
                DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(vreg0, src, sregLower);
                DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(dst, vreg0, sregLower, preg);
            }
        }
    }

    __aicore__ inline void ComputeMode4(__ubuf__ T *dst, __ubuf__ T *src)
    {
        uint32_t sreg = (uint32_t)Count;
        if constexpr (TraitNum == 1) {
            RegTensor<T, Reg::RegTraitNumOne> vreg0;
            uint32_t sregLower = 32;
            uint16_t repeatTimes = CeilDivision(dstSize, sregLower);
            MaskReg preg;
            Reg::UnalignReg u0;
            Reg::UnalignReg u1;
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                preg = UpdateMask<T, Reg::RegTraitNumOne>(sreg);
                Reg::DataCopyUnAlignPre(u0, src);
                Reg::DataCopyUnAlign(vreg0, u0, src, sregLower);
                Reg::DataCopyUnAlign(dst, vreg0, u1, sregLower);
            }
            Reg::DataCopyUnAlignPost(dst, u1, 0);
        } else {
            RegTensor<T, Reg::RegTraitNumTwo> vreg0;
            uint32_t sregLower = 64;
            uint16_t repeatTimes = CeilDivision(dstSize, sregLower);
            MaskReg preg;
            Reg::UnalignReg u0;
            Reg::UnalignReg u1;
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                preg = UpdateMask<T, Reg::RegTraitNumTwo>(sreg);
                Reg::DataCopyUnAlignPre(u0, src);
                Reg::DataCopyUnAlign(vreg0, u0, src, sregLower);
                Reg::DataCopyUnAlign(dst, vreg0, u1, sregLower);
            }
            Reg::DataCopyUnAlignPost(dst, u1, 0);
        }
    }

    __aicore__ inline void ComputeMode5(__ubuf__ T *dst, __ubuf__ T *src)
    {
        constexpr uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T) / 8);
        uint16_t repeatTimes = CeilDivision(dstSize, sregLower);
        __VEC_SCOPE__
        {
            MaskReg preg0;
            int32_t sreg = (int32_t)dstSize;
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                Reg::DataCopy(preg0, src + i * sregLower);
                Reg::DataCopy(dst + i * sregLower, preg0);
            }
        }
    }

    __aicore__ inline void ComputeMode6(__ubuf__ T *dst, __ubuf__ T *src)
    {
        constexpr uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / 8);
        uint16_t repeatTimes = CeilDivision(dstSize * sizeof(T), sregLower);
        __VEC_SCOPE__
        {
            MaskReg preg0;
            int32_t sreg = (int32_t)sregLower;
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE,
                    Reg::MaskDist::DIST_NORM>(preg0, src, sreg);
                Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE,
                    Reg::MaskDist::DIST_NORM>(dst, preg0, sreg);
            }
        }
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        LocalTensor<uint32_t> tmp = dstLocal.template ReinterpretCast<uint32_t>();
        Duplicate(tmp, (uint32_t)0, dstSize * 2);
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();

        __ubuf__ T *src = (__ubuf__ T *)srcLocal[0].GetPhyAddr();
        __ubuf__ T *dst = (__ubuf__ T *)dstLocal[0].GetPhyAddr();
        __VEC_SCOPE__
        {
            if constexpr (Mode < 100) {
                ComputeMode0(dst, src);
            } else if constexpr (Mode == 200) {
                ComputeMode1(dst, src);
            } else if constexpr (Mode == 300) {
                ComputeMode2(dst, src);
            } else if constexpr (Mode == 500) {
                __ubuf__ T *src1 = (__ubuf__ T *)srcLocal[dstSize - Count].GetPhyAddr();
                __ubuf__ T *dst1 = (__ubuf__ T *)dstLocal[dstSize - Count].GetPhyAddr();
                ComputeMode4(dst1, src1);
            } else if constexpr (Mode == 600) {
                ComputeMode5(dst, src);
            } else if constexpr (Mode == 700) {
                ComputeMode6(dst, src);
            }
        }

        outQueue.EnQue<T>(dstLocal);
        inQueue.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = outQueue.DeQue<T>();
        DataCopy(dstGlobal, dstLocal, dstSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueue;
    TQue<TPosition::VECOUT, 1> outQueue;
    uint32_t dstSize;
};

template <typename T, int Mode, int Count, int TraitNum>
__global__ __aicore__ void MicroDatacopyB64(uint8_t *dstGm, uint8_t *srcGm, uint32_t size)
{
    KernelMicroDataCopyB64<T, Mode, Count, TraitNum> op;
    op.Init(srcGm, dstGm, size);
    op.Process();
}

template <int dim, int Count, int TraitNum> struct MicroDatacopyModeB64TestParams {
    void (*cal_func)(uint8_t *, uint8_t *, uint32_t);
    uint32_t size;
};

template <int Mode, int Count, int TraitNum>
class MicroDatacopyB64Testsuite : public testing::Test,
    public testing::WithParamInterface<MicroDatacopyModeB64TestParams<Mode, Count, TraitNum>> {};

#define MICRO_DATACOPY_B64_TEST_CASE(MODE, COUNT, TRAIT_NUM)                                                          \
    using MicroDatacopyB64Testsuite_mode##MODE##COUNT##TRAIT_NUM = MicroDatacopyB64Testsuite<MODE, COUNT, TRAIT_NUM>; \
    INSTANTIATE_TEST_CASE_P(TEST_MicroDatacopyB64, MicroDatacopyB64Testsuite_mode##MODE##COUNT##TRAIT_NUM,            \
        ::testing::Values(                                                                                            \
        MicroDatacopyModeB64TestParams<MODE, COUNT, TRAIT_NUM>{ MicroDatacopyB64<uint64_t, MODE, COUNT, TRAIT_NUM>,   \
        128 },                                                                                                        \
        MicroDatacopyModeB64TestParams<MODE, COUNT, TRAIT_NUM>{ MicroDatacopyB64<int64_t, MODE, COUNT, TRAIT_NUM>,    \
        128 }));                                                                                                      \
                                                                                                                      \
    TEST_P(MicroDatacopyB64Testsuite_mode##MODE##COUNT##TRAIT_NUM, MicroDatacopyB64TestCase)                          \
    {                                                                                                                 \
        auto param = GetParam();                                                                  \
        uint8_t* srcGm = new uint8_t[param.size * 4 * 10]{0};                                      \
        uint8_t* dstGm = new uint8_t[param.size * 4 * 10]{0};                                      \
        param.cal_func(dstGm, srcGm, param.size);                                                 \
        for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {                        \
            EXPECT_EQ(dstGm[i], 0x00);                                                            \
        }                                                                                         \
        delete [] srcGm;                                                                          \
        delete [] dstGm;                                                                          \
    }

MICRO_DATACOPY_B64_TEST_CASE(0, 128, 1);
MICRO_DATACOPY_B64_TEST_CASE(1, 128, 1);
MICRO_DATACOPY_B64_TEST_CASE(2, 128, 1);
MICRO_DATACOPY_B64_TEST_CASE(3, 128, 1);
MICRO_DATACOPY_B64_TEST_CASE(4, 128, 1);
MICRO_DATACOPY_B64_TEST_CASE(5, 128, 1);
MICRO_DATACOPY_B64_TEST_CASE(6, 128, 1);
MICRO_DATACOPY_B64_TEST_CASE(7, 128, 1);
MICRO_DATACOPY_B64_TEST_CASE(8, 128, 1);
MICRO_DATACOPY_B64_TEST_CASE(9, 128, 1);
MICRO_DATACOPY_B64_TEST_CASE(10, 128, 1);
MICRO_DATACOPY_B64_TEST_CASE(11, 128, 1);
MICRO_DATACOPY_B64_TEST_CASE(12, 128, 1);
MICRO_DATACOPY_B64_TEST_CASE(13, 128, 1);
MICRO_DATACOPY_B64_TEST_CASE(15, 128, 1);
MICRO_DATACOPY_B64_TEST_CASE(200, 128, 1);
MICRO_DATACOPY_B64_TEST_CASE(300, 128, 1);
MICRO_DATACOPY_B64_TEST_CASE(500, 128, 1);
MICRO_DATACOPY_B64_TEST_CASE(600, 128, 1);
MICRO_DATACOPY_B64_TEST_CASE(700, 128, 1);
MICRO_DATACOPY_B64_TEST_CASE(0, 128, 2);
MICRO_DATACOPY_B64_TEST_CASE(1, 128, 2);
MICRO_DATACOPY_B64_TEST_CASE(2, 128, 2);
MICRO_DATACOPY_B64_TEST_CASE(3, 128, 2);
MICRO_DATACOPY_B64_TEST_CASE(4, 128, 2);
MICRO_DATACOPY_B64_TEST_CASE(5, 128, 2);
MICRO_DATACOPY_B64_TEST_CASE(6, 128, 2);
MICRO_DATACOPY_B64_TEST_CASE(7, 128, 2);
MICRO_DATACOPY_B64_TEST_CASE(8, 128, 2);
MICRO_DATACOPY_B64_TEST_CASE(9, 128, 2);
MICRO_DATACOPY_B64_TEST_CASE(10, 128, 2);
MICRO_DATACOPY_B64_TEST_CASE(11, 128, 2);
MICRO_DATACOPY_B64_TEST_CASE(12, 128, 2);
MICRO_DATACOPY_B64_TEST_CASE(13, 128, 2);
MICRO_DATACOPY_B64_TEST_CASE(15, 128, 2);
MICRO_DATACOPY_B64_TEST_CASE(200, 128, 2);
MICRO_DATACOPY_B64_TEST_CASE(300, 128, 2);
MICRO_DATACOPY_B64_TEST_CASE(500, 128, 2);
MICRO_DATACOPY_B64_TEST_CASE(600, 128, 2);
MICRO_DATACOPY_B64_TEST_CASE(700, 128, 2);