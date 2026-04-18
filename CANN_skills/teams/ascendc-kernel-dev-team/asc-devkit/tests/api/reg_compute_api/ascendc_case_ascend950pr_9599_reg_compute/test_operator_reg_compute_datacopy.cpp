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

using Reg::DataCopyMode;

template <typename T, int mode>
class KernelMicroDataCopy {
public:
    __aicore__ inline KernelMicroDataCopy() {}
    __aicore__ inline void Init(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t count)
    {
        const int alginSize = static_cast<int>(32 / sizeof(T));
        dstSize = (count + alginSize - 1) / alginSize * alginSize;
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
        Reg::RegTensor<T> vreg0;
        uint32_t sreg = static_cast<uint32_t>(dstSize);
        Reg::MaskReg preg;
        constexpr uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
        uint16_t repeatTimes = CeilDivision(dstSize, sregLower);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            preg = Reg::UpdateMask<T>(sreg);
            Reg::DataCopy(vreg0, src + i * sregLower);
            Reg::DataCopy(dst + i * sregLower, vreg0, preg);
        }
    }

    __aicore__ inline void ComputeMode1(__ubuf__ T *dst, __ubuf__ T *src)
    {
        Reg::RegTensor<T> vreg0;
        uint32_t sreg = static_cast<uint32_t>(dstSize);
        Reg::MaskReg preg;
        constexpr uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
        uint16_t repeatTimes = CeilDivision(dstSize, sregLower);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            preg = Reg::UpdateMask<T>(sreg);
            Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(vreg0, src, sregLower);
            Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(dst, vreg0, sregLower, preg);
        }
    }

    __aicore__ inline void ComputeMode2(__ubuf__ T *dst, __ubuf__ T *src)
    {
        Reg::RegTensor<T> vreg0;
        uint32_t sreg = static_cast<uint32_t>(dstSize);
        Reg::MaskReg preg;
        constexpr uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
        uint16_t repeatTimes = CeilDivision(dstSize, sregLower);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            preg = Reg::UpdateMask<T>(sreg);
            auto aReg = Reg::CreateAddrReg<T>(sregLower);
            Reg::DataCopy(vreg0, src + i * sregLower);
            Reg::DataCopy(dst, vreg0, aReg, preg);
        }
    }

    __aicore__ inline void ComputeMode3(__ubuf__ T *dst, __ubuf__ T *src)
    {
        Reg::RegTensor<T> vreg0;
        uint32_t sreg = static_cast<uint32_t>(dstSize);
        Reg::MaskReg preg;
        constexpr uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
        uint16_t repeatTimes = CeilDivision(dstSize, sregLower);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            preg = Reg::UpdateMask<T>(sreg);
            auto aReg = Reg::CreateAddrReg<T>(sregLower);
            Reg::DataCopy(vreg0, src, aReg);
            Reg::DataCopy(dst, vreg0, aReg, preg);
        }
    }

    __aicore__ inline void ComputeMode301(__ubuf__ T *dst, __ubuf__ T *src)
    {
        constexpr uint32_t repeatElm = GetVecLen() / sizeof(T) * 2;
        uint16_t repeatTimes = CeilDivision(dstSize, repeatElm);
        Reg::RegTensor<T> reg0;
        Reg::RegTensor<T> reg1;
        Reg::MaskReg mask = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_DINTLV_B8>(reg0, reg1, src + i * repeatElm);
            Reg::DataCopy<T, Reg::StoreDist::DIST_INTLV_B8>(dst + i * repeatElm, reg0, reg1, mask);
        }
    }

    __aicore__ inline void ComputeMode302(__ubuf__ T *dst, __ubuf__ T *src)
    {
        constexpr uint32_t repeatElm = GetVecLen() / sizeof(T) * 2;
        uint16_t repeatTimes = CeilDivision(dstSize, repeatElm);
        Reg::RegTensor<T> reg0;
        Reg::RegTensor<T> reg1;
        Reg::MaskReg mask = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B8>(
                reg0, reg1, src, repeatElm);
            Reg::DataCopy<T, Reg::StoreDist::DIST_INTLV_B8>(dst + i * repeatElm, reg0, reg1, mask);
        }
    }

    __aicore__ inline void ComputeMode4(__ubuf__ T *dst, __ubuf__ T *src)
    {
        Reg::RegTensor<T> vreg0;
        Reg::MaskReg preg = Reg::CreateMask<T>();
        constexpr uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
        uint16_t repeatTimes = CeilDivision(dstSize, sregLower);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            Reg::DataCopy<T, DataCopyMode::DATA_BLOCK_COPY>(vreg0, src + i * sregLower, 1, preg);
            Reg::DataCopy<T, DataCopyMode::DATA_BLOCK_COPY>(dst + i * sregLower, vreg0, 1, preg);
        }
    }

    __aicore__ inline void ComputeMode5(__ubuf__ T *dst, __ubuf__ T *src)
    {
        Reg::RegTensor<T> vreg0;
        Reg::MaskReg preg = Reg::CreateMask<T>();
        constexpr uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
        uint16_t repeatTimes = CeilDivision(dstSize, sregLower);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            Reg::DataCopy<T, DataCopyMode::DATA_BLOCK_COPY>(vreg0, src, 1, preg);
            Reg::DataCopy<T, DataCopyMode::DATA_BLOCK_COPY>(dst, vreg0, 1, preg);
        }
    }

    __aicore__ inline void ComputeMode6(__ubuf__ T *dst, __ubuf__ T *src)
    {
        Reg::RegTensor<T> vreg0;
        Reg::MaskReg preg = Reg::CreateMask<T>();
        constexpr uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
        uint16_t repeatTimes = CeilDivision(dstSize, sregLower);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            Reg::DataCopy<T, DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(vreg0, src, 1, 8, preg);
            Reg::DataCopy<T, DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(dst, vreg0, 1, 8, preg);
        }
    }

    __aicore__ inline void ComputeMode7(__ubuf__ T *dst, __ubuf__ T *src)
    {
        Reg::RegTensor<T> vreg0;
        Reg::UnalignReg u0;
        Reg::UnalignReg u1;
        Reg::MaskReg preg = Reg::CreateMask<T>();
        constexpr uint32_t sregLower = 15;  // to process unalign data
        uint16_t repeatTimes = CeilDivision(dstSize, sregLower);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            auto srcUbT = src + i * sregLower;
            Reg::DataCopyUnAlignPre(u0, srcUbT);
            Reg::LocalMemBar<Reg::MemType::VEC_LOAD, Reg::MemType::VEC_STORE>();
            Reg::DataCopyUnAlign(vreg0, u0, srcUbT);
            Reg::DataCopyUnAlign(dst, vreg0, u1, sregLower);
        }
        Reg::DataCopyUnAlignPost(dst, u1, 0);
    }

    __aicore__ inline void ComputeMode8(__ubuf__ T *dst, __ubuf__ T *src)
    {
        Reg::RegTensor<T> vreg0;
        Reg::UnalignReg u0;
        Reg::UnalignReg u1;
        Reg::MaskReg preg = Reg::CreateMask<T>();
        constexpr uint32_t sregLower = 15;  // to process unalign data
        uint16_t repeatTimes = CeilDivision(dstSize, sregLower);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            Reg::DataCopyUnAlignPre(u0, src);
            Reg::DataCopyUnAlign(vreg0, u0, src, sregLower);
            Reg::DataCopyUnAlign(dst, vreg0, u1, sregLower);
        }
        Reg::DataCopyUnAlignPost(dst, u1, 0);
    }

    __aicore__ inline void ComputeMode81(__ubuf__ T *dst, __ubuf__ T *src)
    {
        Reg::RegTensor<T> vreg0;
        Reg::UnalignReg u0;
        Reg::UnalignReg u1;
        Reg::MaskReg preg = Reg::CreateMask<T>();
        constexpr uint32_t sregLower = 15;  // to process unalign data
        uint16_t repeatTimes = CeilDivision(dstSize, sregLower);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            vldas(u0, src);
            vldus(vreg0, u0, src, sregLower, POST_UPDATE);
            vstus(u1, sregLower, vreg0, dst, POST_UPDATE);
        }
        vstas(u1, dst, 0);
    }

    __aicore__ inline void  ComputeMode9(
        __ubuf__ T *dst, __ubuf__ T *src)
    {
        Reg::MaskReg preg = Reg::CreateMask<T>();
        constexpr uint32_t repeatElm = static_cast<uint32_t>(VECTOR_REG_WIDTH / 8 / sizeof(T));
        uint16_t repeatTimes = CeilDivision(dstSize, repeatElm);
        if constexpr (sizeof(T) == 8) {
            for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
                auto aReg = Reg::CreateAddrReg<uint32_t>(i, repeatElm);
                Reg::DataCopy(preg, src, aReg);
                Reg::DataCopy(dst, preg, aReg);
            }
        } else {
            for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
                auto aReg = Reg::CreateAddrReg<T>(i, repeatElm);
                Reg::DataCopy(preg, src, aReg);
                Reg::DataCopy(dst, preg, aReg);
            }
        }
    }

    __aicore__ inline void ComputeMode10(
        __ubuf__ T *dst, __ubuf__ T *src)
    {
        Reg::RegTensor<T> vreg;
        Reg::MaskReg preg = Reg::CreateMask<T>();
        constexpr uint32_t repeatElm = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
        uint16_t repeatTimes = CeilDivision(dstSize, repeatElm);
        if constexpr (sizeof(T) == 8) {
            for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
                auto aReg = Reg::CreateAddrReg<uint32_t>(i, repeatElm);
                Reg::DataCopy(vreg, src, aReg);
                Reg::DataCopy(dst, vreg, aReg, preg);
            }
        } else {
            for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
                auto aReg = Reg::CreateAddrReg<T>(i, repeatElm);
                Reg::DataCopy(vreg, src, aReg);
                Reg::DataCopy(dst, vreg, aReg, preg);
            }
        }
    }

    __aicore__ inline void ComputeMode11(
        __ubuf__ T *dst, __ubuf__ T *src)
    {
        Reg::RegTensor<T> vreg0, vreg1;
        Reg::MaskReg preg = Reg::CreateMask<T>();
        constexpr uint32_t repeatElm = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
        uint16_t repeatTimes = CeilDivision(dstSize, repeatElm);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            auto aReg = Reg::CreateAddrReg<T>(i, repeatElm);
            Reg::DataCopy<T, Reg::LoadDist::DIST_DINTLV_B8>(vreg0, vreg1, src, aReg);
            Reg::DataCopy<T, Reg::StoreDist::DIST_INTLV_B8>(dst, vreg0, vreg1, aReg, preg);
        }
    }

    __aicore__ inline void ComputeMode12(
        __ubuf__ T *dst, __ubuf__ T *src)
    {
        Reg::RegTensor<T> vreg;
        Reg::MaskReg preg = Reg::CreateMask<T>();
        Reg::UnalignReg u0, u1;
        Reg::AddrReg aReg0;
        constexpr uint32_t repeatElm = 15;  // to process unalign data
        uint16_t repeatTimes = CeilDivision(dstSize, repeatElm);
        if constexpr (sizeof(T) == 8) {
            for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
                aReg0 = Reg::CreateAddrReg<uint32_t>(i, repeatElm);
                Reg::DataCopyUnAlignPre(u0, src, aReg0);
                Reg::DataCopyUnAlign<T>(vreg, u0, src, aReg0, 0);
                Reg::DataCopyUnAlign(dst, vreg, u1, aReg0);
            }
        } else {
            for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
                aReg0 = Reg::CreateAddrReg<T>(i, repeatElm);
                Reg::DataCopyUnAlignPre(u0, src, aReg0);
                Reg::DataCopyUnAlign<T>(vreg, u0, src, aReg0, 0);
                Reg::DataCopyUnAlign(dst, vreg, u1, aReg0);
            }
        }
        Reg::DataCopyUnAlignPost(dst, u1, aReg0);
    }

    __aicore__ inline void ComputeMode13(__ubuf__ T *dst, __ubuf__ T *src)
    {
        Reg::RegTensor<T> vreg0;
        constexpr uint32_t count = 15;
        uint16_t repeatTimes = CeilDivision(dstSize, count);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            Reg::Load(vreg0, src + i * count);
            Reg::Store(dst + i * count, vreg0, count);
        }
    }

    __aicore__ inline void ComputeMode14(__ubuf__ T *dst, __ubuf__ T *src)
    {
        Reg::RegTensor<T> vreg0;
        constexpr uint32_t count = GetVecLen() / sizeof(T);
        uint16_t repeatTimes = CeilDivision(dstSize, count);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            Reg::Load(vreg0, src + i * count);
            Reg::Store(dst + i * count, vreg0);
        }
    }

    __aicore__ inline void Compute()
    {
        // test other api
        Nop<0>();
        Nop<10>();
        constexpr uint32_t dataBlockSize = GetDataBlockSizeInBytes();
        EXPECT_EQ(dataBlockSize, 32);
        constexpr uint32_t ubSize = GetUBSizeInBytes();
        EXPECT_EQ(ubSize, 248 * 1024);
        constexpr uint32_t vecLen = GetVecLen();
        EXPECT_EQ(vecLen, 256);
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();

        __ubuf__ T *src = (__ubuf__ T *)srcLocal.GetPhyAddr();
        __ubuf__ T *dst = (__ubuf__ T *)dstLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            if constexpr (mode == 0) {
                ComputeMode0(dst, src);
            } else if constexpr (mode == 1) {
                ComputeMode1(dst, src);
            } else if constexpr (mode == 2) {
                ComputeMode2(dst, src);
            } else if constexpr (mode == 3) {
                ComputeMode3(dst, src);
            } else if constexpr (mode == 301) {
                ComputeMode301(dst, src);
            } else if constexpr (mode == 302) {
                ComputeMode302(dst, src);
            } else if constexpr (mode == 4) {
                ComputeMode4(dst, src);
            } else if constexpr (mode == 5) {
                ComputeMode5(dst, src);
            } else if constexpr (mode == 6) {
                ComputeMode6(dst, src);
            } else if constexpr (mode == 7) {
                ComputeMode7(dst, src);
            } else if constexpr (mode == 8) {
                ComputeMode8(dst, src);
            } else if constexpr (mode == 81) {
                ComputeMode81(dst, src);
            } else if constexpr (mode == 9) {
                ComputeMode9(dst, src);
            } else if constexpr (mode == 10) {
                ComputeMode10(dst, src);
            } else if constexpr (mode == 11) {
                ComputeMode11(dst, src);
            } else if constexpr (mode == 12) {
                ComputeMode12(dst, src);
            } else if constexpr (mode == 13) {
                ComputeMode13(dst, src);
            } else if constexpr (mode == 14) {
                ComputeMode14(dst, src);
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

template <typename T, int mode>
__global__ __aicore__ void MicroDatacopy(uint8_t *dstGm, uint8_t *srcGm, uint32_t size)
{
    KernelMicroDataCopy<T, mode> op;
    op.Init(srcGm, dstGm, size);
    op.Process();
}

template <int dim>
struct MicroDatacopyModeTestParams {
    void (*cal_func)(uint8_t *, uint8_t *, uint32_t);
    uint32_t size;
};

template <int mode>
class MicroDatacopyTestsuite
    : public testing::Test
    , public testing::WithParamInterface<MicroDatacopyModeTestParams<mode>> {
};

#define MICRO_DATACOPY_TEST_CASE(mode)                                                             \
    using MicroDatacopyTestsuite_mode##mode = MicroDatacopyTestsuite<mode>;                        \
    INSTANTIATE_TEST_CASE_P(TEST_MicroDatacopy, MicroDatacopyTestsuite_mode##mode,                 \
        ::testing::Values(MicroDatacopyModeTestParams<mode>{ MicroDatacopy<uint8_t, mode>, 1024 }, \
        MicroDatacopyModeTestParams<mode>{ MicroDatacopy<int8_t, mode>, 1024 },                    \
        MicroDatacopyModeTestParams<mode>{ MicroDatacopy<uint16_t, mode>, 1024 },                  \
        MicroDatacopyModeTestParams<mode>{ MicroDatacopy<int16_t, mode>, 1024 },                   \
        MicroDatacopyModeTestParams<mode>{ MicroDatacopy<uint32_t, mode>, 1024 },                  \
        MicroDatacopyModeTestParams<mode>{ MicroDatacopy<int32_t, mode>, 1024 },                   \
        MicroDatacopyModeTestParams<mode>{ MicroDatacopy<float, mode>, 1024 },                     \
        MicroDatacopyModeTestParams<mode>{ MicroDatacopy<bfloat16_t, mode>, 1024 },                \
        MicroDatacopyModeTestParams<mode>{ MicroDatacopy<half, mode>, 1024 },                      \
        MicroDatacopyModeTestParams<mode>{ MicroDatacopy<bool, mode>, 1024 }));                    \
                                                                                                   \
    TEST_P(MicroDatacopyTestsuite_mode##mode, MicroDatacopyTestCase)                               \
    {                                                                                              \
        auto param = GetParam();                                                                   \
        uint8_t srcGm[param.size] = { 0 };                                                         \
        uint8_t dstGm[param.size] = { 0 };                                                         \
        param.cal_func(dstGm, srcGm, param.size);                                                  \
        for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {                         \
            EXPECT_EQ(dstGm[i], 0x00);                                                             \
        }                                                                                          \
    }

MICRO_DATACOPY_TEST_CASE(0);
MICRO_DATACOPY_TEST_CASE(1);
MICRO_DATACOPY_TEST_CASE(301);
MICRO_DATACOPY_TEST_CASE(302);
MICRO_DATACOPY_TEST_CASE(4);
MICRO_DATACOPY_TEST_CASE(5);
MICRO_DATACOPY_TEST_CASE(6);
MICRO_DATACOPY_TEST_CASE(8);
MICRO_DATACOPY_TEST_CASE(9);
MICRO_DATACOPY_TEST_CASE(10);
MICRO_DATACOPY_TEST_CASE(11);
MICRO_DATACOPY_TEST_CASE(12);
MICRO_DATACOPY_TEST_CASE(13);
MICRO_DATACOPY_TEST_CASE(14);

#define MICRO_DATACOPY_TEST_CASE_B64(mode)                                                        \
    using MicroDatacopyTestsuite_mode_b64##mode = MicroDatacopyTestsuite<mode>;                   \
    INSTANTIATE_TEST_CASE_P(TEST_MicroDatacopyB64,                                                \
        MicroDatacopyTestsuite_mode_b64##mode,                                                    \
        ::testing::Values(MicroDatacopyModeTestParams<mode>{MicroDatacopy<uint64_t, mode>, 1024}, \
            MicroDatacopyModeTestParams<mode>{MicroDatacopy<int64_t, mode>, 1024}));              \
                                                                                                  \
    TEST_P(MicroDatacopyTestsuite_mode_b64##mode, MicroDatacopyTestCase)                          \
    {                                                                                             \
        auto param = GetParam();                                                                  \
        uint8_t srcGm[param.size] = {0};                                                          \
        uint8_t dstGm[param.size] = {0};                                                          \
        param.cal_func(dstGm, srcGm, param.size);                                                 \
        for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {                        \
            EXPECT_EQ(dstGm[i], 0x00);                                                            \
        }                                                                                         \
    }

MICRO_DATACOPY_TEST_CASE_B64(0);
MICRO_DATACOPY_TEST_CASE_B64(1);
MICRO_DATACOPY_TEST_CASE_B64(8);
MICRO_DATACOPY_TEST_CASE_B64(81);
MICRO_DATACOPY_TEST_CASE_B64(9);
MICRO_DATACOPY_TEST_CASE_B64(10);
MICRO_DATACOPY_TEST_CASE_B64(12);
