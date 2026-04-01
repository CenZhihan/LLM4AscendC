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
#include <type_traits>
#include "kernel_operator.h"

using namespace AscendC;
using Reg::RegTensor;
using Reg::MaskReg;
using Reg::CreateMask;
using Reg::UpdateMask;
using Reg::Cast;
using Reg::MaskMergeMode;
using Reg::RegLayout;
using Reg::SatMode;
using Reg::Truncate;

template <typename ScrT, typename DstT, int32_t mD, MaskMergeMode MzMd, RegLayout RegLayoutMd,
    SatMode SatMd, RoundMode RoundMd>
class KernelCast {
public:
    __aicore__ inline KernelCast()
    {}
    __aicore__ inline void Init(GM_ADDR dstGm, GM_ADDR srcGm, uint32_t dstNums,
                uint32_t srcNums, uint32_t vecMask)
    {
        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ScrT *>(srcGm), srcNums);
        dstglobal.SetGlobalBuffer(reinterpret_cast<__gm__ DstT *>(dstGm), dstNums);

        pipe.InitBuffer(inQueueX, 1, srcNums * sizeof(ScrT));
        pipe.InitBuffer(outQueue, 1, dstNums * sizeof(DstT));
        srcDataSize = srcNums;
        dstDataSize = dstNums;
        mask = vecMask;

    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }
    static constexpr Reg::CastTrait castTrait = {RegLayoutMd, SatMd, MzMd, RoundMd};
private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<ScrT> srcLocal = inQueueX.AllocTensor<ScrT>();
        DataCopy(srcLocal, srcGlobal, srcDataSize);
        inQueueX.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<DstT> dstLocal = outQueue.AllocTensor<DstT>();
        DstT zero = 0;
        Duplicate(dstLocal, zero, dstDataSize);
        LocalTensor<ScrT> srcLocal = inQueueX.DeQue<ScrT>();
        uint16_t maskBitSize = 256;
        uint16_t srcRepSize = maskBitSize/sizeof(ScrT);
        uint16_t dstRepSize = maskBitSize/sizeof(DstT);
        uint16_t rep = srcDataSize/srcRepSize;
        __ubuf__ DstT* dstPtr = (__ubuf__ DstT*)dstLocal.GetPhyAddr();
        __ubuf__ ScrT* srcPtr = (__ubuf__ ScrT*)srcLocal.GetPhyAddr();
        uint16_t align32byte = 32;
        __VEC_SCOPE__
        {
            RegTensor<ScrT> vSrcReg;
            RegTensor<DstT> vDstReg;
            vector_bool castReg;
            RegTensor<int64_t> tmpReg;
            RegTensor<hifloat8_t> hif8TmpReg;
            RegTensor<fp8_e4m3fn_t> f8e4m3TmpReg;
            RegTensor<fp8_e5m2_t> f8e5m2TmpReg;
            RegTensor<fp4x2_e2m1_t> f4e2m1TmpReg;
            RegTensor<fp4x2_e1m2_t> f4e1m2TmpReg;
            RegTensor<fp8_e8m0_t> f8e8m0TmpReg;
            uint32_t sreg = static_cast<uint32_t>(mask);
            MaskReg maskReg;
            maskReg = UpdateMask<ScrT>(sreg);
            uint32_t sreg2 = static_cast<uint32_t>(mask);
            MaskReg dstMaskReg = UpdateMask<DstT>(sreg2);
            uint32_t sreg3 = static_cast<uint32_t>(256);
            MaskReg f8Mask = UpdateMask<hifloat8_t>(sreg3);


            for (uint16_t i = 0; i < static_cast<uint16_t>(rep); i++) {
                Reg::DataCopy(vSrcReg, srcPtr + i * srcRepSize);

                if constexpr (mD == 4) {
                    if constexpr (std::is_same<ScrT, int32_t>::value && std::is_same<DstT, int32_t>::value) {
                        Cast<int64_t, ScrT, castTrait>(tmpReg, vSrcReg, maskReg);
                    } else {
                        Cast<DstT, ScrT, castTrait>(vDstReg, vSrcReg, maskReg);
                    }

                } else if constexpr (mD == 5) {
                    Cast<DstT, ScrT, castTrait>(vDstReg, vSrcReg, maskReg);
                } else if constexpr (mD == 6) {
                    Cast<DstT, ScrT, castTrait>(vDstReg, vSrcReg, maskReg);
                } else if constexpr (mD == 7) {
                    Cast<DstT, ScrT, castTrait>(vDstReg, vSrcReg, maskReg);
                } else if constexpr (mD == 8) {
                    Cast<DstT, ScrT, castTrait>(vDstReg, vSrcReg, maskReg);
                } else if constexpr (mD == 9) {
                    Cast<DstT, ScrT, castTrait>(vDstReg, vSrcReg, maskReg);
                } else if constexpr (mD == 10) {
                    Cast<DstT, ScrT, castTrait>(vDstReg, vSrcReg, maskReg);
                } else if constexpr (mD == 11) {
                    Cast<DstT, ScrT, castTrait>(vDstReg, vSrcReg, maskReg);
                } else if constexpr (mD == 12) {
                    Cast<hifloat8_t, ScrT, castTrait>(hif8TmpReg, vSrcReg, maskReg);
                    Cast<DstT, hifloat8_t, castTrait>(vDstReg, hif8TmpReg, f8Mask);
                } else if constexpr (mD == 13) {
                    Cast<hifloat8_t, ScrT, castTrait>(hif8TmpReg, vSrcReg, maskReg);
                    Cast<DstT, hifloat8_t, castTrait>(vDstReg, hif8TmpReg, f8Mask);
                } else if constexpr (mD == 14) {
                    Cast<fp8_e4m3fn_t, ScrT, castTrait>(f8e4m3TmpReg, vSrcReg, maskReg);
                    Cast<DstT, fp8_e4m3fn_t, castTrait>(vDstReg, f8e4m3TmpReg, f8Mask);
                } else if constexpr (mD == 15) {
                    Cast<fp8_e5m2_t, ScrT, castTrait>(f8e5m2TmpReg, vSrcReg, maskReg);
                    Cast<DstT, fp8_e5m2_t, castTrait>(vDstReg, f8e5m2TmpReg, f8Mask);
                } else if constexpr (mD == 16) {
                    Truncate<DstT, RoundMd, MzMd>(vDstReg, vSrcReg, maskReg);
                } else if constexpr (mD == 17) {
                    Cast<DstT, ScrT, castTrait>(vDstReg, vSrcReg, maskReg);
                } else if constexpr (mD == 18) {
                    Cast<fp4x2_e2m1_t, ScrT, castTrait>(f4e2m1TmpReg, vSrcReg, maskReg);
                    DataCopy((__ubuf__ fp4x2_e2m1_t*)dstPtr + i * dstRepSize, f4e2m1TmpReg, dstMaskReg);
                } else if constexpr (mD == 19) {
                    Cast<fp4x2_e1m2_t, ScrT, castTrait>(f4e1m2TmpReg, vSrcReg, maskReg);
                    DataCopy((__ubuf__ fp4x2_e1m2_t*)dstPtr + i * dstRepSize, f4e1m2TmpReg, dstMaskReg);
                } else if constexpr (mD == 20) {
                    DataCopy(f4e2m1TmpReg, (__ubuf__ fp4x2_e2m1_t*)srcPtr + i * srcRepSize);
                    Cast<DstT, fp4x2_e2m1_t, castTrait>(vDstReg, f4e2m1TmpReg, maskReg);
                    DataCopy(dstPtr + i * dstRepSize, vDstReg, dstMaskReg);
                } else if constexpr (mD == 21) {
                    DataCopy(f4e1m2TmpReg, (__ubuf__ fp4x2_e1m2_t*)srcPtr + i * srcRepSize);
                    Cast<DstT, fp4x2_e1m2_t, castTrait>(vDstReg, f4e1m2TmpReg, maskReg);
                    DataCopy(dstPtr + i * dstRepSize, vDstReg, dstMaskReg);
                } else if constexpr (mD == 22) {
                    if constexpr (std::is_same<ScrT, bfloat16_t>::value) {
                        Cast<DstT, ScrT, castTrait>(f8e8m0TmpReg, vSrcReg, maskReg);
                        DataCopy(dstPtr + i * dstRepSize, (RegTensor<fp8_e8m0_t> &)f8e8m0TmpReg, dstMaskReg);
                    } else {
                        Cast<DstT, ScrT, castTrait>(vDstReg, (RegTensor<fp8_e8m0_t> &)vSrcReg, maskReg);
                        DataCopy(dstPtr + i * dstRepSize, vDstReg, dstMaskReg);
                    }
                } else if constexpr (mD == 23) {
                    Cast<DstT, int4x2_t, castTrait>(vDstReg, (RegTensor<int4x2_t> &)vSrcReg, f8Mask);
                    DataCopy(dstPtr + i * dstRepSize, vDstReg, dstMaskReg);
                }
                if constexpr (mD < 18) {
                    DataCopy(dstPtr + i * dstRepSize, vDstReg, dstMaskReg);
                }
            }
        }
        outQueue.EnQue<DstT>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<DstT> dstLocal = outQueue.DeQue<DstT>();
        DataCopy(dstglobal, dstLocal, dstDataSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<ScrT> srcGlobal;
    GlobalTensor<DstT> dstglobal;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX;
    TQue<TPosition::VECOUT, 1> outQueue;
    uint32_t srcDataSize = 0;
    uint32_t dstDataSize = 0;
    uint32_t mask;

};

#define KERNEL_CAST_OPERATOR_TEST(srcType, dstType, m)                                               \
    extern "C" __global__ __aicore__ void kernel_micro_cast_kernel##_##srcType##_##dstType##_##m( \
        GM_ADDR dstGm, \
            GM_ADDR srcGm, uint64_t dstDataSize, uint64_t srcDataSize, \
            uint64_t MzMode, uint64_t partMd, uint64_t ppMd, \
            uint64_t satMd, uint64_t roundMd, \
            uint64_t vecMask, uint64_t mode)      \
    {                                                                                                            \
        kernel_cast_operator<srcType, dstType>(dstGm, srcGm, dstDataSize, srcDataSize,            \
        vecMask);        \
    }

struct MicroCastParams {
    void (*CallFunc)();
};

template<typename ScrT, typename DstT, int32_t mode, MaskMergeMode MzMd, RegLayout RegLayoutMd,
        SatMode SatMd, RoundMode RoundMd>
void MicroCastCase() {
    int srcByteSize = sizeof(ScrT);
    int dstByteSize = sizeof(DstT);
    int shapeSize = 1024;
    int mask = 256;
    int dataSize = 1024;
    uint8_t dstGm[shapeSize * dstByteSize] = {0};
    uint8_t srcGm[shapeSize * srcByteSize] = {0};

    KernelCast<ScrT, DstT, mode, MzMd, RegLayoutMd, SatMd, RoundMd> op;
    op.Init(dstGm, srcGm, dataSize, dataSize, mask);
    op.Process();
}

class MicroCastTestsuite : public testing::Test, public testing::WithParamInterface<MicroCastParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(MicroCastTestCase, MicroCastTestsuite,
    ::testing::Values(
                    MicroCastParams {
                        MicroCastCase<uint16_t, uint32_t, 4, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::UNKNOWN, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<half, float, 4, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::UNKNOWN, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<bfloat16_t, float, 4, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::UNKNOWN, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<uint8_t, uint16_t, 4, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::UNKNOWN, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<int8_t, int16_t, 4, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::UNKNOWN, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<int16_t, uint32_t, 4, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::UNKNOWN, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<int32_t, int32_t, 4, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::UNKNOWN, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<int16_t, int32_t, 4, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::UNKNOWN, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<uint8_t, half, 4, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::UNKNOWN, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<int8_t, half, 4, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::UNKNOWN, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<int16_t, float, 4, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::UNKNOWN, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<int32_t, int16_t, 5, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::NO_SAT, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<uint16_t, uint8_t, 5, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::NO_SAT, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<int16_t, uint8_t, 5, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::NO_SAT, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<uint32_t, uint16_t, 5, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::NO_SAT, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<uint32_t, int16_t, 5, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::NO_SAT, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<int32_t, uint16_t, 5, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::NO_SAT, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<int8_t, int32_t, 6, MaskMergeMode::ZEROING,
                        RegLayout::ONE, SatMode::UNKNOWN, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<int32_t, uint8_t, 7, MaskMergeMode::ZEROING,
                        RegLayout::ONE, SatMode::NO_SAT, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<float, int16_t, 8, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::NO_SAT, RoundMode::CAST_RINT>
                    },
                    MicroCastParams {
                        MicroCastCase<half, uint8_t, 8, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::NO_SAT, RoundMode::CAST_RINT>
                    },
                    MicroCastParams {
                        MicroCastCase<half, int8_t, 8, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::NO_SAT, RoundMode::CAST_RINT>
                    },
                    MicroCastParams {
                        MicroCastCase<bfloat16_t, int32_t, 8, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::NO_SAT, RoundMode::CAST_RINT>
                    },
                    MicroCastParams {
                        MicroCastCase<float, half, 8, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::NO_SAT, RoundMode::CAST_RINT>
                    },
                    MicroCastParams {
                        MicroCastCase<float, bfloat16_t, 8, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::NO_SAT, RoundMode::CAST_RINT>
                    },
                    MicroCastParams {
                        MicroCastCase<float, int32_t, 9, MaskMergeMode::ZEROING,
                        RegLayout::UNKNOWN, SatMode::NO_SAT, RoundMode::CAST_RINT>
                    },
                    MicroCastParams {
                        MicroCastCase<half, int32_t, 10, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::UNKNOWN, RoundMode::CAST_RINT>
                    },
                    MicroCastParams {
                        MicroCastCase<int16_t, half, 11, MaskMergeMode::ZEROING,
                        RegLayout::UNKNOWN, SatMode::UNKNOWN, RoundMode::CAST_RINT>
                    },
                    MicroCastParams {
                        MicroCastCase<half, bfloat16_t, 11,MaskMergeMode::ZEROING,
                        RegLayout::UNKNOWN, SatMode::UNKNOWN, RoundMode::CAST_RINT>
                    },
                    MicroCastParams {
                        MicroCastCase<float, float, 12, MaskMergeMode::ZEROING,
                        RegLayout::ONE, SatMode::NO_SAT, RoundMode::CAST_ROUND>
                    },
                    MicroCastParams {
                        MicroCastCase<half, half, 13, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::NO_SAT, RoundMode::CAST_ROUND>
                    },
                    MicroCastParams {
                        MicroCastCase<float, float, 14, MaskMergeMode::ZEROING,
                        RegLayout::ONE, SatMode::NO_SAT, RoundMode::CAST_RINT>
                    },
                    MicroCastParams {
                        MicroCastCase<float, float, 15, MaskMergeMode::ZEROING,
                        RegLayout::ONE, SatMode::NO_SAT, RoundMode::CAST_RINT>
                    },
                    MicroCastParams {
                        MicroCastCase<float, float, 16, MaskMergeMode::ZEROING,
                        RegLayout::UNKNOWN, SatMode::UNKNOWN, RoundMode::CAST_RINT>
                    },
                    MicroCastParams {
                        MicroCastCase<half, half, 16, MaskMergeMode::ZEROING,
                        RegLayout::UNKNOWN, SatMode::UNKNOWN, RoundMode::CAST_RINT>
                    },
                    MicroCastParams {
                        MicroCastCase<bfloat16_t, bfloat16_t, 16, MaskMergeMode::ZEROING,
                        RegLayout::UNKNOWN, SatMode::UNKNOWN, RoundMode::CAST_RINT>
                    },
                    MicroCastParams {
                        MicroCastCase<bfloat16_t, half, 17, MaskMergeMode::ZEROING,
                        RegLayout::UNKNOWN, SatMode::NO_SAT, RoundMode::CAST_RINT>
                    },
                    MicroCastParams {
                        MicroCastCase<bfloat16_t, uint8_t, 18, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::UNKNOWN, RoundMode::CAST_RINT>
                    },
                    MicroCastParams {
                        MicroCastCase<bfloat16_t, uint8_t, 19, MaskMergeMode::ZEROING,
                        RegLayout::ONE, SatMode::UNKNOWN, RoundMode::CAST_RINT>
                    },
                    MicroCastParams {
                        MicroCastCase<uint8_t, bfloat16_t, 20, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::UNKNOWN, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<uint8_t, bfloat16_t, 21, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::UNKNOWN, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<bfloat16_t, fp8_e8m0_t, 22, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::UNKNOWN, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<fp8_e8m0_t, bfloat16_t, 22, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::UNKNOWN, RoundMode::CAST_NONE>
                    },
                    MicroCastParams {
                        MicroCastCase<int4x2_t, bfloat16_t, 23, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::UNKNOWN, RoundMode::UNKNOWN>
                    },
                    MicroCastParams {
                        MicroCastCase<int4x2_t, int16_t, 23, MaskMergeMode::ZEROING,
                        RegLayout::ZERO, SatMode::UNKNOWN, RoundMode::UNKNOWN>
                    }
                     ));

TEST_P(MicroCastTestsuite, MicroCastTestCase)
{
    auto param = GetParam();
    param.CallFunc();
}
