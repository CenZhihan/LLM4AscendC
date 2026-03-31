/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/


/* !
 * \file nd2nz_utils.h
 * \brief
 */

#ifndef ND_2_NZ_UTILS_H
#define ND_2_NZ_UTILS_H

#include "kernel_operator.h"

namespace MatmulNzCustom {
constexpr uint32_t BLOCK_COUNT_MAX = 4095;
constexpr uint64_t SINGLE_COPY_SIZE = 256;
constexpr uint64_t BLOCK_SIZE_BYTE = 32;
constexpr uint64_t REPEAT_TIMES_MAX = 255;
constexpr uint64_t ALIGNED_H = 16;
constexpr uint32_t EVENT_ID_0 = 0;

/**
 * Ceiling division of a/b (uint64_t)
 * @param a Dividend
 * @param b Divisor (return a if b=0)
 * @return Ceiling result of a/b
 */
__aicore__ inline uint64_t MMDivCeil(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template <class T>
__aicore__ inline void Copy(const AscendC::LocalTensor<T>& dstLocal, const AscendC::LocalTensor<T>& srcLocal, uint32_t count)
{
    constexpr uint32_t copyLen = SINGLE_COPY_SIZE / sizeof(T);
    // vnchw parameters, which is used to decide stride, burstlength of Copy
    const AscendC::CopyRepeatParams para(AscendC::DEFAULT_BLK_STRIDE, AscendC::DEFAULT_BLK_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE);
    uint32_t repeatTimes = count / copyLen;
    uint32_t tail = count % copyLen;
    uint32_t offset = repeatTimes * copyLen;
    Copy(dstLocal, srcLocal, copyLen, repeatTimes, para);
    if (tail != 0) {
        Copy(dstLocal[offset], srcLocal[offset], tail, 1, para);
    }
}

template <class T>
__aicore__ inline void CopyPadGm2Ub(const AscendC::LocalTensor<T>& ubLocal1, const AscendC::GlobalTensor<T>& srcGlobal,
                                    uint32_t baseH, uint32_t baseW, uint32_t orgHeight, uint32_t orgWidth,
                                    uint8_t padH, uint8_t padW)
{
    uint64_t c0Size = AscendC::AscendCUtils::GetC0Count(sizeof(T));
    uint32_t width = baseW - padW;
    uint32_t height = baseH - padH;
    // Process gm->ub
    uint16_t blockLen = width * sizeof(T);
    uint32_t srcStride = (orgWidth - width) * sizeof(T);
    uint32_t numIter = height / BLOCK_COUNT_MAX;
    for (uint32_t i = 0; i < numIter; i++) {
        DataCopyPad(ubLocal1[BLOCK_COUNT_MAX * i * baseW],
                    srcGlobal[static_cast<uint64_t>(BLOCK_COUNT_MAX) * i * orgWidth],
                    {BLOCK_COUNT_MAX, blockLen, srcStride, 0, 0}, {false, 0, padW, 0});
    }
    uint16_t blockCountTail = height % BLOCK_COUNT_MAX;

    if (blockCountTail) {
        DataCopyPad(ubLocal1[BLOCK_COUNT_MAX * numIter * baseW],
                    srcGlobal[static_cast<uint64_t>(BLOCK_COUNT_MAX) * numIter * orgWidth],
                    {blockCountTail, blockLen, srcStride, 0, 0}, {false, 0, padW, 0});
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(EVENT_ID_0));
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(EVENT_ID_0));

    // padding
    if (padH) {
        AscendC::Duplicate(ubLocal1[height * baseW], (T)0, padH * baseW);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <class T>
__aicore__ inline void CopyPadNd2Nz(const AscendC::GlobalTensor<T>& dstGlobal, const AscendC::GlobalTensor<T>& srcGlobal,
                                    uint32_t baseH, uint32_t baseW, uint32_t orgHeight, uint32_t orgWidth,
                                    const AscendC::LocalTensor<T>& ubLocal1, const AscendC::LocalTensor<T>& ubLocal2,
                                    uint8_t padH, uint8_t padW)
{
    uint64_t c0Size = AscendC::AscendCUtils::GetC0Count(sizeof(T));

    // Process gm->ub
    CopyPadGm2Ub<T>(ubLocal1, srcGlobal, baseH, baseW, orgHeight, orgWidth,  padH, padW);

    // Process ub->ub
    uint32_t nRepeat = SINGLE_COPY_SIZE / sizeof(T);
    uint16_t nRowBlock = baseW / c0Size;
    uint32_t numIterI = baseH / REPEAT_TIMES_MAX;
    uint32_t heightTail = baseH % REPEAT_TIMES_MAX;
    uint32_t numIterJ = baseW / nRepeat;
    uint32_t widthTail = baseW % nRepeat;

    for (uint32_t i = 0; i < numIterI; i++) {
        for (uint32_t j = 0; j < numIterJ; j++) {
            Copy(ubLocal2[baseH * nRepeat * j + i * REPEAT_TIMES_MAX * c0Size],
                 ubLocal1[nRepeat * j + i * REPEAT_TIMES_MAX * baseW], nRepeat, REPEAT_TIMES_MAX,
                 {static_cast<uint16_t>(baseH), 1, 1, nRowBlock});
        }
        if (widthTail) {
            Copy(ubLocal2[baseH * nRepeat * numIterJ + i * REPEAT_TIMES_MAX * c0Size],
                 ubLocal1[nRepeat * numIterJ + i * REPEAT_TIMES_MAX * baseW], widthTail, REPEAT_TIMES_MAX,
                 {static_cast<uint16_t>(baseH), 1, 1, nRowBlock});
        }
    }
    for (uint32_t j = 0; j < numIterJ; j++) {
        Copy(ubLocal2[baseH * nRepeat * j + numIterI * REPEAT_TIMES_MAX * c0Size],
             ubLocal1[nRepeat * j + numIterI * REPEAT_TIMES_MAX * baseW], nRepeat, heightTail,
             {static_cast<uint16_t>(baseH), 1, 1, nRowBlock});
    }
    if (widthTail) {
        Copy(ubLocal2[baseH * nRepeat * numIterJ + numIterI * REPEAT_TIMES_MAX * c0Size],
             ubLocal1[nRepeat * numIterJ + numIterI * REPEAT_TIMES_MAX * baseW], widthTail, heightTail,
             {static_cast<uint16_t>(baseH), 1, 1, nRowBlock});
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(EVENT_ID_0));
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(EVENT_ID_0));
    uint64_t orgHeightRound = MMDivCeil(orgHeight, ALIGNED_H) * ALIGNED_H;
    // Process ub->gm
    if (orgHeightRound - baseH <= UINT16_MAX) {
        DataCopy(dstGlobal, ubLocal2, {nRowBlock, static_cast<uint16_t>(baseH), 0, uint16_t(orgHeightRound - baseH)});
    } else {
        for (uint16_t i = 0; i < nRowBlock; i++) {
            DataCopy(dstGlobal[orgHeightRound * c0Size * i], ubLocal2[baseH * c0Size * i],
                     {1, static_cast<uint16_t>(baseH), 0, 0});
        }
    }
}

template <>
__aicore__ inline void CopyPadNd2Nz<bfloat16_t>(const AscendC::GlobalTensor<bfloat16_t>& dstGlobal,
                                                const AscendC::GlobalTensor<bfloat16_t>& srcGlobal, uint32_t baseH,
                                                uint32_t baseW, uint32_t orgHeight, uint32_t orgWidth,
                                                const AscendC::LocalTensor<bfloat16_t>& ubLocal1,
                                                const AscendC::LocalTensor<bfloat16_t>& ubLocal2, uint8_t padH, uint8_t padW)
{
    AscendC::GlobalTensor<half> dstGlobalTrans;
    AscendC::GlobalTensor<half> srcGlobalTrans;
    dstGlobalTrans.SetGlobalBuffer((__gm__ half*)dstGlobal.GetPhyAddr(0));
    srcGlobalTrans.SetGlobalBuffer((__gm__ half*)srcGlobal.GetPhyAddr(0));
    CopyPadNd2Nz<half>(dstGlobalTrans, srcGlobalTrans, baseH, baseW, orgHeight, orgWidth,
                       ubLocal1.ReinterpretCast<half>(), ubLocal2.ReinterpretCast<half>(), padH, padW);
}
/**
 * @brief  Convert Matrix ND into NZ format.
 * @param  oriN: Original N value of ND matrix.
 * @param  oriD: Original D value of ND matrix.
 * @param  nValue: Alignment n value of ND matrix.
 * @param  dValue: Alignment d value of ND matrix.
 * @param  baseN: The base nValue size of ND matrix.
 * @param  baseD: The base dValue size of ND matrix.
 * @param  usedCoreNum: The total use core number.
 * @param  srcGlobal: Source global tensor of ND matrix.
 * @param  dstGlobal: Dst global tensor of NZ matrix.
 * @param  calcBuf: Use UB as the format conversion buffer.
 * @retval None
 */
template <class T>
__aicore__ inline void MatrixtoNZ(uint64_t oriN, uint64_t oriD, uint64_t nValue, uint64_t dValue, uint32_t baseN,
                                  uint32_t baseD, uint32_t usedCoreNum, const AscendC::GlobalTensor<T>& srcGlobal,
                                  const AscendC::GlobalTensor<T>& dstGlobal, AscendC::TBuf<AscendC::TPosition::VECCALC>& calcBuf)
{
    uint32_t vBlockIndex = AscendC::GetBlockIdx();
    AscendC::LocalTensor<T> tempUb = calcBuf.Get<T>();
    AscendC::LocalTensor<T> transBuf = tempUb[(AscendC::TOTAL_UB_SIZE / 2) / sizeof(T)];
    uint64_t nCnt = MMDivCeil(oriN, baseN);            // Total number of base blocks in nValue
    uint64_t dCnt = MMDivCeil(dValue, baseD);          // Total number of base blocks in dValue
    uint64_t tailN = nValue - (nCnt - 1) * baseN;      // Tail base blocks in nValue
    uint64_t tailD = dValue - (dCnt - 1) * baseD;      // Tail base blocks in dValue
    uint64_t totalCnt = nCnt * dCnt;                   // Total number of base blocks
    uint32_t round = MMDivCeil(totalCnt, usedCoreNum); // each core total loops
    uint32_t preCoreNum = (totalCnt % usedCoreNum) == 0 ? usedCoreNum : totalCnt % usedCoreNum;
    uint32_t index = vBlockIndex < preCoreNum ? vBlockIndex * round : vBlockIndex * (round - 1) + preCoreNum;
    uint32_t realRound = vBlockIndex < preCoreNum ? round : round - 1; // each core total loops
    uint32_t nCalcLen = 0; // The real value of nValue, may use baseN or tailN
    uint32_t dCalcLen = 0; // The real value of dValue, may use baseD or tailD
    for (uint32_t j = 0; j < realRound; ++j) {
        if (index >= totalCnt) {
            continue;
        }
        if ((index + 1) % totalCnt == 0) {
            nCalcLen = tailN;
            dCalcLen = tailD;
        } else if ((index + 1) % totalCnt > (totalCnt - dCnt)) {
            nCalcLen = tailN;
            dCalcLen = baseD;
        } else if ((index + 1) % dCnt == 0) {
            nCalcLen = baseN;
            dCalcLen = tailD;
        } else {
            nCalcLen = baseN;
            dCalcLen = baseD;
        }
        // calc pad_value
        const auto& nIndx = index / dCnt;
        const auto& dIndx = index % dCnt;
        const uint8_t padN = (nIndx == nCnt - 1) ? nValue - oriN : 0; // The pad value of n Dims
        const uint8_t padD = (dIndx == dCnt - 1) ? dValue - oriD : 0; // The pad value of d Dims
        auto srcGmIdx = (dIndx * baseD + nIndx * baseN * oriD);
        auto dstGmIdx = (dIndx * nValue * baseD + nIndx * baseN * AscendC::AscendCUtils::GetC0Count(sizeof(T)));
        CopyPadNd2Nz(dstGlobal[dstGmIdx], srcGlobal[srcGmIdx], nCalcLen, dCalcLen, oriN, oriD, tempUb, transBuf,
                     padN, padD);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(EVENT_ID_0));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(EVENT_ID_0));
        index += 1;
    }
}

/**
 * @brief  Convert Matrix B ND into NZ format.
 * @param  dst: Gm dst addr of B matrix.
 * @param  src: Gm source addr of B matrix.
 * @param  tiling: TCubeTiling.
 * @param  isTransposeB: transpose of B matrix.
 * @param  calcBuf: Use UB as the format conversion buffer.
 * @param  baseBN: The base nValue size of ND matrix.
 * @param  baseBD: The base dValue size of ND matrix.
 * @retval None
 */
template <class T>
__aicore__ inline void MatrixBtoNZ(GM_ADDR dst, GM_ADDR src, const TCubeTiling& tiling, bool isTransposeB,
                                   AscendC::TBuf<AscendC::TPosition::VECCALC>& calcBuf, uint32_t baseBN, uint32_t baseBD)
{
    uint64_t c0Size = AscendC::AscendCUtils::GetC0Count(sizeof(T));
    uint32_t usedCoreNum = tiling.usedCoreNum * AscendC::GetTaskRation(); // use max core nums
    uint64_t alignedNSize = 0;
    uint64_t alignedKSize = 0;
    alignedNSize = isTransposeB ? MMDivCeil(tiling.N, ALIGNED_H) * ALIGNED_H : MMDivCeil(tiling.N, c0Size) * c0Size;
    alignedKSize = isTransposeB ? MMDivCeil(tiling.Kb, c0Size) * c0Size : MMDivCeil(tiling.Kb, ALIGNED_H) * ALIGNED_H;
    uint64_t oriN = isTransposeB ? tiling.N : tiling.Kb;
    uint64_t oriD = isTransposeB ? tiling.Kb : tiling.N;
    uint64_t nValue = isTransposeB ? alignedNSize : alignedKSize;
    uint64_t dValue = isTransposeB ? alignedKSize : alignedNSize;
    AscendC::GlobalTensor<T> srcGlobal;
    AscendC::GlobalTensor<T> dstGlobal;
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst), alignedNSize * alignedKSize);
    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src), oriD * oriN);
    MatrixtoNZ(oriN, oriD, nValue, dValue, baseBN, baseBD, usedCoreNum, srcGlobal, dstGlobal, calcBuf);
}
} // namespace MatmulNzCustom

#endif // ND_2_NZ_UTILS_H