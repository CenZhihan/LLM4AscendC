/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef VERIFY_SINGLE_HEADER
#include "kernel_operator.h"
#else
#include "kernel_operator_proposal_intf.h"
#endif

// __aicore__ inline void MrgSort4(const LocalTensor<T>& dst, const MrgSortSrcList<T>& src, const MrgSort4Info& params);
extern "C" __global__ __aicore__ void KernelTestMrgSort41() {
    AscendC::LocalTensor<float> dst;
    AscendC::MrgSortSrcList<float> src;
    AscendC::MrgSort4Info params;
    AscendC::MrgSort4(dst, src, params);
}

// __aicore__ inline void RpSort16(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime);
extern "C" __global__ __aicore__ void KernelTestRpSort161() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    int32_t repeatTime = 0;
    AscendC::RpSort16(dst, src, repeatTime);
}

// __aicore__ inline void MrgSort(const LocalTensor<T>& dst, const MrgSortSrcList<T>& src, const MrgSort4Info& params);
extern "C" __global__ __aicore__ void KernelTestMrgSort1() {
    AscendC::LocalTensor<float> dst;
    AscendC::MrgSortSrcList<float> src;
    AscendC::MrgSort4Info params;
    AscendC::MrgSort(dst, src, params);
}

// __aicore__ inline void Sort32(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<uint32_t>& src1, const int32_t repeatTime);
extern "C" __global__ __aicore__ void KernelTestSort321() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src0;
    AscendC::LocalTensor<uint32_t> src1;
    int32_t repeatTime = 0;
    AscendC::Sort32(dst, src0, src1, repeatTime);
}

// __aicore__ inline void ProposalConcat(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t modeNumber);
extern "C" __global__ __aicore__ void KernelTestProposalConcat1() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    int32_t repeatTime = 0;
    int32_t modeNumber = 0;
    AscendC::ProposalConcat(dst, src, repeatTime, modeNumber);
}

// __aicore__ inline void ProposalExtract(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t modeNumber);
extern "C" __global__ __aicore__ void KernelTestProposalExtract1() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> src;
    int32_t repeatTime = 0;
    int32_t modeNumber = 0;
    AscendC::ProposalExtract(dst, src, repeatTime, modeNumber);
}

// __aicore__ inline void Concat(LocalTensor<T> &concat, const LocalTensor<T> &src, const LocalTensor<T> &tmp, const int32_t repeatTime);
extern "C" __global__ __aicore__ void KernelTestConcat1() {
    AscendC::LocalTensor<float> concat;
    AscendC::LocalTensor<float> src;
    AscendC::LocalTensor<float> tmp;
    int32_t repeatTime = 0;
    AscendC::Concat(concat, src, tmp, repeatTime);
}

// __aicore__ inline void Extract(const LocalTensor<T> &dstValue, const LocalTensor<uint32_t> &dstIndex, const LocalTensor<T> &sorted, const int32_t repeatTime);
extern "C" __global__ __aicore__ void KernelTestExtract1() {
    AscendC::LocalTensor<float> dstValue;
    AscendC::LocalTensor<uint32_t> dstIndex;
    AscendC::LocalTensor<float> sorted;
    int32_t repeatTime = 0;
    AscendC::Extract(dstValue, dstIndex, sorted, repeatTime);
}

// __aicore__ inline void MrgSort(const LocalTensor<T> &dst, const MrgSortSrcList<T> &sortList, const uint16_t elementCountList[4], uint32_t sortedNum[4], uint16_t validBit, const int32_t repeatTime);
extern "C" __global__ __aicore__ void KernelTestMrgSort2() {
    AscendC::LocalTensor<float> dst;
    AscendC::MrgSortSrcList<float> sortList;
    uint16_t elementCountList[4];
    uint32_t sortedNum[4];
    uint16_t validBit = 1;
    int32_t repeatTime = 0;
    AscendC::MrgSort(dst, sortList, elementCountList, sortedNum, validBit, repeatTime);
}

// template <typename T, bool isFullSort>
// __aicore__ inline void Sort(const LocalTensor<T> &dst, const LocalTensor<T> &concat, const LocalTensor<uint32_t> &index, LocalTensor<T> &tmp, const int32_t repeatTime);
extern "C" __global__ __aicore__ void KernelTestSort1() {
    AscendC::LocalTensor<float> dst;
    AscendC::LocalTensor<float> concat;
    AscendC::LocalTensor<uint32_t> index;
    AscendC::LocalTensor<float> tmp;
    int32_t repeatTime = 0;
    AscendC::Sort<float, false>(dst, concat, index, tmp, repeatTime);
}

// __aicore__ inline __inout_pipe__(S) void GetMrgSortResult( uint16_t &mrgSortList1, uint16_t &mrgSortList2, uint16_t &mrgSortList3, uint16_t &mrgSortList4);
extern "C" __global__ __aicore__ void KernelTestGetMrgSortResult1() {
    uint16_t mrgSortList1 = 1;
    uint16_t mrgSortList2 = 1;
    uint16_t mrgSortList3 = 1;
    uint16_t mrgSortList4 = 1;
    AscendC::GetMrgSortResult(mrgSortList1, mrgSortList2, mrgSortList3, mrgSortList4);
}
