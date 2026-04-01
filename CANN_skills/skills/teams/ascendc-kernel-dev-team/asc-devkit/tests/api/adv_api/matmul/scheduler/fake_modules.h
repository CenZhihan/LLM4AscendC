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
using namespace std;
using namespace AscendC;

template <typename T>
const LocalTensor<T> EMPTY_TENSOR;

template <typename IMPL, typename TYPE, const auto& MM_CFG, typename = void>
class CustomLoadToL0
{
    using A_T = typename TYPE::T;
public:

    __aicore__ inline void Prepare(bool isTranspose, uint16_t kL1, uint16_t mnL1Len = 0) {}

    __aicore__ inline void Load(const LocalTensor<A_T> &l0A, const LocalTensor<A_T> &l1A,
    uint16_t aL1M, uint16_t aL1K, uint16_t madM, uint16_t madK, uint16_t aL1MOffset, uint16_t aL1KOffset,
    bool isATranspose) {}
};

template <typename IMPL, typename C_T, typename A_TYPE, typename B_T, const auto& MM_CFG>
class CustomMmadCompute {
using A_T = typename A_TYPE::T;
public:
    inline __aicore__ void Compute(const LocalTensor<C_T>& cMatrix, const LocalTensor<A_T>& l0A,
        const LocalTensor<B_T>& l0B, uint16_t mmadM, uint16_t mmadK, uint16_t mmadN, bool isATrans, bool isBTrans,
        uint8_t unitFlag = 0, bool cmatrixSource = false, bool cmatrixInitVal = true, bool isBias = false) {}
};

template <typename IMPL, class A_TYPE, class B_TYPE, const auto& MM_CFG, typename = void>
class CustomTBufPoolL0 {
public:
    __aicore__ inline void Init(uint16_t = 0)
    {
        GetTPipePtr()->InitBuffer(l0aBuf_, 10240);
    }
    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline CustomTBufPoolL0& Allocate() {
        return *this;
    }

    template <TPosition Pos, typename T, bool IS_INTRA_BLOCK = false>
    __aicore__ inline LocalTensor<T> GetBuffer(uint8_t = 0) {
        LocalTensor<T> tempTensor = l0aBuf_.Get<T>();
        return tempTensor;
    }

    template <TPosition Pos>
    __aicore__ inline bool Hit(uint32_t pos = 0) {
        return false;
    }

    __aicore__ inline void ResetCache() {}

    __aicore__ inline void EnQue() {}
    
    __aicore__ inline void DeQue() {}

    __aicore__ inline void Free() {

    }
    TBuf<TPosition::A2> l0aBuf_;
};


template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG, typename = void>
class CustomCopyCubeIn
{
    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename INPUT_TYPE::T;
public:
    template <typename ScheduleContext = int>
    __aicore__ inline LocalTensor<TransT> LoadData(int curRow, int curCol, int tileHeight, int tileWidth,
                                                   const ScheduleContext& context = 0)
    {
        return EMPTY_TENSOR<TransT>;
    }

    __aicore__ inline void ClearLoadData(const LocalTensor<TransT>& tensor = EMPTY_TENSOR<TransT>,
        int32_t curRow = 0, int32_t curCol = 0) {}

    template <typename ScheduleContext = int>
    __aicore__ inline LocalTensor<TransT> AsyncLoadData(
        int32_t curRow, int32_t curCol, int32_t tileHeight, int32_t tileWidth, const ScheduleContext& context = 0)
    {
        return EMPTY_TENSOR<TransT>;
    }

    __aicore__ inline void AwaitLoadData() {}

    __aicore__ inline void Reset() {}

    template <typename T>
    __aicore__ inline void SetInput(const T& dst, bool isTranspose) {}

    // batch
    __aicore__ inline LocalTensor<TransT> AllocTensor(int32_t iterIndex = 0) {
        TPipe pipe;
        TQue<TPosition::VECIN, 2> leftMatrix;
        pipe.InitBuffer(leftMatrix, 1, 1024);
        LocalTensor<TransT> fakeTensor;
        fakeTensor = leftMatrix.AllocTensor<SrcT>();
        return fakeTensor;
    }

    __aicore__ inline void BatchLoad(LocalTensor<TransT>& dstTensor, const uint32_t matrixStride,
        const int32_t outerIdx, const int32_t splitIdx, const int32_t splitSize) {}

    __aicore__ inline void BatchDestroy(const LocalTensor<TransT>& tensor = LocalTensor<TransT>{}) {}

    __aicore__ inline void Destroy() {}
};

template <typename IMPL, typename L0cT, const auto& MM_CFG, typename = void>
class CustomCubeOutBuffer
{
public:
    __aicore__ inline LocalTensor<L0cT> AllocTensor() {
        return EMPTY_TENSOR<L0cT>;
    }

    __aicore__ inline LocalTensor<L0cT> GetTensor() {
        LocalTensor<L0cT> out;
        TBuffAddr addr;
        addr.logicPos = 2;
        addr.absAddr = 0;
        out.SetAddr(addr);
        out.SetSize(1);
        return out;
    }

    __aicore__ inline void EnQue(LocalTensor<L0cT>& tensor) {}

    __aicore__ inline LocalTensor<L0cT> DeQue() {
        return EMPTY_TENSOR<L0cT>;
    }

    __aicore__ inline void FreeTensor(LocalTensor<L0cT>& co1Local) {}

    __aicore__ inline void Destroy() {}
};

template <typename IMPL, typename L0cT, const auto& MM_CFG, typename = void>
class CustomCubeOutBufferForMNDB
{
public:
    __aicore__ inline void Init(uint32_t len) {
        constexpr int32_t DB_NUM = 2;
        GetTPipePtr()->InitBuffer(CO1_, DB_NUM * len * sizeof(L0cT));
    }

    __aicore__ inline LocalTensor<L0cT> AllocTensor() {
        cMatrix_ = CO1_.template Get<L0cT>();
        return cMatrix_;
    }

    __aicore__ inline LocalTensor<L0cT> GetTensor() {
        return cMatrix_;
    }

    __aicore__ inline void EnQue(LocalTensor<L0cT>& tensor) {}

    __aicore__ inline LocalTensor<L0cT> DeQue() {
        return cMatrix_;
    }

    __aicore__ inline void FreeTensor(LocalTensor<L0cT>& co1Local) {}

    __aicore__ inline void Destroy() {}
private:
    TBuf<TPosition::CO1> CO1_;
    LocalTensor<L0cT> cMatrix_;
};


template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, const auto& MM_CFG, typename = void>
class CustomCopyCubeOut
{
    using DstT = typename C_TYPE::T;
    using SrcT = typename GetMmDstType<typename A_TYPE::T>::Type;
public:
    template <bool enSequentialWrite = false, typename ScheduleContext = int>
    __aicore__ inline void Copy(const GlobalTensor<DstT>& gm, const LocalTensor<SrcT>& co1Local, int curRow,
                                   int curCol, int32_t baseHeight, int32_t baseWidth, int32_t baseBlockHeight,
                                   int32_t baseBlockWidth, const ScheduleContext& context = 0)
    {}
};

template <typename IMPL, class A_TYPE, class BIAS_TYPE, const auto& MM_CFG, typename = void>
class CustomCopyBiasIn {
    using BiasT = typename BIAS_TYPE::T;

public:
    __aicore__ inline CustomCopyBiasIn() = default;
    __aicore__ inline ~CustomCopyBiasIn() = default;
    __aicore__ inline void Init(int32_t bufferLen) {}
    __aicore__ inline void SetBias(bool enableBias = false) {}
    __aicore__ inline bool IsBias() const {
        return true;
    }
    __aicore__ inline void SetInput(const LocalTensor<BiasT>& biasLocal) {}
    __aicore__ inline void SetInput(const GlobalTensor<BiasT>& biasGlobal) {}
    __aicore__ inline LocalTensor<BiasT> LoadData(int32_t dataLen, int32_t dataNum = 1, int32_t srcOffset = 0)
    {
        return EMPTY_TENSOR<BiasT>;
    }
    __aicore__ inline LocalTensor<BiasT> AsyncLoadData(int32_t dataLen, int32_t dataNum = 1, int32_t srcOffset = 0)
    {
        return EMPTY_TENSOR<BiasT>;
    }
    __aicore__ inline void AwaitLoadData() {}
    __aicore__ inline void ClearLoadData(const LocalTensor<BiasT>& bias = EMPTY_TENSOR<BiasT>) {}
    __aicore__ inline void Destroy() {}
};

template <typename IMPL, class A_TYPE, class BIAS_TYPE, const auto &MM_CFG, typename = void>
class CustomLoadBias2C2 {
    using BiasT = typename BIAS_TYPE::T;
public:
    __aicore__ inline CustomLoadBias2C2() = default;
    __aicore__ inline ~CustomLoadBias2C2() = default;
    __aicore__ inline void Init() {}
    __aicore__ inline void LoadData(LocalTensor<BiasT> &bias, int32_t sMadN = 0, int32_t srcOffset = 0) {}
    __aicore__ inline void ClearLoadData() {}
};

template <typename IMPL, class A_TYPE, class B_TYPE, class BIAS_TYPE, const auto &MM_CFG, typename = void>
class CustomBiasScheduler {
    using BiasT = typename BIAS_TYPE::T;
    using TensorT = typename Conditional<(PhyPosIsGM(BIAS_TYPE::pos) || !Impl::Detail::MatmulFeatureTrait<MM_CFG>::IsSupportUBToL1()),
                                         GlobalTensor<BiasT>, LocalTensor<BiasT>>::type;

public:
    __aicore__ inline CustomBiasScheduler() = default;
    __aicore__ inline ~CustomBiasScheduler() = default;
    __aicore__ inline void SetBias(bool enableBias = false) {}
    __aicore__ inline bool IsBias() const
    {
        return false;
    }
    __aicore__ inline void SetInput(const TensorT& srcTensor) {}
    __aicore__ inline void Init(int32_t batchNum = 0) {}
    __aicore__ inline void End() {}
    __aicore__ inline LocalTensor<BiasT> CopyIn(int32_t dataLen, int32_t dataNum = 1, int32_t srcOffset = 0)
    {
        return EMPTY_TENSOR<BiasT>;
    }
    __aicore__ inline void Free(LocalTensor<BiasT> &biasC1) {}
    __aicore__ inline void SplitLoad(LocalTensor<BiasT> &biasC1, int32_t dataLen = 0, int32_t srcOffset = 0) {}
    __aicore__ inline void Free() {}
    __aicore__ inline void SetSingleOffset(int32_t offset = 0) {}
    __aicore__ inline void Destroy(LocalTensor<BiasT>& bias = LocalTensor<BiasT>{}) {}
    __aicore__ inline void StopBias(LocalTensor<BiasT> &bias) {}
};

template <typename IMPL, const auto &MM_CFG, class INPUT_TYPE, typename = void>
class CustomMatmulTensorInfo {
    using SrcT = typename INPUT_TYPE::T;
public:
    __aicore__ inline GlobalTensor<SrcT> GetGlobalTensor() const
    {
        GlobalTensor<SrcT> globalMatrix;
        uint8_t fakeData[1024] = {0};
        globalMatrix.SetGlobalBuffer(reinterpret_cast<__gm__ SrcT*>(fakeData), 1024);
        return globalMatrix;
    }

};
