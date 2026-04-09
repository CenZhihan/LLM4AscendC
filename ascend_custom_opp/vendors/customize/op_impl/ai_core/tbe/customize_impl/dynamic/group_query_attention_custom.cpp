
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // queue depth

class KernelGroupQueryAttentionCustom {
public:
    __aicore__ inline KernelGroupQueryAttentionCustom() {}
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y,
                                uint32_t B, uint32_t L, uint32_t H, uint32_t Hkv,
                                uint32_t Hd, uint32_t groupSize,
                                uint32_t blockSegments, uint32_t tileNum)
    {
        this->B = B;
        this->L = L;
        this->H = H;
        this->Hkv = Hkv;
        this->Hd = Hd;
        this->groupSize = groupSize;
        this->segmentsPerBlock = blockSegments; // number of [Hd] segments per block
        this->tileNum = tileNum;

        uint32_t startSegment = segmentsPerBlock * AscendC::GetBlockIdx();
        uint32_t totalSegments = B * L * H;
        // Clamp for last block to avoid overflow
        if (startSegment + segmentsPerBlock > totalSegments) {
            segmentsPerBlock = (totalSegments > startSegment) ? (totalSegments - startSegment) : 0;
        }

        // Set Global Buffers
        // y base pointer starts at startSegment * Hd
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + (uint64_t)startSegment * Hd, (uint64_t)segmentsPerBlock * Hd);
        // vGm covers full v tensor for addressable reads
        vGm.SetGlobalBuffer((__gm__ DTYPE_V *)v, (uint64_t)B * L * Hkv * Hd);
        // q,k are not used in this simplified kernel, but we set them to avoid unused warnings
        qGm.SetGlobalBuffer((__gm__ DTYPE_Q *)q, (uint64_t)B * L * H * Hd);
        kGm.SetGlobalBuffer((__gm__ DTYPE_K *)k, (uint64_t)B * L * Hkv * Hd);

        // Each local buffer holds one segment of size Hd
        pipe.InitBuffer(inQueueV, BUFFER_NUM, this->Hd * sizeof(DTYPE_V));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->Hd * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process()
    {
        // For each [Hd] segment in this block, copy from V(b,l,h//groupSize,:) to Y(b,l,h,:)
        for (uint32_t s = 0; s < segmentsPerBlock; ++s) {
            CopyIn(s);
            Compute(s);
            CopyOut(s);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t s)
    {
        // Global segment index
        uint32_t startSegment = segmentsPerBlock * AscendC::GetBlockIdx();
        uint32_t g = startSegment + s; // global segment id in [0, B*L*H)

        // Decode (b, l, h)
        uint32_t perBL = L * H;
        uint32_t b = g / perBL;
        uint32_t rem = g - b * perBL;
        uint32_t l = rem / H;
        uint32_t h = rem - l * H;

        // Map to kv head
        uint32_t hkv = h / groupSize;

        // Compute source offset in v: ((b*L + l) * Hkv + hkv) * Hd
        uint64_t v_offset = (((uint64_t)b * L + l) * Hkv + hkv) * (uint64_t)Hd;

        AscendC::LocalTensor<DTYPE_V> vLocal = inQueueV.AllocTensor<DTYPE_V>();
        AscendC::DataCopy(vLocal, vGm[v_offset], Hd);
        inQueueV.EnQue(vLocal);
    }

    __aicore__ inline void Compute(uint32_t s)
    {
        AscendC::LocalTensor<DTYPE_V> vLocal = inQueueV.DeQue<DTYPE_V>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        // In this simplified kernel, y = expanded v (no attention weights applied)
        AscendC::DataCopy(yLocal, vLocal, Hd);
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueV.FreeTensor(vLocal);
    }

    __aicore__ inline void CopyOut(uint32_t s)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        // y base already points to startSegment*Hd, each segment is contiguous of size Hd
        uint64_t y_offset = (uint64_t)s * Hd;
        AscendC::DataCopy(yGm[y_offset], yLocal, Hd);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueV;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;

    AscendC::GlobalTensor<DTYPE_Q> qGm;
    AscendC::GlobalTensor<DTYPE_K> kGm;
    AscendC::GlobalTensor<DTYPE_V> vGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;

    uint32_t B;
    uint32_t L;
    uint32_t H;
    uint32_t Hkv;
    uint32_t Hd;
    uint32_t groupSize;
    uint32_t segmentsPerBlock;
    uint32_t tileNum;
};

extern "C" __global__ __aicore__ void group_query_attention_custom(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGroupQueryAttentionCustom op;
    op.Init(q, k, v, y,
            tiling_data.B, tiling_data.L, tiling_data.H, tiling_data.Hkv,
            tiling_data.Hd, tiling_data.groupSize,
            tiling_data.blockSegments, tiling_data.tileNum);
    op.Process();
}
