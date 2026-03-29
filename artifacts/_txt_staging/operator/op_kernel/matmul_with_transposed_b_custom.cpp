
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

extern "C" __global__ __aicore__ void matmul_with_transposed_b_custom(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    // Copy TCubeTiling from GM
    TCubeTiling tiling;
    auto tempTilingGM = (__gm__ uint32_t*)tilingGM;
    auto tempTiling = (uint32_t*)&tiling;
    for (int i = 0; i < (int)(sizeof(TCubeTiling) / sizeof(uint32_t)); ++i, ++tempTilingGM, ++tempTiling) {
        *tempTiling = *tempTilingGM;
    }

    // Global tensors
    AscendC::GlobalTensor<half> aGlobal;
    AscendC::GlobalTensor<half> bGlobal;
    AscendC::GlobalTensor<float> cGlobal;

    int32_t sizeA = tiling.ALayoutInfoB * tiling.singleCoreM * tiling.singleCoreK * sizeof(half);
    int32_t sizeB = tiling.BLayoutInfoB * tiling.singleCoreK * tiling.singleCoreN * sizeof(half);
    int32_t sizeC = tiling.CLayoutInfoB * tiling.singleCoreM * tiling.singleCoreN * sizeof(float);

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(aGM), sizeA);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(bGM), sizeB);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGM), sizeC);

    // Shared sizes (per docs)
    tiling.shareMode = 0;
    tiling.shareL1Size = 512 * 1024;
    tiling.shareL0CSize = 128 * 1024;
    tiling.shareUbSize = 0;

    // Slice a single core chunk
    int offset_a = 0, offset_b = 0, offset_c = 0;
    AscendC::GlobalTensor<half> gm_a;
    gm_a.SetGlobalBuffer(const_cast<__gm__ half*>(aGlobal[offset_a].GetPhyAddr()), tiling.singleCoreM * tiling.singleCoreK);
    AscendC::GlobalTensor<half> gm_b;
    gm_b.SetGlobalBuffer(const_cast<__gm__ half*>(bGlobal[offset_b].GetPhyAddr()), tiling.singleCoreK * tiling.singleCoreN);
    AscendC::GlobalTensor<float> gm_c;
    gm_c.SetGlobalBuffer(const_cast<__gm__ float*>(cGlobal[offset_c].GetPhyAddr()), tiling.singleCoreM * tiling.singleCoreN);

    // Define Matmul types: A(M,K) half, B(N,K) half with transpose enabled, C(M,N) float
    typedef matmul::MatmulType <AscendC::TPosition::GM, CubeFormat::ND, half, false, LayoutMode::NORMAL> aType;
    typedef matmul::MatmulType <AscendC::TPosition::GM, CubeFormat::ND, half, true,  LayoutMode::NORMAL> bType;
    typedef matmul::MatmulType <AscendC::TPosition::GM, CubeFormat::ND, float, false, LayoutMode::NORMAL> cType;
    typedef matmul::MatmulType <AscendC::TPosition::GM, CubeFormat::ND, float> biasType;

    constexpr static MatmulConfig MM_CFG = GetNormalConfig(false, false, false, BatchMode::BATCH_LESS_THAN_L1);
    matmul::Matmul<aType, bType, cType, biasType, MM_CFG> mm;

    AscendC::TPipe pipe;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm);

    mm.Init(&tiling);
    // A is not transposed; B is provided as [N, K], enable transpose to compute A(M,K) x B^T(K,N)
    mm.SetTensorA(gm_a, false);
    mm.SetTensorB(gm_b, true);
    mm.SetWorkspace(workspaceGM, 0);

    // Single-batch GEMM
    mm.IterateBatch(gm_c, 1, 1, false);
}
