# 尾核&尾块<a name="ZH-CN_TOPIC_0000002500468234"></a>

对于不同shape的输入进行数据切分时，可能会发生数据无法平均分配到多个核、同时每个核内的数据无法均分的情况。参考[核间均分场景下的尾块处理](尾块Tiling.md)与[核间不均分场景下的尾核处理](尾核Tiling.md)的处理方式，将两者结合起来考虑整核的尾块、尾核的尾块的处理方式。

## Tiling实现<a name="zh-cn_topic_0000002236277533_section132981926144714"></a>

由于本场景中核间、核内的数据均无法均分，在[核间不均分场景下的尾核处理](尾核Tiling.md)定义的Tiling结构体的基础上增加两个成员变量：

-   formerLastTileLength：数据量多的核最后一个分块大小，即整核的尾块大小。

    计算时，先按[尾核Tiling](尾核Tiling.md)中提到的分核策略，切分数据量多的核。

    ```
    // shape需要对齐到的datablock
    uint32_t totalLengthAligned = (totalLength % alignNum == 0U) ?
                          static_cast<uint32_t>(totalLength) :
                          ((static_cast<uint32_t>(totalLength) + alignNum - 1) / alignNum) * alignNum;
    // 计算整核数量
    uint32_t formerNum = (totalLengthAligned / alignNum) % numBlocks;
    // 计算整核的数据量
    uint32_t formerLength = static_cast<uint32_t>(((totalLengthAligned + numBlocks - 1) / numBlocks + alignNum - 1) / alignNum) * alignNum;
    ```

    再按[尾块Tiling](尾块Tiling.md)中的切分策略，计算尾块长度。

    ```
    TilingParamsCalc(formerLength, alignNum, formerTileNum, formerTileLength, formerLastTileLength);
    
    void TilingParamsCalc(uint32_t length, uint32_t alignNum, uint32_t& tileNum, uint32_t& tileLength,
                          uint32_t& lastTileLength)
    {
        tileNum = length / (alignNum * UB_BLOCK_NUM);
        if (tileNum == 0U) {
            tileLength = 0U;
            lastTileLength = static_cast<uint32_t>(((length + alignNum - 1) / alignNum) * alignNum);
        } else if (static_cast<uint32_t>(length / alignNum) % UB_BLOCK_NUM == 0U) {
            tileLength = UB_BLOCK_NUM * alignNum;
            lastTileLength = 0U;
        } else {
            tileLength = UB_BLOCK_NUM * alignNum;
            lastTileLength = static_cast<uint32_t>(length - tileNum * tileLength);
        }
    }
    ```

-   tailLastTileLength：数据量少的核最后一个分块大小，即尾核的尾块大小。

    计算时，先按[尾核Tiling](尾核Tiling.md)中提到的分核策略，切分数据量少的核。

    ```
    // 计算尾核数量
    uint32_t tailNum = numBlocks - formerNum;
    // 计算尾核的数据量
    uint32_t tailLength = (totalLengthAligned / numBlocks / alignNum) * alignNum;
    ```

    再按[尾块Tiling](尾块Tiling.md)中的切分策略，计算尾块长度。

    ```
    TilingParamsCalc(tailLength, alignNum, tailTileNum, tailTileLength, tailLastTileLength);
    
    void TilingParamsCalc(uint32_t length, uint32_t alignNum, uint32_t& tileNum, uint32_t& tileLength,
                          uint32_t& lastTileLength)
    {
        tileNum = length / (alignNum * UB_BLOCK_NUM);
        if (tileNum == 0U) {
            tileLength = 0U;
            lastTileLength = static_cast<uint32_t>(((length + alignNum - 1) / alignNum) * alignNum);
        } else if (static_cast<uint32_t>(length / alignNum) % UB_BLOCK_NUM == 0U) {
            tileLength = UB_BLOCK_NUM * alignNum;
            lastTileLength = 0U;
        } else {
            tileLength = UB_BLOCK_NUM * alignNum;
            lastTileLength = static_cast<uint32_t>(length - tileNum * tileLength);
        }
    }
    ```

## 算子类实现<a name="zh-cn_topic_0000002236277533_section182075114713"></a>

Kernel侧Init函数和Process函数的实现需将[核间均分场景下的尾块处理](尾块Tiling.md)与[核间不均分场景下的尾核处理](尾核Tiling.md)的实现结合起来。

Init函数中由于整核和尾核对应的tileLength和lastTileLength不同。因此需按照[核间不均分场景下的尾核处理](尾核Tiling.md)中提到的分别处理整核和尾核。后续对主块和尾块的CopyIn、Compute、CopyOut函数的处理方式与[核间均分场景下的处理](尾块Tiling.md)方式相同。

Init函数实现代码如下：

```
__aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling, AscendC::TPipe* pipeIn)
{
    pipe = pipeIn;
    if (AscendC::GetBlockIdx() < tiling.formerNum) {
        this->tileNum = tiling.formerTileNum;
        this->tileLength = tiling.formerTileLength;
        this->lastTileLength = tiling.formerLastTileLength;
        uint64_t offset = tiling.formerLength * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ half *)x + offset, tiling.formerLength);
        yGm.SetGlobalBuffer((__gm__ half *)y + offset, tiling.formerLength);
        zGm.SetGlobalBuffer((__gm__ half *)z + offset, tiling.formerLength);
    } else {
        this->tileNum = tiling.tailTileNum;
        this->tileLength = tiling.tailTileLength;
        this->lastTileLength = tiling.tailLastTileLength;
        uint64_t offset = tiling.formerLength * tiling.formerNum
                          + tiling.tailLength * (AscendC::GetBlockIdx() - tiling.formerNum);
        xGm.SetGlobalBuffer((__gm__ half *)x + offset, tiling.tailLength);
        yGm.SetGlobalBuffer((__gm__ half *)y + offset, tiling.tailLength);
        zGm.SetGlobalBuffer((__gm__ half *)z + offset, tiling.tailLength);
    }
    
    // 只有尾块的场景下，tileLength为0，因此取tileLength和lastTileLength的最大值来初始化
    uint32_t initBufferLength = AscendC::Std::max(this->tileLength, this->lastTileLength);
    pipe->InitBuffer(inQueueX, 1, this->initBufferLength * sizeof(half));
    pipe->InitBuffer(inQueueY, 1, this->initBufferLength * sizeof(half));
    pipe->InitBuffer(outQueueZ, 1, this->initBufferLength * sizeof(half));
}
```

