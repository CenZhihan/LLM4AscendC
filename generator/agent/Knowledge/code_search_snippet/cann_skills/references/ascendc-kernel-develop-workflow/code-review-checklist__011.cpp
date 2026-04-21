// 单行（AR 模板）
DataCopyPad(dst, src, 
    {rowsThisLoop, rLength, rLengthAlign, 0, 0});
// - rowsThisLoop: 处理的行数
// - rLength: 有效数据长度（非对齐）
// - rLengthAlign: 对齐后长度（UB 存储）

// 多维（ARA 模板）
DataCopyPad(dst, src,
    {rLength, a0TileLen, a0Length, rLengthAlign, a0TileLen});
// - rLength: R 维度有效长度
// - a0TileLen: A0 单次处理长度
// - a0Length: A0 总长度
// - rLengthAlign: R 维度对齐长度（UB 存储）
// - a0TileLen: A0 维度对齐长度（UB 存储）
