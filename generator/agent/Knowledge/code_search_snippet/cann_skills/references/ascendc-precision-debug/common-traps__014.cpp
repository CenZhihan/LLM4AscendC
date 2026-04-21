enum class RoundMode {
    CAST_NONE = 0,   // 无精度损失时不舍入，有精度损失时等同 CAST_RINT
    CAST_RINT,       // 四舍六入五成双（银行家舍入）
    CAST_FLOOR,      // 向负无穷舍入
    CAST_CEIL,       // 向正无穷舍入
    CAST_ROUND,      // 四舍五入
    CAST_TRUNC,      // 向零舍入
    CAST_ODD,        // 最近邻奇数舍入
    CAST_HYBRID,     // 随机舍入（特定场景）
};
