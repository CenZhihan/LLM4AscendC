{
    "审视阶段": "步骤6：审视代码",
    "审视结果": {
        "数据搬运API": "❌ 错误 - 使用了 DataCopy，应使用 DataCopyPad"
    },
    "发现问题": ["发现 DataCopy(xLocal, xGm, ...) - 违反 GM-UB 数据搬运规范"],
    "修复动作": ["将 DataCopy 替换为 DataCopyPad"],
    "状态": "❌ 审视不通过 - 必须修复后重新审视"
}
