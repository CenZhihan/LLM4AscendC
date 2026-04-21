def verify_report(result, branch_name):
    # 检查 0: ⚠️ 审视代码结果（强制，缺少则直接失败）
    if "审视代码结果" not in result:
        print(f"❌ 报告缺少：审视代码结果，步骤 6 未执行")
        return False
