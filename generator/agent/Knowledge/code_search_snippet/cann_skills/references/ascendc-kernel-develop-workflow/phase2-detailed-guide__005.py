def verify_report(result, branch_name):
    """验证 通用 Agent A 的实现报告"""
    
    # 检查 0: ⚠️ 审视代码结果（强制，缺少则直接失败）
    if "审视代码结果" not in result:
        print(f"❌ 报告缺少：审视代码结果，步骤 6 未执行")
        print(f"❌ 验收直接失败，评分：0/10")
        return False
    
    review = result["审视代码结果"]
    required_review_fields = ["审视结果", "发现问题", "修复动作"]
    for field in required_review_fields:
        if field not in review:
            print(f"❌ 审视代码结果缺少：{field}")
            return False
    
    # 检查审视结果的完整性
    if "审视结果" in review:
        required_checks = ["Tiling计算位置", "CMake配置", "API使用", "数据类型"]
        for check in required_checks:
            if check not in review["审视结果"]:
                print(f"❌ 审视代码结果不完整，缺少检查项：{check}")
                return False
    
    print(f"✅ 步骤 6 审视代码已执行且完整")
    
    # 检查 1: 报告完整性（5 个部分）
    required_parts = ['环境信息', '编译结果', 
                      '测试结果', '文件清单', '验证命令']
    for part in required_parts:
        if part not in result:
            print(f"❌ 报告缺少：{part}")
            return False
    
    # 检查 2: 文件存在
    branch_file = f"ops/{operator_name}/{operator_name}_{branch_name}.h"
    if not os.path.exists(branch_file):
        print(f"❌ 文件不存在：{branch_file}")
        return False
    
    # 检查 3: 编译产物
    binary = f"ops/{operator_name}/build/{operator_name}_custom"
    if not os.path.exists(binary):
        print(f"❌ 编译产物不存在")
        return False
    
    # 检查 4: 所有验证命令可执行成功
    verify_cmd = result.get('验证命令', {})
    for key, cmd in verify_cmd.items():
        exit_code = bash(cmd).returncode
        if exit_code != 0:
            print(f"❌ 验证命令失败：{key}")
            return False
    
    print(f"✅ {branch_name} 实现报告验证通过")
    return True
