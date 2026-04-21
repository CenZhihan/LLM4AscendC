必需:
  --level {L0,L1} [...]  用例级别（必填，支持多个）
                        L0: 单因子覆盖（≤200条）
                        L1: 两两组合覆盖（500~700条）

可选:
  --aclnn-name NAME      算子名称（默认从参数定义中提取）
  --target-count N       L1目标用例数量（默认500，仅L1有效）
                        【L0使用此参数会报错退出】
  --seed N               随机数种子（用于复现L1补齐，仅L1有效）
                        【L0使用此参数会报错退出】
  --report-output FILE   覆盖度报告文件名（默认: {level}_coverage_report.yaml）
                        批量生成时自动添加级别前缀（L0_, L1_）
  --case-output FILE     测试用例文件名（默认: {level}_test_cases.csv）
                        批量生成时自动添加级别前缀（L0_, L1_）
  --verbose              详细输出模式
