usage: generate_factor_values.py [-h] [--max-cases N] [--seed N] [--verbose] solver_config constraints factors output

参数说明:
  solver_config  求解配置YAML文件（06_求解配置.yaml）
  constraints    约束定义YAML文件（05_约束定义.yaml）
  factors        测试因子YAML文件（04_测试因子.yaml）
  output         输出的CSV文件（07_因子值.csv）
  --max-cases    最大用例数（默认10000）
  --seed         随机数种子（用于复现结果）
  --verbose      详细输出模式
