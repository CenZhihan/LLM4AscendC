# 运行程序（崩溃时生成 core 文件）
./your_executable

# 使用 GDB 分析 coredump
gdb <executable> <core_file>

# GDB 常用命令 bt              # 查看调用栈 bt full         # 查看完整调用栈（包含局部变量） frame N         # 切换到第 N 层栈帧 info locals     # 查看局部变量 p variable      # 打印变量值
