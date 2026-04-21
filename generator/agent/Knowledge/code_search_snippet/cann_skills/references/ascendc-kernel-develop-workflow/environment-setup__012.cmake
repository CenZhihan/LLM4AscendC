# ❌ 错误：手动指定编译器路径
set(ASCEND_HOME $ENV{ASCEND_HOME})  # 错误的环境变量名
add_custom_target(kernel
    COMMAND ${ASCEND_HOME}/compiler/tikcpp/tikcc ...  # 错误的编译器
    COMMAND ${ASCEND_HOME}/compiler/tikcpp/bangc ...  # 错误的编译器
)
