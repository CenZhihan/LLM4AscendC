{
    # 完整报告（包含审视代码+编译+测试）
    "环境信息": {
        "CANN 版本": "9.0.0",
        "NPU 可用": true,
        "芯片型号": "dav-2201"
    },
    
    "审视代码结果": {
        "设计一致性": "通过/有问题",
        "API 正确性": "通过/有问题",
        "编码规范": "通过/有问题",
        "问题列表": [...]
    },
    
    "编译结果": {
        "命令": "cmake .. && make",
        "输出": "成功/失败",
        "错误信息": ""
    },
    
    "测试结果": [
        {
            "用例": "1",
            "Shape": "(4, 128)",
            "Axis": -1,
            "Dtype": "FP32",
            "结果": "通过/失败"
        }
    ],
    
    "文件清单": [
        "ops/{operator_name}/CMakeLists.txt",
        "ops/{operator_name}/{operator_name}_common.h",
        "ops/{operator_name}/{operator_name}_{分支名}.h"
    ],
    
    "验证命令": {
        "编译": "cd ops/{operator_name}/build && cmake .. && make",
        "运行": "./build/{operator_name}_custom",
        "精度验证": "python verify_precision.py"
    },
    
    "实现总结": "..."
}
