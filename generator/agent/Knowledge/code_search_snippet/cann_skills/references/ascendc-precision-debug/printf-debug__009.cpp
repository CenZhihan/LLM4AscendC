// 定义调试开关
#define DEBUG_PRECISION 1

#if DEBUG_PRECISION
    #define DEBUG_PRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
    #define DEBUG_PRINT(fmt, ...)
#endif

// 使用
DEBUG_PRINT("Debug info: value=%.6f\n", value);
