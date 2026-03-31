# asc_release_buf
## 产品支持情况

| 产品 | 是否支持  |
| :-----------| :------: |
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

用于AI Core内部异步通道同步的指令，用于释放由asc_get_buf指令申请的缓存。

## 函数原型

    ```cpp
    __aicore__ inline void asc_release_buf(pipe_t pipe, uint8_t buf_id, bool mode)
    __aicore__ inline void asc_release_buf(pipe_t pipe, uint64_t buf_id, bool mode)
    ```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :---  | :--- | :--- |
| pipe | 输入 | 该指令的执行流水线。 |
| buf_id | 输入 | buffer标号。|
| mode | 输入 | 执行模式。<br>&bull; false：该指令等待pipe所对应的流水线中所有前置指令完成后执行。<br>&bull; true：该指令等待pipe所对应的流水线中所有前置指令完成且相同buf_id的所有asc_release_buf指令执行完成后执行。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

- 必须与asc_get_buf搭配使用，且需要使用相同的buf_id与mode。此外，asc_release_buf必须始终插在对应的asc_get_buf指令之后，否则硬件行为无定义。
- 具有相同buf_id的asc_get_buf与asc_release_buf组合，无论pipe与mode是否相同，均不得在编程顺序中嵌套，否则硬件行为将不可预测。
- 对于程序顺序中连续出现的，具有相同pipe与buf_id的指令对，后一个asc_get_buf将不再阻塞流水线运行，若需实现同一流水线的依赖关系，则必须使用指令asc_sync_data_barrier。

## 调用示例

```cpp
//buffer标号为1
int64_t buf_id = 1;
//等待PIPE_S中所有前置指令完成后释放标号为1的缓存
bool mode = false;
asc_get_buf(PIPE_S, buf_id, mode);  
asc_release_buf(PIPE_S, buf_id, mode);  
```
