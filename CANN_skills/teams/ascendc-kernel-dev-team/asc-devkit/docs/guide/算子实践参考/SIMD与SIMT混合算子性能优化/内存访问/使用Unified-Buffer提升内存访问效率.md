# 使用Unified Buffer提升内存访问效率<a name="ZH-CN_TOPIC_0000002511242528"></a>

>[!NOTE]说明 
>该性能优化建议适用于如下型号：
>-   Ascend 950PR/Ascend 950DT

【优先级】高

【描述】SIMT访问Global Memory的粒度为128B，在随机访问Global Memory中的数据时，访存效率较低。当所需访问的数据量远小于最大可用Unified Buffer空间（256KB - 系统预留8KB - 最小Dcache 32KB）时，可以使用SIMD搬运接口将数据从Global Memory搬运到Unified Buffer，使SIMT编程能够直接从Unified Buffer读取数据，从而提高内存访问效率，提升算子的整体性能。

【样例介绍】以SIMD与SIMT混合编程方式实现的gather算子为例，该算子从长度为8192的一维向量中获取指定索引的65536个数据。通过将输入数据预先搬运到Unified Buffer中，提高离散内存访问的效率，从而提升算子的整体性能。

**表 1**  算子规格

<a name="table336121125718"></a>
<table><thead align="left"><tr id="row153611165711"><th class="cellrowborder" valign="top" width="20%" id="mcps1.2.6.1.1"><p id="p136120185710"><a name="p136120185710"></a><a name="p136120185710"></a>名称</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.2.6.1.2"><p id="p1323211157579"><a name="p1323211157579"></a><a name="p1323211157579"></a>name</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.2.6.1.3"><p id="p182321715105719"><a name="p182321715105719"></a><a name="p182321715105719"></a>shape</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.2.6.1.4"><p id="p3233415135710"><a name="p3233415135710"></a><a name="p3233415135710"></a>data type</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.2.6.1.5"><p id="p1523341515715"><a name="p1523341515715"></a><a name="p1523341515715"></a>format</p>
</th>
</tr>
</thead>
<tbody><tr id="row163625113578"><td class="cellrowborder" rowspan="2" valign="top" width="20%" headers="mcps1.2.6.1.1 "><p id="p8423914575"><a name="p8423914575"></a><a name="p8423914575"></a>算子输入</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.6.1.2 "><p id="p1387892665716"><a name="p1387892665716"></a><a name="p1387892665716"></a>input</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.6.1.3 "><p id="p11878726195716"><a name="p11878726195716"></a><a name="p11878726195716"></a>8192</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.6.1.4 "><p id="p1287822610573"><a name="p1287822610573"></a><a name="p1287822610573"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.6.1.5 "><p id="p48783266570"><a name="p48783266570"></a><a name="p48783266570"></a>ND</p>
</td>
</tr>
<tr id="row133621117575"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p787812612573"><a name="p787812612573"></a><a name="p787812612573"></a>index</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1187842645716"><a name="p1187842645716"></a><a name="p1187842645716"></a>65536</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p128789266579"><a name="p128789266579"></a><a name="p128789266579"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p3878152615711"><a name="p3878152615711"></a><a name="p3878152615711"></a>ND</p>
</td>
</tr>
<tr id="row13621016571"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.6.1.1 "><p id="p536291185717"><a name="p536291185717"></a><a name="p536291185717"></a>算子输出</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.6.1.2 "><p id="p2087802613572"><a name="p2087802613572"></a><a name="p2087802613572"></a>output</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.6.1.3 "><p id="p48785261571"><a name="p48785261571"></a><a name="p48785261571"></a>65536</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.6.1.4 "><p id="p128781326125711"><a name="p128781326125711"></a><a name="p128781326125711"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.6.1.5 "><p id="p18785262574"><a name="p18785262574"></a><a name="p18785262574"></a>ND</p>
</td>
</tr>
</tbody>
</table>

SIMT线程层次结构为：

-   线程块数：64
-   单个线程块中线程数：1024

完整样例请参考[SIMD与SIMT混合编程使用UB提高内存访问效率](https://gitcode.com/cann/asc-devkit/tree/master/examples/01_simd_cpp_api/04_best_practices/03_fusion_compute_practices/simt_gather_with_ub)。

【反例】

SIMT随机访问Global Memory上的input数据，触发数据加载到Dcache，随机访存效率低，代码如下。

```
namespace {
    constexpr uint32_t THREAD_COUNT = 1024;
    constexpr uint32_t INPUT_SIZE = 8192;
    constexpr uint32_t INDEX_SIZE = 65536;
}

__simt_vf__ __launch_bounds__(THREAD_COUNT) inline void simt_gather(
    __gm__ float* input,
    __gm__ uint32_t* index,
    __gm__ float* output)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= INDEX_SIZE) {
        return;
    }

    uint32_t gather_idx = index[idx];
    if (gather_idx > INPUT_SIZE) {
        return;
    }

    output[idx] = input[gather_idx];
}


__global__ __vector__ void gather_kernel(__gm__ float* input, __gm__ uint32_t* index, __gm__ float* output)
{
    asc_vf_call<simt_gather>(dim3(THREAD_COUNT), input, index, output);
}
```

【正例】

使用SIMD接口将数据从Global Memory搬运到Unified Buffer，基于SIMT编程方式直接从Unified Buffer读取数据，访存效率远高于读取Global Memory上的数据，代码如下。

```
namespace {
    constexpr uint32_t THREAD_COUNT = 1024;
    constexpr uint32_t INPUT_SIZE = 8192;
    constexpr uint32_t INDEX_SIZE = 65536;
}

__simt_vf__ __launch_bounds__(THREAD_COUNT) inline void simt_gather(
    __ubuf__ float* input,
    __gm__ uint32_t* index,
    __gm__ float* output)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= INDEX_SIZE) {
        return;
    }

    uint32_t gather_idx = index[idx];
    if (gather_idx >= INPUT_SIZE) {
        return;
    }

    output[idx] = input[gather_idx];
}


__global__ __vector__ void gather_kernel(__gm__ float* input, __gm__ uint32_t* index, __gm__ float* output)
{
    __ubuf__ float input_buf[INPUT_SIZE];
    uint32_t blk_length = INPUT_SIZE * sizeof(float);
    asc_copy_gm2ub_align(input_buf, input, 1, blk_length, 0, 0, false, 0, 0, 0);

    if ASC_IS_AIV {
        asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
        asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);
    }

    asc_vf_call<simt_gather>(dim3(THREAD_COUNT), input_buf, index, output);
}
```

【性能对比】

下图为反例的流水图，线程中有两条SIMT\_LDG指令，对应表示从Global Memory上加载数据，其中第二条指令占用区间分布不均，指令启动时间差异明显，同一个线程块中随机访问输入数据input，单次访存加载128B数据，而针对单精度浮点数，实际有效数据仅为4B，导致访存效率低。

**图 1**  反例指令流水图（仿真数据）<a name="fig1771716554412"></a>  
![](../../../figures/反例指令流水图（仿真数据）.png "反例指令流水图（仿真数据）")

下图为反例的内存负载分析图，L2 Cache到Dcache数据传输带宽为10.04GB/s。

**图 2**  反例内存负载分析（上板数据）<a name="fig14500206121615"></a>  
![](../../../figures/反例内存负载分析（上板数据）.png "反例内存负载分析（上板数据）")

下图为正例的流水图，只有一条占用大区间的SIMT\_LDG指令，MTE2流水新增搬运指令MOV\_SRC\_TO\_DST\_ALIGNv2。

**图 3**  正例指令流水图（仿真数据）<a name="fig419412567501"></a>  
![](../../../figures/正例指令流水图（仿真数据）.png "正例指令流水图（仿真数据）")

下图为正例的内存负载分析图，L2 Cache到Dcache数据传输带宽降低为1.61GB/s，L2 Cache到Unified Buffer数据传输带宽提升至12.93GB/s。

**图 4**  正例内存负载分析（上板数据）<a name="fig2368628165417"></a>  
![](../../../figures/正例内存负载分析（上板数据）.png "正例内存负载分析（上板数据）")

对比算子运行时间，反例约为4.56us，正例约为3.57us，整体性能提升约21.71%。

**图 5**  反例算子运行时间（上板数据）<a name="fig207642436475"></a>  
![](../../../figures/反例算子运行时间（上板数据）.png "反例算子运行时间（上板数据）")

**图 6**  正例算子运行时间（上板数据）<a name="fig1387010585528"></a>  
![](../../../figures/正例算子运行时间（上板数据）.png "正例算子运行时间（上板数据）")

