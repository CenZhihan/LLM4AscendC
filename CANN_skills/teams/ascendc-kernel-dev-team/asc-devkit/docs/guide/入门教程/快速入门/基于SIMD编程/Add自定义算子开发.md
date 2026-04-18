# Add自定义算子开发<a name="ZH-CN_TOPIC_0000002500781060"></a>

本入门教程，将会引导你完成以下任务，体验Ascend C  SIMD算子开发基本流程。

1.  算子分析，明确数学表达式和计算逻辑等内容；
2.  Add算子核函数开发；
3.  算子核函数运行验证。

在正式的开发之前，还需要先完成环境准备工作，开发Ascend C算子的基本流程如下图所示：

**图 1**  开发Ascend C算子的基本流程<a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_fig72899531739"></a>  
![](../../../figures/开发Ascend-C算子的基本流程.png "开发Ascend-C算子的基本流程")

>[!NOTE]说明
>-   请点击[LINK](https://gitcode.com/cann/asc-devkit/tree/master/examples/01_simd_cpp_api/00_introduction/01_add/basic_api_tque_add)获取样例代码。
>-   使用本教程只需要您具有一定的C/C++基础，在此基础上，如果您已经对Ascend C编程模型有一定的了解，您可以在实际操作的过程中加深对理论的理解；如果您还没有开始了解Ascend C编程模型，也无需担心，您可以先尝试跑通教程中的样例，参考教程最后的[指引](#zh-cn_topic_0000001565030288_section128349412384)进行进一步的学习。

## 环境准备<a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_section412999115218"></a>

-   CANN软件安装

    开发算子前，需要先准备好开发环境和运行环境，开发环境和运行环境的介绍和具体的安装步骤可参见《CANN 软件安装指南》。

-   环境变量配置

    安装CANN软件后，使用CANN运行用户进行编译、运行时，需要以CANN运行用户登录环境，执行**source $_\{INSTALL\_DIR\}_/set\_env.sh**命令设置环境变量。$\{INSTALL\_DIR\}请替换为CANN软件安装后文件存储路径。以root用户安装为例，安装后文件默认存储路径为：/usr/local/Ascend/cann。

## 算子分析<a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_section1242991675619"></a>

主要分析算子的数学表达式、输入输出的数量、Shape范围以及计算逻辑的实现，明确需要调用的Ascend C接口。下文以Add算子为例，介绍具体的分析过程。

1.  明确算子的数学表达式及计算逻辑。

    Add算子的数学表达式为：

    ![](../../../figures/zh-cn_formulaimage_0000002501241896.png)

    计算逻辑是：从外部存储Global Memory搬运数据至内部存储Local Memory，然后使用Ascend C计算接口完成两个输入参数相加，得到最终结果，再搬运到Global Memory上。

2.  明确输入和输出。
    -   Add算子有两个输入：x与y，输出为z。
    -   本样例中算子输入支持的数据类型为float，算子输出的数据类型与输入数据类型相同。
    -   算子输入支持的shape为（8，2048），输出shape与输入shape相同。
    -   算子输入支持的[format](../../../编程指南/概念原理和术语/神经网络和算子/数据排布格式.md)为：ND。

3.  确定核函数名称和参数。
    -   本样例中核函数命名为add\_custom。
    -   根据对算子输入输出的分析，确定核函数有3个参数x，y，z；x，y为输入参数，z为输出参数。

4.  确定算子实现所需接口。
    -   实现涉及外部存储和内部存储间的数据搬运，查看Ascend C  API参考中的数据搬运接口，需要使用DataCopy来实现数据搬移。
    -   本样例只涉及矢量计算的加法操作，查看Ascend C  API参考中的矢量计算接口Memory矢量计算，初步分析可使用Add接口Add实现x+y。
    -   计算中使用到的Tensor数据结构，使用AllocTensor、FreeTensor进行申请和释放。
    -   并行流水任务之间使用Queue队列完成同步，会使用到EnQue、DeQue等接口。

通过以上分析，得到Ascend C  Add算子的设计规格如下：

**表 1** Ascend C  Add算子设计规格

<a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_table164881913121819"></a>
<table><tbody><tr id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_row1848911314188"><th class="firstcol" valign="top" id="mcps1.2.6.1.1"><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p54891613141818"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p54891613141818"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p54891613141818"></a>算子类型（OpType）</p>
</th>
<td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.1 "><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p1748971311817"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p1748971311817"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p1748971311817"></a>AddCustom</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_row108131259145713"><th class="firstcol" rowspan="3" valign="top" width="19.71%" id="mcps1.2.6.2.1"><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p84891313191819"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p84891313191819"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p84891313191819"></a>算子输入</p>
</th>
<td class="cellrowborder" valign="top" width="19.250000000000004%" headers="mcps1.2.6.2.1 "><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p1081305915577"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p1081305915577"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p1081305915577"></a><strong id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_b1720613329588"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_b1720613329588"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_b1720613329588"></a>name</strong></p>
</td>
<td class="cellrowborder" valign="top" width="21.970000000000002%" headers="mcps1.2.6.2.1 "><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p481313597575"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p481313597575"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p481313597575"></a><strong id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_b621853235814"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_b621853235814"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_b621853235814"></a>shape</strong></p>
</td>
<td class="cellrowborder" valign="top" width="21.610000000000003%" headers="mcps1.2.6.2.1 "><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p1781315915718"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p1781315915718"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p1781315915718"></a><strong id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_b4228123295815"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_b4228123295815"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_b4228123295815"></a>data type</strong></p>
</td>
<td class="cellrowborder" valign="top" width="17.460000000000004%" headers="mcps1.2.6.2.1 "><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p12813559125720"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p12813559125720"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p12813559125720"></a><strong id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_b622933295812"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_b622933295812"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_b622933295812"></a>format</strong></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_row19489161331815"><td class="cellrowborder" valign="top" headers="mcps1.2.6.2.1 "><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p134891113131814"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p134891113131814"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p134891113131814"></a>x</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.2.1 "><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p1283525118245"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p1283525118245"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p1283525118245"></a>(8, 2048)</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.2.1 "><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p12624946102416"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p12624946102416"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p12624946102416"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.2.1 "><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p173195184235"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p173195184235"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p173195184235"></a>ND</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_row1348951391811"><td class="cellrowborder" valign="top" headers="mcps1.2.6.2.1 "><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p13489191311184"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p13489191311184"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p13489191311184"></a>y</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.2.1 "><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p18710686591"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p18710686591"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p18710686591"></a>(8, 2048)</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.2.1 "><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p76241246152416"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p76241246152416"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p76241246152416"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.2.1 "><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p0319161811238"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p0319161811238"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p0319161811238"></a>ND</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_row2489171310187"><th class="firstcol" valign="top" width="19.71%" id="mcps1.2.6.5.1"><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p4489913171810"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p4489913171810"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p4489913171810"></a>算子输出</p>
</th>
<td class="cellrowborder" valign="top" width="19.250000000000004%" headers="mcps1.2.6.5.1 "><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p34899136188"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p34899136188"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p34899136188"></a>z</p>
</td>
<td class="cellrowborder" valign="top" width="21.970000000000002%" headers="mcps1.2.6.5.1 "><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p13645129155915"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p13645129155915"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p13645129155915"></a>(8, 2048)</p>
</td>
<td class="cellrowborder" valign="top" width="21.610000000000003%" headers="mcps1.2.6.5.1 "><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p4624546192418"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p4624546192418"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p4624546192418"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="17.460000000000004%" headers="mcps1.2.6.5.1 "><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p1396317231699"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p1396317231699"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p1396317231699"></a>ND</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_row628260192410"><th class="firstcol" valign="top" id="mcps1.2.6.6.1"><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p22821601246"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p22821601246"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p22821601246"></a>核函数名称</p>
</th>
<td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.6.1 "><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p1359311404260"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p1359311404260"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p1359311404260"></a>add_custom</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_row24890138181"><th class="firstcol" rowspan="4" valign="top" id="mcps1.2.6.7.1"><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p64894131189"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p64894131189"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p64894131189"></a>使用的主要接口</p>
</th>
<td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.7.1 "><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p193715912115"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p193715912115"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p193715912115"></a>DataCopy：数据搬运接口</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_row9293141202412"><td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.7.1 "><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p2293241142412"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p2293241142412"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p2293241142412"></a>Add：矢量基础算术接口</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001565030288_row2463132420116"><td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.7.1 "><p id="zh-cn_topic_0000001565030288_p10878133151120"><a name="zh-cn_topic_0000001565030288_p10878133151120"></a><a name="zh-cn_topic_0000001565030288_p10878133151120"></a>AllocTensor、FreeTensor：内存管理接口</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_row1036344318916"><td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.7.1 "><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p4174848399"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p4174848399"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p4174848399"></a>EnQue、DeQue接口：Queue队列管理接口</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_row54819813236"><th class="firstcol" valign="top" id="mcps1.2.6.11.1"><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p74958142316"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p74958142316"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_zh-cn_topic_0229825559_zh-cn_topic_0229823837_zh-cn_topic_0211294710_p74958142316"></a>算子实现文件名称</p>
</th>
<td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.11.1 "><p id="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p1180818388211"><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p1180818388211"></a><a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_p1180818388211"></a>add_custom.asc</p>
</td>
</tr>
</tbody>
</table>

## 核函数开发<a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_section822273613219"></a>

完成环境准备和初步的算子分析后，即可开始Ascend C核函数的开发。开发之前请先从[LINK](https://gitcode.com/cann/asc-devkit/tree/master/examples/01_simd_cpp_api/00_introduction/01_add/basic_api_tque_add)获取样例代码，以下样例代码在add\_custom.asc中实现。

本样例中使用多核并行计算，即把数据进行分片，分配到多个核上进行处理。Ascend C核函数是在一个核上的处理函数，所以只处理部分数据。分配方案是：假设共启用8个核，数据整体长度为8 \* 2048个元素，平均分配到8个核上运行，每个核上处理的数据大小为2048个元素。对于单核上的处理数据，也可以进行数据切块，实现对数据的流水并行处理。

1.  根据分配方案设计一个结构体AddCustomTilingData，用于保存并行数据切分相关的参数。AddCustomTilingData定义了两个参数：totalLength、tileNum。totalLength指待处理的数据总大小为（8 \* 2048）个元素，tileNum指每个核需要计算的数据块个数。

    ```
    struct AddCustomTilingData
    {
        uint32_t totalLength;
        uint32_t tileNum;
    };
    ```

2.  根据[核函数定义和调用](../../../编程指南/编程模型/AI-Core-SIMD编程/核函数.md#zh-cn_topic_0000001447989210_section1915102519220)中介绍的规则进行**核函数的定义，**并在核函数中调用算子类的Init和Process函数，算子类实现在后续步骤中介绍。

    ```
    __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
    {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);    // 设置Kernel类型为Vector核（用于矢量计算）
        KernelAdd op;
        op.Init(x, y, z, tiling.totalLength, tiling.tileNum);
        op.Process();
    }
    ```

    -   使用\_\_global\_\_函数类型限定符来标识它是一个核函数，可以被<<<\>\>\>调用；使用\_\_aicore\_\_函数类型限定符来标识该核函数在设备端AI Core上执行。指针入参变量需要增加变量类型限定符\_\_gm\_\_，表明该指针变量指向Global Memory上某处内存地址。为了统一表达，使用GM\_ADDR宏来修饰入参，GM\_ADDR宏定义如下：

        ```
        #define GM_ADDR __gm__ uint8_t*
        ```

    -   算子类的Init函数，完成内存初始化相关工作，Process函数完成算子实现的核心逻辑。

3.  **根据矢量编程范式实现算子类**，本样例中定义KernelAdd算子类，其具体成员如下**：**

    ```
    class KernelAdd {
    public:
        __aicore__ inline KernelAdd(){}
        // 初始化函数，完成内存初始化相关操作
        __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum){}
        // 核心处理函数，实现算子逻辑，调用私有成员函数CopyIn、Compute、CopyOut完成矢量算子的三级流水操作
        __aicore__ inline void Process(){}
    
    private:
        // 搬入函数，从Global Memory搬运数据至Local Memory，被核心Process函数调用
        __aicore__ inline void CopyIn(int32_t progress){}
        // 计算函数，完成两个输入参数相加，得到最终结果，被核心Process函数调用
        __aicore__ inline void Compute(int32_t progress){}
        // 搬出函数，将最终结果从Local Memory搬运到Global Memory上，被核心Process函数调用
        __aicore__ inline void CopyOut(int32_t progress){}
    
    private:
        AscendC::TPipe pipe;  // TPipe内存管理对象
        AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;  // 输入数据Queue队列管理对象，TPosition为VECIN
        AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;  // 输出数据Queue队列管理对象，TPosition为VECOUT
        AscendC::GlobalTensor<float> xGm;  // 管理输入输出Global Memory内存地址的对象，其中xGm, yGm为输入，zGm为输出
        AscendC::GlobalTensor<float> yGm;
        AscendC::GlobalTensor<float> zGm;
        uint32_t blockLength; // 每个核的计算数据长度
        uint32_t tileNum; // 每个核需要计算的数据块个数
        uint32_t tileLength; // 每个核内每个数据块的长度
    };
    ```

    内部函数的调用关系示意图如下：

    **图 2**  核函数调用关系图<a name="zh-cn_topic_0000001565030288_fig8676257399"></a>  
    ![](../../../figures/核函数调用关系图.png "核函数调用关系图")

    由此可见除了Init函数完成初始化外，Process中完成了对**流水任务**“搬入、计算、搬出”的调用，开发者可以重点关注三个流水任务的实现。

4.  初始化函数**Init**主要完成以下内容：设置输入输出Global Tensor的Global Memory内存地址，通过TPipe内存管理对象为输入输出Queue分配内存。

    **上文我们介绍到，本样例将数据切分成8块，平均分配到8个核上运行，每个核上处理的数据大小为2048个元素。那么我们是如何实现这种切分的呢？**

    每个核上处理的数据地址需要在起始地址上增加GetBlockIdx\(\) \* blockLength（每个block处理的数据长度）的偏移来获取。这样也就实现了多核并行计算的数据切分。

    以输入x为例，x + blockLength \* GetBlockIdx\(\)即为单核处理程序中x在Global Memory上的内存偏移地址，获取偏移地址后，使用GlobalTensor类的SetGlobalBuffer接口设定该核上Global Memory的起始地址以及长度。具体示意图如下。

    **图 3**  多核并行处理示意图<a name="zh-cn_topic_0000001565030288_fig744312161511"></a>  
    ![](../../../figures/多核并行处理示意图.png "多核并行处理示意图")

    **上面已经实现了多核数据的切分，那么单核上的处理数据如何进行切分？**

    对于单核上的处理数据，可以进行数据切块（Tiling），在本示例中，仅作为参考，将数据切分成8块（并不意味着8块就是性能最优）。切分后的每个数据块再次切分成2块，即可开启[double buffer](../../../编程指南/概念原理和术语/性能优化技术原理/DoubleBuffer.md)，实现流水线之间的并行。

    这样单核上的数据（2048个数）被切分成16块，每块tileLength（128）个数据。TPipe为inQueueX分配了两块大小为tileLength \* sizeof\(float\)个字节的内存块，每个内存块能容纳tileLength（128）个float类型数据。数据切分示意图如下。

    **图 4**  单核数据切分示意图<a name="zh-cn_topic_0000001565030288_fig184434221514"></a>  
    ![](../../../figures/单核数据切分示意图.png "单核数据切分示意图")

    具体的初始化函数代码如下：

    ```
    // Kernel侧所需头文件
    #include "kernel_operator.h"
    
    constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        
         this->blockLength = totalLength / AscendC::GetBlockNum();     // length computed of each core
         this->tileNum = tileNum;                                      // split data into 8 tiles for each core
         this->tileLength = this->blockLength / tileNum / BUFFER_NUM;  // separate to 2 parts, due to double buffer
         // get start index for current core, core parallel
         xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
         yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
         zGm.SetGlobalBuffer((__gm__ float *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
         // pipe alloc memory to queue, the unit is Bytes
         pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
         pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
         pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
    }
    ```

5.  基于矢量编程范式，将核函数的实现分为3个基本任务：CopyIn，Compute，CopyOut。**Process**函数中通过如下方式调用这三个函数。

    ```
    __aicore__ inline void Process()
    {
        // loop count need to be doubled, due to double buffer
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        // tiling strategy, pipeline parallel
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }
    ```

    1.  CopyIn函数实现。

        1.  使用DataCopy接口将GlobalTensor数据拷贝到LocalTensor。
        2.  使用EnQue将LocalTensor放入VecIn的Queue中。

        ```
        __aicore__ inline void CopyIn( int32_t progress)
        {
            // alloc tensor from queue memory
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            AscendC::LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
            // copy progress_th tile from global tensor to local tensor
            AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
            AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
            // enque input tensors to VECIN queue
            inQueueX.EnQue(xLocal);
            inQueueY.EnQue(yLocal);
        }
        ```

    2.  Compute函数实现。

        1.  使用DeQue从VecIn中取出LocalTensor。
        2.  使用Ascend C接口Add完成矢量计算。
        3.  使用EnQue将计算结果LocalTensor放入到VecOut的Queue中。
        4.  使用FreeTensor将释放不再使用的LocalTensor。

        ```
        __aicore__ inline void Compute(int32_t progress)
        {
            // deque input tensors from VECIN queue
            AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
            AscendC::LocalTensor<float> yLocal = inQueueY.DeQue<float>();
            AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
            // call Add instr for computation
            AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
            // enque the output tensor to VECOUT queue
            outQueueZ.EnQue<float>(zLocal);
            // free input tensors for reuse
            inQueueX.FreeTensor(xLocal);
            inQueueY.FreeTensor(yLocal);
        }
        ```

    3.  CopyOut函数实现。

        1.  使用DeQue接口从VecOut的Queue中取出LocalTensor。
        2.  使用DataCopy接口将LocalTensor拷贝到GlobalTensor上。
        3.  使用FreeTensor将不再使用的LocalTensor进行回收。

        ```
         __aicore__ inline void CopyOut(int32_t progress)
        {
            // deque output tensor from VECOUT queue
            AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
            // copy progress_th tile from local tensor to global tensor
            AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
            // free output tensor for reuse
            outQueueZ.FreeTensor(zLocal);
        }
        ```

## 核函数运行验证<a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_section136721210114419"></a>

完成Kernel侧核函数开发后，即可编写Host侧的核函数调用程序。实现从Host侧的APP程序调用算子，执行计算过程。

1.  Host侧应用程序框架的编写。

    ```
    // Host侧应用程序需要包含的头文件
    #include "acl/acl.h"
    // Kernel侧需要包含的头文件
    #include "kernel_operator.h"
    // 核函数开发部分
    ...
    
    __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
    {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);    
        KernelAdd op;
        op.Init(x, y, z, tiling.totalLength, tiling.tileNum);
        op.Process();
    }
    
    // 通过<<<...>>>内核调用符调用算子
    std::vector<float> kernel_add(std::vector<float> &x, std::vector<float> &y)
    {
    ...
    }
    
    
    // 计算结果比对
    uint32_t VerifyResult(std::vector<float> &output, std::vector<float> &golden)
    {
        auto printTensor = [](std::vector<float> &tensor, const char *name) {
            constexpr size_t maxPrintSize = 20;
            std::cout << name << ": ";
            std::copy(tensor.begin(), tensor.begin() + std::min(tensor.size(), maxPrintSize),
                std::ostream_iterator<float>(std::cout, " "));
            if (tensor.size() > maxPrintSize) {
                std::cout << "...";
            }
            std::cout << std::endl;
        };
        printTensor(output, "Output");
        printTensor(golden, "Golden");
        if (std::equal(golden.begin(), golden.end(), output.begin())) {
            std::cout << "[Success] Case accuracy is verification passed." << std::endl;
            return 0;
        } else {
            std::cout << "[Failed] Case accuracy is verification failed!" << std::endl;
            return 1;
        }
        return 0;
    }
    
    // 算子验证主程序
    int32_t main(int32_t argc, char *argv[])
    {
        constexpr uint32_t totalLength = 8 * 2048;
        constexpr float valueX = 1.2f;
        constexpr float valueY = 2.3f;
        std::vector<float> x(totalLength, valueX);
        std::vector<float> y(totalLength, valueY);
    
        std::vector<float> output = kernel_add(x, y);
    
        std::vector<float> golden(totalLength, valueX + valueY);
        return VerifyResult(output, golden);
    }
    ```

2.  编写**通过<<<...\>\>\>内核调用符调用**算子的代码。

    **图 5**  调用步骤<a name="zh-cn_topic_0000001565030288_fig1034911325594"></a>  
    ![](../../../figures/调用步骤.png "调用步骤")

    如下示例中的acl API使用方法请参考《应用开发指南 \(C&C++\)》中的“Runtime API参考 > Stream管理”章节。

    ```
    std::vector<float> kernel_add(std::vector<float> &x, std::vector<float> &y)
    {
        constexpr uint32_t numBlocks = 8;
        uint32_t totalLength = x.size();
        size_t totalByteSize = totalLength * sizeof(float);
        int32_t deviceId = 0;
        aclrtStream stream = nullptr;
        AddCustomTilingData tiling = {/*totalLength:*/totalLength, /*tileNum:*/8};
        uint8_t *xHost = reinterpret_cast<uint8_t *>(x.data());
        uint8_t *yHost = reinterpret_cast<uint8_t *>(y.data());
        uint8_t *zHost = nullptr;
        uint8_t *xDevice = nullptr;
        uint8_t *yDevice = nullptr;
        uint8_t *zDevice = nullptr;
    
        // 初始化
        aclInit(nullptr);
        // 运行管理资源申请
        aclrtSetDevice(deviceId);
        aclrtCreateStream(&stream);
        // 分配Host内存
        aclrtMallocHost((void **)(&zHost), totalByteSize);
        // 分配Device内存
        aclrtMalloc((void **)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&yDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&zDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        // 将Host上的输入数据拷贝到Device侧
        aclrtMemcpy(xDevice, totalByteSize, xHost, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(yDevice, totalByteSize, yHost, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        // 用内核调用符<<<...>>>调用核函数完成指定的运算
        add_custom<<<numBlocks, nullptr, stream>>>(xDevice, yDevice, zDevice, tiling);
        aclrtSynchronizeStream(stream);
        // 将Device上的运算结果拷贝回Host
        aclrtMemcpy(zHost, totalByteSize, zDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        std::vector<float> z((float *)zHost, (float *)(zHost + totalByteSize));
        // 释放申请的资源
        aclrtFree(xDevice);
        aclrtFree(yDevice);
        aclrtFree(zDevice);
        aclrtFreeHost(zHost);
        // 去初始化
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return z;
    }
    ```

3.  CMake编译配置如下：

    ```
    cmake_minimum_required(VERSION 3.16)
    # find_package(ASC)是CMake中用于查找和配置Ascend C编译工具链的命令
    find_package(ASC REQUIRED)
    # 指定项目支持的语言包括ASC和CXX，ASC表示支持使用毕昇编译器对Ascend C编程语言进行编译
    project(kernel_samples LANGUAGES ASC CXX)
    
    add_executable(demo
        add_custom.asc
    )
    
    # 通过编译选项设置NPU架构
    target_compile_options(demo PRIVATE   
       $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
    )
    ```

4.  编译和运行步骤如下

    ```
    mkdir -p build && cd build; 
    cmake ..;make -j;
    ./demo
    ```

    >[!NOTE]说明 
    >-   该样例仅支持如下型号：
    >    -   Ascend 950PR/Ascend 950DT
    >    -   Atlas A3 训练系列产品/Atlas A3 推理系列产品
    >    -   Atlas A2 训练系列产品/Atlas A2 推理系列产品
    >-   _-_-npu-arch用于指定NPU的架构版本，dav-后为架构版本号，请替换为您实际使用的架构版本号。各AI处理器型号对应的架构版本号请通过[AI处理器型号和\_\_NPU\_ARCH\_\_的对应关系](../../../编程指南/语言扩展层/SIMD-BuiltIn关键字.md#table65291052154114)进行查询。

## 接下来的引导<a name="zh-cn_topic_0000001565030288_section128349412384"></a>

如果您对教程中的多核并行、流水编程等概念不了解，导致阅读过程有些吃力，可以参考[编程模型](../../../编程指南/编程模型/)学习基本概念，再来回顾本教程；如果您已经了解相关概念，并跑通了该样例，您可以参考[矢量编程](../../../算子实践参考/SIMD算子实现/矢量编程/)了解Ascend C矢量编程中的更多细节。

