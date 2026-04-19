# show\_kernel\_debug\_data工具<a name="ZH-CN_TOPIC_0000002046969553"></a>

在Ascend C算子程序代码中，用户可以使用AscendC::DumpTensor、AscendC::printf、AscendC::PrintTimeStamp、ascendc\_assert接口打印相关调试信息，并通过“acl API参考（C） \> 初始化和去初始化 \> aclInit”或直接配置acl.json文件，启用Dump配置，导出Ascend C算子Kernel的调测信息。本工具提供了对调测信息的离线解析能力，帮助用户获取并解析调试信息，即将导出的bin文件解析成可读格式。本工具的使用示例可参考[show\_kernel\_debug\_data样例](https://gitcode.com/cann/asc-tools/tree/master/examples/01_show_kernel_debug_data)。

>[!NOTE]说明 
>show\_kernel\_debug\_data支持多用户并发调用，但用户需要指定不同的落盘路径，否则可能出现落盘内容被覆盖等问题。

## 产品支持情况<a name="section15615918103719"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1613032013400"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p11131020174013"><a name="p11131020174013"></a><a name="p11131020174013"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p31311820124011"><a name="p31311820124011"></a><a name="p31311820124011"></a>√</p>
</td>
</tr>
<tr id="row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p48327011813"><a name="p48327011813"></a><a name="p48327011813"></a><span id="ph583230201815"><a name="ph583230201815"></a><a name="ph583230201815"></a><term id="zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p7948163910184"><a name="p7948163910184"></a><a name="p7948163910184"></a>√</p>
</td>
</tr>
<tr id="row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p14832120181815"><a name="p14832120181815"></a><a name="p14832120181815"></a><span id="ph1483216010188"><a name="ph1483216010188"></a><a name="ph1483216010188"></a><term id="zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1230510157397"><a name="p1230510157397"></a><a name="p1230510157397"></a>√</p>
</td>
</tr>
<tr id="row103361763242"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1983180181813"><a name="p1983180181813"></a><a name="p1983180181813"></a><span id="ph783112021813"><a name="ph783112021813"></a><a name="ph783112021813"></a><term id="zh-cn_topic_0000001312391781_term354143892110"><a name="zh-cn_topic_0000001312391781_term354143892110"></a><a name="zh-cn_topic_0000001312391781_term354143892110"></a>Atlas 200I/500 A2 推理产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p6304315123915"><a name="p6304315123915"></a><a name="p6304315123915"></a>√</p>
</td>
</tr>
<tr id="row18403312418"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p78319051815"><a name="p78319051815"></a><a name="p78319051815"></a><span id="ph1383116081815"><a name="ph1383116081815"></a><a name="ph1383116081815"></a><term id="zh-cn_topic_0000001312391781_term4363218112215"><a name="zh-cn_topic_0000001312391781_term4363218112215"></a><a name="zh-cn_topic_0000001312391781_term4363218112215"></a>Atlas 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1230412153394"><a name="p1230412153394"></a><a name="p1230412153394"></a>√</p>
</td>
</tr>
<tr id="row17253142120252"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p38313021813"><a name="p38313021813"></a><a name="p38313021813"></a><span id="ph58317041819"><a name="ph58317041819"></a><a name="ph58317041819"></a><term id="zh-cn_topic_0000001312391781_term71949488213"><a name="zh-cn_topic_0000001312391781_term71949488213"></a><a name="zh-cn_topic_0000001312391781_term71949488213"></a>Atlas 训练系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p2029083115583"><a name="p2029083115583"></a><a name="p2029083115583"></a>x</p>
</td>
</tr>
</tbody>
</table>

## 工具安装<a name="section857412719218"></a>

1.  安装工具。

    工具跟随CANN软件包发布（参考[环境准备](../../入门教程/环境准备.md)完成CANN安装），其路径默认为“$\{INSTALL\_DIR\}/tools/show\_kernel\_debug\_data”，其中$\{INSTALL\_DIR\}请替换为CANN软件安装后文件存储路径。以root用户安装为例，安装后文件默认存储路径为：/usr/local/Ascend/cann。

2.  设置环境变量。
    -   root用户安装Ascend-cann-toolkit包时

        ```
        source /usr/local/Ascend/cann/set_env.sh
        ```

    -   非root用户安装Ascend-cann-toolkit包时

        ```
        source ${HOME}/Ascend/cann/set_env.sh
        ```

3.  检查工具是否安装成功。

    执行如下命令，若能正常显示--help或-h信息，则表示工具环境正常，功能可正常使用。

    ```
    show_kernel_debug_data -h
    ```

## 使用方法<a name="section943033882116"></a>

-   **命令行方式**

    ```
    show_kernel_debug_data <bin_file_path> [<output_path>]
    ```

    <a name="table1331233514187"></a>
    <table><thead align="left"><tr id="row1931213353185"><th class="cellrowborder" valign="top" width="19.39%" id="mcps1.1.4.1.1"><p id="p031233531816"><a name="p031233531816"></a><a name="p031233531816"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="28.110000000000003%" id="mcps1.1.4.1.2"><p id="p1579144713356"><a name="p1579144713356"></a><a name="p1579144713356"></a>可选/必选</p>
    </th>
    <th class="cellrowborder" valign="top" width="52.5%" id="mcps1.1.4.1.3"><p id="p173127357185"><a name="p173127357185"></a><a name="p173127357185"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1431210350181"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.1.4.1.1 "><p id="p1272774591520"><a name="p1272774591520"></a><a name="p1272774591520"></a>&lt;bin_file_path&gt;</p>
    </td>
    <td class="cellrowborder" valign="top" width="28.110000000000003%" headers="mcps1.1.4.1.2 "><p id="p11791194718355"><a name="p11791194718355"></a><a name="p11791194718355"></a>必选</p>
    </td>
    <td class="cellrowborder" valign="top" width="52.5%" headers="mcps1.1.4.1.3 "><p id="p74714124281"><a name="p74714124281"></a><a name="p74714124281"></a>kernel侧调试信息落盘的bin文件或包含bin文件的目录路径，例如“/input/dump_workspace.bin”。</p>
    </td>
    </tr>
    <tr id="row17312193519184"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.1.4.1.1 "><p id="p193121335101811"><a name="p193121335101811"></a><a name="p193121335101811"></a>&lt;output_path&gt;</p>
    </td>
    <td class="cellrowborder" valign="top" width="28.110000000000003%" headers="mcps1.1.4.1.2 "><p id="p1279144713515"><a name="p1279144713515"></a><a name="p1279144713515"></a>可选</p>
    </td>
    <td class="cellrowborder" valign="top" width="52.5%" headers="mcps1.1.4.1.3 "><p id="p1573112228295"><a name="p1573112228295"></a><a name="p1573112228295"></a>解析结果的保存路径，例如“/output_dir”。默认是当前命令行执行目录下。</p>
    </td>
    </tr>
    </tbody>
    </table>

-   **API方式**

    获取kernel侧调试信息并解析成可读文件。函数原型如下。

    ```
    def show_kernel_debug_data(bin_file_path: str, output_path: str = './') -> None
    ```

    其中，输入参数说明如下。函数无输出参数和返回值。

    -   bin\_file\_path：kernel侧调试信息落盘的bin文件或包含bin文件的目录路径，字符串类型。
    -   output\_path：解析结果的保存路径，字符串类型，默认是当前接口调用脚本所在目录下。

    调用示例参考如下代码。

    ```
    from show_kernel_debug_data import show_kernel_debug_data
    show_kernel_debug_data(./input/dump_workspace.bin)
    ```

## 产物说明<a name="section7841124516211"></a>

工具解析结果文件目录结构如下。其中，dump\_data目录是对DumpTensor、PrintTimeStamp接口解析的结果，index0对应DumpTensor接口中第二个参数desc=0时的打印，loop0表示切分后首个分块的数据打印。

```
├ ${output_path}
├── PARSER_${timestamp}                     // ${timestamp}表示时间戳
     ├── dump_data
     │   ├── 0                             // core0解析结果
     │   ├── core_0_index_0_loop_0.bin     // core0 desc0 progress0落盘信息
     │   ├── core_0_index_0_loop_0.txt     // core0 desc0 progress0解析结果
      ...
     │   ├── core_0_index_2_loop_15.bin    // core0 desc2 progress15落盘信息
     │   ├── core_0_index_2_loop_15.txt    // core0 desc2 progress15解析结果
     │   └── time_stamp_core_0.csv         // 时间戳信息
     │   ├── 1                             // core1解析结果
     │   ├── 2                             // core2解析结果
      ...         
     │   └── index_dtype.json              // index与数据类型的映射关系
     └── parser.log                         // 解析日志，包括printf、ascendc_assert接口的打印信息
```

