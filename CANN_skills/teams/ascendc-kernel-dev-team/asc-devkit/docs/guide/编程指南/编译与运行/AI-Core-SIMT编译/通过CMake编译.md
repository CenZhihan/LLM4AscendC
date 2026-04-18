# 通过CMake编译<a name="ZH-CN_TOPIC_0000002531056204"></a>

项目中可以使用CMake来更简便地使用毕昇编译器编译Ascend C SIMT算子，生成可执行文件。

以下是CMake脚本的示例及其核心步骤说明：

```
cmake_minimum_required(VERSION 3.16)

# 1、find_package(ASC)是CMake中用于查找和配置Ascend C编译工具链的命令
find_package(ASC)  

# 2、指定项目支持的语言包括ASC和CXX，ASC表示支持使用毕昇编译器对Ascend C编程语言进行编译
project(kernel_samples LANGUAGES ASC CXX)

# 3、使用CMake接口编译可执行文件
add_executable(demo
    add_custom.asc
)
#.....
target_compile_options(demo PRIVATE
    # --npu-arch用于指定NPU的架构版本，dav-后为架构版本号
    # <COMPILE_LANGUAGE:ASC>:表明该编译选项仅对语言ASC生效
    $<$<COMPILE_LANGUAGE:ASC>: --npu-arch=dav-3510>
    # 开启SIMT编程模型的编译功能 
    --enable-simt   
)
```

下文列出了使用CMake编译时默认链接库。

**表 1**  默认链接库

<a name="table201231542115513"></a>
<table><thead align="left"><tr id="row171231542205510"><th class="cellrowborder" valign="top" width="23.98%" id="mcps1.2.3.1.1"><p id="p11123114295513"><a name="p11123114295513"></a><a name="p11123114295513"></a>名称</p>
</th>
<th class="cellrowborder" valign="top" width="76.02%" id="mcps1.2.3.1.2"><p id="p1412374225512"><a name="p1412374225512"></a><a name="p1412374225512"></a>作用描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row5123842135514"><td class="cellrowborder" valign="top" width="23.98%" headers="mcps1.2.3.1.1 "><p id="p1212364212559"><a name="p1212364212559"></a><a name="p1212364212559"></a>libascendc_runtime.a</p>
</td>
<td class="cellrowborder" valign="top" width="76.02%" headers="mcps1.2.3.1.2 "><p id="p1112394218551"><a name="p1112394218551"></a><a name="p1112394218551"></a>Ascend C算子参数等组装库。</p>
</td>
</tr>
<tr id="row612324285519"><td class="cellrowborder" valign="top" width="23.98%" headers="mcps1.2.3.1.1 "><p id="p01231423552"><a name="p01231423552"></a><a name="p01231423552"></a>libruntime.so</p>
</td>
<td class="cellrowborder" valign="top" width="76.02%" headers="mcps1.2.3.1.2 "><p id="p8123164255511"><a name="p8123164255511"></a><a name="p8123164255511"></a>Runtime运行库。</p>
</td>
</tr>
<tr id="row1612374285512"><td class="cellrowborder" valign="top" width="23.98%" headers="mcps1.2.3.1.1 "><p id="p2012315425551"><a name="p2012315425551"></a><a name="p2012315425551"></a>libprofapi.so</p>
</td>
<td class="cellrowborder" valign="top" width="76.02%" headers="mcps1.2.3.1.2 "><p id="p12123164265514"><a name="p12123164265514"></a><a name="p12123164265514"></a>Ascend C算子运行性能数据采集库。</p>
</td>
</tr>
<tr id="row10123134212552"><td class="cellrowborder" valign="top" width="23.98%" headers="mcps1.2.3.1.1 "><p id="p812374235515"><a name="p812374235515"></a><a name="p812374235515"></a>libunified_dlog.so</p>
</td>
<td class="cellrowborder" valign="top" width="76.02%" headers="mcps1.2.3.1.2 "><p id="p412314426554"><a name="p412314426554"></a><a name="p412314426554"></a>CANN日志收集库。</p>
</td>
</tr>
<tr id="row1012384210552"><td class="cellrowborder" valign="top" width="23.98%" headers="mcps1.2.3.1.1 "><p id="p15123104219559"><a name="p15123104219559"></a><a name="p15123104219559"></a>libmmpa.so</p>
</td>
<td class="cellrowborder" valign="top" width="76.02%" headers="mcps1.2.3.1.2 "><p id="p13123242135519"><a name="p13123242135519"></a><a name="p13123242135519"></a>CANN系统接口库。</p>
</td>
</tr>
<tr id="row17124154245516"><td class="cellrowborder" valign="top" width="23.98%" headers="mcps1.2.3.1.1 "><p id="p612484265518"><a name="p612484265518"></a><a name="p612484265518"></a>libascend_dump.so</p>
</td>
<td class="cellrowborder" valign="top" width="76.02%" headers="mcps1.2.3.1.2 "><p id="p101241842175512"><a name="p101241842175512"></a><a name="p101241842175512"></a>CANN维测信息库。</p>
</td>
</tr>
<tr id="row6124164213551"><td class="cellrowborder" valign="top" width="23.98%" headers="mcps1.2.3.1.1 "><p id="p111246426558"><a name="p111246426558"></a><a name="p111246426558"></a>libc_sec.so</p>
</td>
<td class="cellrowborder" valign="top" width="76.02%" headers="mcps1.2.3.1.2 "><p id="p31241442185512"><a name="p31241442185512"></a><a name="p31241442185512"></a>CANN安全函数库。</p>
</td>
</tr>
<tr id="row171241342175514"><td class="cellrowborder" valign="top" width="23.98%" headers="mcps1.2.3.1.1 "><p id="p6124124218556"><a name="p6124124218556"></a><a name="p6124124218556"></a>liberror_manager.so</p>
</td>
<td class="cellrowborder" valign="top" width="76.02%" headers="mcps1.2.3.1.2 "><p id="p61248424557"><a name="p61248424557"></a><a name="p61248424557"></a>CANN错误信息管理库。</p>
</td>
</tr>
<tr id="row512404213550"><td class="cellrowborder" valign="top" width="23.98%" headers="mcps1.2.3.1.1 "><p id="p151243425553"><a name="p151243425553"></a><a name="p151243425553"></a>libascendcl.so</p>
</td>
<td class="cellrowborder" valign="top" width="76.02%" headers="mcps1.2.3.1.2 "><p id="p1012424213555"><a name="p1012424213555"></a><a name="p1012424213555"></a>acl相关接口库。</p>
</td>
</tr>
</tbody>
</table>

