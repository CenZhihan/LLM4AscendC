# 通过CMake编译<a name="ZH-CN_TOPIC_0000002428982142"></a>

项目中可以使用CMake来更简便地使用毕昇编译器编译Ascend C算子，生成可执行文件、动态库、静态库或二进制文件。

以下是CMake脚本的示例及其核心步骤说明：

```
# 1、find_package(ASC)是CMake中用于查找和配置Ascend C编译工具链的命令
find_package(ASC)  

# 2、指定项目支持的语言包括ASC和CXX，ASC表示支持使用毕昇编译器对Ascend C编程语言进行编译
project(kernel_samples LANGUAGES ASC CXX)

# 3、使用CMake接口编译可执行文件、动态库、静态库、二进制文件
add_executable(demo
    add_custom.asc
)
#.....
target_compile_options(demo PRIVATE
    # --npu-arch用于指定NPU的架构版本，dav-后为架构版本号，各产品型号对应的架构版本号请通过[对应关系表](../../语言扩展层/SIMD-BuiltIn关键字.md#table65291052154114)进行查询。
    # <COMPILE_LANGUAGE:ASC>:表明该编译选项仅对语言ASC生效
    $<$<COMPILE_LANGUAGE:ASC>: --npu-arch=dav-2201>    
)
```

以下是动态库、静态库编译示例，同时展示如何将源文件切换为用语言ASC编译：

-   编译.cpp文件生成动态库

    ```
    # 将.cpp文件置为ASC属性，启用Ascend C语言进行编译
    set_source_files_properties(
        add_custom_base.cpp 
        sub_custom_base.cpp
        PROPERTIES LANGUAGE ASC
    )
    
    add_library(kernel_lib SHARED
        add_custom_base.cpp 
        sub_custom_base.cpp
    )
    
    target_compile_options(kernel_lib PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>: --npu-arch=dav-2201>
    )
    
    add_executable(demo
        main.cpp
    )
    target_link_libraries(demo PRIVATE
        kernel_lib
    )
    ```

-   编译.asc文件生成静态库

    ```
    # .asc文件会默认启用Ascend C语言进行编译，不需要通过set_source_files_properties进行设置
    add_library(kernel_lib STATIC
        add_custom_base.asc 
        sub_custom_base.asc
    )
    
    target_compile_options(kernel_lib PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>: --npu-arch=dav-2201>
    )
    
    add_executable(demo
        main.cpp
    )
    target_link_libraries(demo PRIVATE
        kernel_lib
    )
    ```

下文列出了使用CMake编译时常用的链接库、以及默认链接库。

**表 1**  常用的链接库（在使用高阶API时，必须链接以下库，因为这些库是高阶API功能所依赖的。在其他场景下，可以根据具体需求选择是否链接这些库。）

<a name="table1632724817328"></a>
<table><thead align="left"><tr id="row1132712485321"><th class="cellrowborder" valign="top" width="42.19%" id="mcps1.2.3.1.1"><p id="p377063619281"><a name="p377063619281"></a><a name="p377063619281"></a>使用场景</p>
</th>
<th class="cellrowborder" valign="top" width="57.809999999999995%" id="mcps1.2.3.1.2"><p id="p5423124282814"><a name="p5423124282814"></a><a name="p5423124282814"></a>名称</p>
</th>
</tr>
</thead>
<tbody><tr id="row131961149143112"><td class="cellrowborder" rowspan="3" valign="top" width="42.19%" headers="mcps1.2.3.1.1 "><p id="p477013682820"><a name="p477013682820"></a><a name="p477013682820"></a>使用高阶API相关的Tiling接口时需要同时链接。</p>
</td>
<td class="cellrowborder" valign="top" width="57.809999999999995%" headers="mcps1.2.3.1.2 "><p id="p94248427280"><a name="p94248427280"></a><a name="p94248427280"></a>libtiling_api.a</p>
</td>
</tr>
<tr id="row957113206355"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p1942416423283"><a name="p1942416423283"></a><a name="p1942416423283"></a>libregister.so</p>
</td>
</tr>
<tr id="row1652295021916"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p134241942162817"><a name="p134241942162817"></a><a name="p134241942162817"></a>libgraph_base.so</p>
</td>
</tr>
<tr id="row025115214213"><td class="cellrowborder" valign="top" width="42.19%" headers="mcps1.2.3.1.1 "><p id="p1777015364285"><a name="p1777015364285"></a><a name="p1777015364285"></a>使用PlatformAscendC相关硬件平台信息接口时需要链接。</p>
</td>
<td class="cellrowborder" valign="top" width="57.809999999999995%" headers="mcps1.2.3.1.2 "><p id="p1642415421284"><a name="p1642415421284"></a><a name="p1642415421284"></a>libplatform.so</p>
</td>
</tr>
</tbody>
</table>

**表 2**  默认链接库

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

