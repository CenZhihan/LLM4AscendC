cmake_minimum_required(VERSION 3.16)

# ✅ 使用 find_package 自动发现 CANN
find_package(ASC REQUIRED)

project(kernel_samples LANGUAGES ASC CXX)

add_executable(demo
    reducemax.asc
)

target_link_libraries(demo PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

# ✅ 指定 NPU 架构
target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>  # A2 服务器
)
