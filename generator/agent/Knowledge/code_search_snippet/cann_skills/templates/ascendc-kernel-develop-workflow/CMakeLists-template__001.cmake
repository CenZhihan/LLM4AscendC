cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project({operator_name}_custom LANGUAGES ASC CXX)

add_executable({operator_name}_custom
    {operator_name}.asc
)

target_include_directories({operator_name}_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries({operator_name}_custom PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_compile_options({operator_name}_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch={NPU架构}>
)
