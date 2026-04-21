cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(softmax0309_custom LANGUAGES ASC CXX)

add_executable(softmax0309_custom
    softmax0309.asc
)

target_include_directories(softmax0309_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(softmax0309_custom PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_compile_options(softmax0309_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
