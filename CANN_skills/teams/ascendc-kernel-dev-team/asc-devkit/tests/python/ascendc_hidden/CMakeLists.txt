cmake_minimum_required(VERSION 3.16)

project(code_hidden CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB_RECURSE CXX_SOURCE_FILES "src/*.cpp" "src/*.h")

add_executable(code_hidden ${CXX_SOURCE_FILES})

target_link_libraries(code_hidden PRIVATE stdc++fs)