# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

include(CMakeCommonLanguageInclude)

# extension for the output of a compile for a single file
if(UNIX)
    set(CMAKE_ASC_OUTPUT_EXTENSION .o)
else()
    set(CMAKE_ASC_OUTPUT_EXTENSION .obj)
endif()

set(CMAKE_INCLUDE_FLAG_ASC "-I")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(debug_compile_options "-O0 -g")
endif()

set(CMAKE_DEPFILE_FLAGS_ASC "-MD -MT <DEP_TARGET> -MF <DEP_FILE>")
if((NOT DEFINED CMAKE_DEPENDS_USE_COMPILER OR CMAKE_DEPENDS_USE_COMPILER) AND CMAKE_GENERATOR MATCHES "Makefiles|WMake")
    # dependencies are computed by the compiler itself
    set(CMAKE_ASC_DEPFILE_FORMAT gcc)
    set(CMAKE_ASC_DEPENDS_USE_COMPILER TRUE)
endif()

# -shared to create .so for shared library
if(NOT DEFINED CMAKE_SHARED_LIBRARY_CREATE_ASC_FLAGS)
    set(CMAKE_SHARED_LIBRARY_CREATE_ASC_FLAGS ${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS})
endif()
# used for -Wl,-soname when creating shared library
if(NOT DEFINED CMAKE_SHARED_LIBRARY_SONAME_ASC_FLAG)
  set(CMAKE_SHARED_LIBRARY_SONAME_ASC_FLAG ${CMAKE_SHARED_LIBRARY_SONAME_C_FLAG})
endif()
# used for -Wl,-rpath when link executable has shared library
if(NOT DEFINED CMAKE_EXECUTABLE_RUNTIME_ASC_FLAG)
    set(CMAKE_EXECUTABLE_RUNTIME_ASC_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG})
endif()

# rule variable to compile a single .o file
# CMAKE_ASC_COMPILER: bisheng
if(NOT CMAKE_ASC_COMPILE_OBJECT)
    set(CMAKE_ASC_COMPILE_OBJECT "<CMAKE_ASC_COMPILER> <DEFINES> <INCLUDES> -fPIC ${debug_compile_options} \
<FLAGS> -o <OBJECT> -c -x asc <SOURCE>")
endif()

# Create a static archive incrementally for large object file counts.
if(NOT DEFINED CMAKE_ASC_ARCHIVE_CREATE)
    set(CMAKE_ASC_ARCHIVE_CREATE "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()
# add without checking duplication
if(NOT DEFINED CMAKE_ASC_ARCHIVE_APPEND)
    set(CMAKE_ASC_ARCHIVE_APPEND "<CMAKE_AR> q <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()
if(NOT DEFINED CMAKE_ASC_ARCHIVE_FINISH)
    set(CMAKE_ASC_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")
endif()


# rule variable to create a shared module
if(NOT CMAKE_ASC_CREATE_SHARED_MODULE)
    set(CMAKE_ASC_CREATE_SHARED_MODULE ${CMAKE_ASC_CREATE_SHARED_LIBRARY})
endif()

# when language is set to ASC, execute when add_executable.
# FLAGS: -D
# ASC_LINK_FLAGS: link options
if(NOT CMAKE_ASC_LINK_EXECUTABLE)
    set(CMAKE_ASC_LINK_EXECUTABLE "<CMAKE_ASC_COMPILER> <FLAGS> <CMAKE_ASC_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o \
<TARGET> <LINK_LIBRARIES>")
endif()

# rule variable to create a shared library
if(NOT CMAKE_ASC_CREATE_SHARED_LIBRARY)
    set(CMAKE_ASC_CREATE_SHARED_LIBRARY "<CMAKE_ASC_COMPILER> <CMAKE_SHARED_LIBRARY_ASC_FLAGS> \
<LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_ASC_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> \
<OBJECTS> <LINK_LIBRARIES>")
endif()

set(CMAKE_ASC_INFORMATION_LOADED 1)   # 标记Cmake已经加载初始化ASC编程语言

if(CMAKE_ASC_RUN_MODE STREQUAL "sim")
    set(_ARCH_TO_DIR_MAP
        "dav-2002" "dav_2002"
        "dav-2201" "dav_2201"
        "dav-3510" "dav_3510"
    )
    list(FIND _ARCH_TO_DIR_MAP "${CMAKE_ASC_ARCHITECTURES}" _index)
    if(_index GREATER -1)
        math(EXPR _val_index "${_index} + 1")
        list(GET _ARCH_TO_DIR_MAP ${_val_index} _ASC_INTERNAL_DIR)
        set(_ASC_SIM_PATH "$ENV{ASCEND_HOME_PATH}/tools/simulator/${_ASC_INTERNAL_DIR}/lib")
        string(APPEND CMAKE_ASC_LINK_FLAGS " -Wl,-rpath,${_ASC_SIM_PATH} -Wl,-L${_ASC_SIM_PATH} -Wl,--disable-new-dtags")
        link_libraries(runtime_camodel npu_drv_camodel)
        message(STATUS "ASC Simulator enabled: ${_ASC_SIM_PATH}")
    else()
        message(FATAL_ERROR "Unsupported ASC architecture for simulator: ${CMAKE_ASC_ARCHITECTURES}")
    endif()
endif()

if(CMAKE_ASC_RUN_MODE STREQUAL "cpu")
    # CPU模式架构映射表
    set(_ARCH_TO_SOC_DIR_MAP
        "dav-2002" "2002" "Ascend310P1"
        "dav-2201" "2201" "Ascend910B1"
        "dav-3510" "3510" "Ascend950PR_9599"
    )
    list(FIND _ARCH_TO_SOC_DIR_MAP "${CMAKE_ASC_ARCHITECTURES}" _index)
    math(EXPR _remainder "${_index} % 3")
    if(_remainder EQUAL 0) # 只允许dav-2002、dav-2201、dav-3510
        math(EXPR _short_index "${_index} + 1")
        math(EXPR _soc_dir_index "${_index} + 2")
        list(GET _ARCH_TO_SOC_DIR_MAP ${_short_index} _SHORT_NPU_ARCH)
        list(GET _ARCH_TO_SOC_DIR_MAP ${_soc_dir_index} _SOC_DIR)
        
        # system include
        set(ASC_CPU_SYSTEM_INCLUDE_OPTIONS "-isystem$ENV{ASCEND_HOME_PATH}/tools/tikicpulib/lib/include -isystem$ENV{ASCEND_HOME_PATH}/include \
-isystem$ENV{ASCEND_HOME_PATH}/asc/impl/adv_api \
-isystem$ENV{ASCEND_HOME_PATH}/asc/impl/basic_api -isystem$ENV{ASCEND_HOME_PATH}/asc/impl/c_api \
-isystem$ENV{ASCEND_HOME_PATH}/asc/impl/micro_api -isystem$ENV{ASCEND_HOME_PATH}/asc/impl/simt_api \
-isystem$ENV{ASCEND_HOME_PATH}/asc/impl/utils -isystem$ENV{ASCEND_HOME_PATH}/asc/include \
-isystem$ENV{ASCEND_HOME_PATH}/asc/include/adv_api -isystem$ENV{ASCEND_HOME_PATH}/asc/include/adv_api/matmul \
-isystem$ENV{ASCEND_HOME_PATH}/asc/include/aicpu_api -isystem$ENV{ASCEND_HOME_PATH}/asc/include/basic_api \
-isystem$ENV{ASCEND_HOME_PATH}/asc/include/c_api -isystem$ENV{ASCEND_HOME_PATH}/asc/include/interface \
-isystem$ENV{ASCEND_HOME_PATH}/asc/include/micro_api -isystem$ENV{ASCEND_HOME_PATH}/asc/include/simt_api \
-isystem$ENV{ASCEND_HOME_PATH}/asc/include/tiling -isystem$ENV{ASCEND_HOME_PATH}/asc/include/utils")
        
        # 添加编译选项
        string(APPEND CMAKE_ASC_FLAGS "-g -D_GLIBCXX_USE_CXX11_ABI=0 -DASCENDC_DEBUG=1 -D__NPU_ARCH__=${_SHORT_NPU_ARCH} --run-mode=cpu ${ASC_CPU_SYSTEM_INCLUDE_OPTIONS}")
        
        # 配置链接选项
        string(APPEND CMAKE_ASC_LINK_FLAGS "--run-mode=cpu -Wl,--disable-new-dtags")
                
        # 配置链接库
        link_libraries(
            -Wl,-rpath,$ENV{ASCEND_HOME_PATH}/lib64
            -Wl,-L$ENV{ASCEND_HOME_PATH}/lib64
            -Wl,-rpath,$ENV{ASCEND_HOME_PATH}/tools/tikicpulib/lib
            -Wl,-L$ENV{ASCEND_HOME_PATH}/tools/tikicpulib/lib
            -Wl,-rpath,$ENV{ASCEND_HOME_PATH}/tools/tikicpulib/lib/${_SOC_DIR}
            -Wl,-L$ENV{ASCEND_HOME_PATH}/tools/tikicpulib/lib/${_SOC_DIR}
            -Wl,-rpath,$ENV{ASCEND_HOME_PATH}/tools/simulator/${_SOC_DIR}/lib
            -Wl,-L$ENV{ASCEND_HOME_PATH}/tools/simulator/${_SOC_DIR}/lib
            -Wl,-Bdynamic,--no-as-needed
            ascendc_acl_stub
            $<$<STREQUAL:${CMAKE_ASC_ARCHITECTURES},dav-2002>:_pvmodel>
            $<$<OR:$<STREQUAL:${CMAKE_ASC_ARCHITECTURES},dav-2201>,$<STREQUAL:${CMAKE_ASC_ARCHITECTURES},dav-3510>>:pem_davinci>
            $<$<STREQUAL:${CMAKE_ASC_ARCHITECTURES},dav-3510>:-pthread>
            cpudebug_cceprint
            cpudebug_npuchk
            cpudebug_stubreg
            cpudebug
            c_sec
            stdc++
            runtime
            register
            error_manager
            profapi
            ge_common_base
            unified_dlog
            mmpa
            dl
            ascend_dump
        )
        
    else()
        message(FATAL_ERROR "Unsupported ASC architecture for CPU mode: ${CMAKE_ASC_ARCHITECTURES}, should be dav-2002, dav-2201, dav-3510")
    endif()
endif()