# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set(LIB_SUPPORT_TYPES SHARED STATIC)

function(library_interface_setup target_name)
    target_include_directories(${target_name} INTERFACE
        ${ASCEND_CANN_PACKAGE_LINUX_PATH}/include
        ${ASCEND_CANN_PACKAGE_LINUX_PATH}/tikcpp/tikcfw/
    )
    target_link_directories(${target_name} INTERFACE
        ${ASCEND_CANN_PACKAGE_PATH}/lib64
    )
    # consistent with default lib in kernellaunch
    target_link_libraries(${target_name} INTERFACE
        ascendc_runtime
        $<$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>:acl_rt>
        $<$<NOT:$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>>:ascendcl>
        runtime
        register
        error_manager
        profapi
        ge_common_base
        unified_dlog
        mmpa
        dl
        ascend_dump
        c_sec
    )

    # for ACLRT_LAUNCH_KERNEL
    if (ENABLE_ACLRT_LAUNCH)
        set(ACLRT_HEADER_INCLUDE_PATH ${CMAKE_BINARY_DIR}/include/${target_name} CACHE PATH "Path for aclrt header files")
        file(MAKE_DIRECTORY ${ACLRT_HEADER_INCLUDE_PATH})

        target_include_directories(${target_name} INTERFACE
            ${ACLRT_HEADER_INCLUDE_PATH}
        )

        target_compile_definitions(${target_name} PRIVATE
            GEN_ACLRT=${ACLRT_HEADER_INCLUDE_PATH}
        )
    endif()
endfunction()

function(ascendc_library target_name target_type)
    if(NOT target_type IN_LIST LIB_SUPPORT_TYPES)
        message(FATAL_ERROR "target_type ${target_type} is unsupported, the support list is ${LIB_SUPPORT_TYPES}")
    endif()

    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE ASC)
    add_library(${target_name} ${target_type} ${ARGN})

    library_interface_setup(${target_name})

    include(GNUInstallDirs)
    install(TARGETS ${target_name}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endfunction()

function(ascendc_fatbin_library target_name)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE ASC)
    file(MAKE_DIRECTORY ${CMAKE_INSTALL_PREFIX}/fatbin/${target_name}/)

    add_library(${target_name} OBJECT ${ARGN})

    target_compile_options(${target_name} PRIVATE
        --aicore-only           # compile device.o
    )
    library_interface_setup(${target_name})

    set(output_path ${CMAKE_INSTALL_PREFIX}/fatbin/${target_name}/${target_name}.o)
    add_custom_target(lld_copy_${target_name} ALL
        COMMAND ${CMAKE_ASC_LLD_LINKER} -m aicorelinux -Ttext=0 $<TARGET_OBJECTS:${target_name}> -static -o ${output_path}
        COMMAND cp ${output_path} ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS
        COMMENT "Generate final device.o from ascendc_fatbin_library"
    )

    add_dependencies(lld_copy_${target_name} ${target_name})
endfunction()

function(ascendc_compile_definitions target_name target_scope)
    target_compile_definitions(${target_name} ${target_scope} ${ARGN})
endfunction()

function(ascendc_compile_options target_name target_scope)
    if(ARGN)
        set(kernel_compile_options_list)
        set(host_compile_options_list)
        set(find_host_compile_options_flag OFF)
        # compile options are split by "-forward-options-to-host-compiler",
        # options after this all belong to host compile options
        foreach(arg ${ARGN})
            if (find_host_compile_options_flag)
                list(APPEND host_compile_options_list ${arg})
            else()
                if (${arg} STREQUAL "-forward-options-to-host-compiler")
                    set(find_host_compile_options_flag ON)
                else()
                    list(APPEND kernel_compile_options_list ${arg})
                endif()
            endif()
        endforeach()

        target_compile_options(${target_name} ${target_scope}
            -Xaicore-start ${kernel_compile_options_list} -Xaicore-end
            -Xhost-start ${host_compile_options_list} -Xhost-end
        )
    endif()
endfunction()

function(ascendc_include_directories target_name target_scope)
    target_include_directories(${target_name} ${target_scope} ${ARGN})
endfunction()