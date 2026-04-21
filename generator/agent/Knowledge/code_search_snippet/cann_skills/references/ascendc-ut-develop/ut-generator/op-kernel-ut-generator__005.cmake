if (UT_TEST_ALL OR OP_KERNEL_UT)
    set(<op>_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/<op>_tiling.cpp
    )
    AddOpTestCase(<op> "ascend910b" "-DDTYPE_X=float" "${<op>_tiling_files}")
endif()
