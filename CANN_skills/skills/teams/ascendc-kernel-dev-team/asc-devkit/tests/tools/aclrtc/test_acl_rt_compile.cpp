/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "acl_rt_compile.h"
#include "stub/asrtc.h"

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <dlfcn.h>

class TEST_ACL_RT_COMPILE : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

using asrtcCreateProgramFuncPtr =
    asrtcResult (*)(asrtcProgram*, const char*, const char*, int, const char* const*, const char* const*);
using asrtcDestroyProgramFuncPtr = asrtcResult (*)(asrtcProgram*);
using asrtcCompileProgramFuncPtr = asrtcResult (*)(asrtcProgram, int, const char* const*);
using asrtcGetDeviceELFSizeFuncPtr = asrtcResult (*)(asrtcProgram, size_t*);
using asrtcGetDeviceELFFuncPtr = asrtcResult (*)(asrtcProgram, char*);
using asrtcAddNameExpressionFuncPtr = asrtcResult (*)(asrtcProgram, const char* const);
using asrtcGetLoweredNameFuncPtr = asrtcResult (*)(asrtcProgram, const char*, const char**);
using asrtcGetProgramLogSizeFuncPtr = asrtcResult (*)(asrtcProgram, size_t*);
using asrtcGetProgramLogFuncPtr = asrtcResult (*)(asrtcProgram, char*);

extern asrtcCreateProgramFuncPtr asrtcCreateProgramPtr;
extern asrtcDestroyProgramFuncPtr asrtcDestroyProgramPtr;
extern asrtcCompileProgramFuncPtr asrtcCompileProgramPtr;
extern asrtcGetDeviceELFSizeFuncPtr asrtcGetDeviceELFSizePtr;
extern asrtcGetDeviceELFFuncPtr asrtcGetDeviceELFPtr;
extern asrtcAddNameExpressionFuncPtr asrtcAddNameExpressionPtr;
extern asrtcGetLoweredNameFuncPtr asrtcGetLoweredNamePtr;
extern asrtcGetProgramLogSizeFuncPtr asrtcGetProgramLogSizePtr;
extern asrtcGetProgramLogFuncPtr asrtcGetProgramLogPtr;

extern aclError aclrtcAddNameExpr(aclrtcProg prog, const char *nameExpr);
extern aclError aclrtcGetLoweredName(aclrtcProg prog, const char *nameExpr, const char **manglingName);
extern inline bool EndsWith(std::string_view str, std::string_view suffix);
extern std::string ExtractCannPath(const std::string& pluginPath);
extern bool PathCheck(const char* path);
extern std::string GetCannPath();
extern void* handle;
extern aclError LoadExtraLib();
extern void UnloadExtraLib();

extern const int ACL_ERROR_RTC_INVALID_PROG = 176000;
extern const int ACL_ERROR_RTC_INVALID_INPUT = 176001;
extern const int ACL_ERROR_RTC_INVALID_OPTION = 176002;
extern const int ACL_ERROR_RTC_COMPILATION = 176003;
extern const int ACL_ERROR_RTC_LINKING = 176004;
extern const int ACL_ERROR_RTC_NO_NAME_EXPR_AFTER_COMPILATION = 176005;
extern const int ACL_ERROR_RTC_NO_LOWERED_NAMES_BEFORE_COMPILATION = 176006;
extern const int ACL_ERROR_RTC_NAME_EXPR_NOT_VALID = 176007;
extern const int ACL_ERROR_RTC_PROG_CREATION_FAILURE = 276000;
extern const int ACL_ERROR_RTC_OUT_OF_MEMORY = 276001;
extern const int ACL_ERROR_RTC_FAILURE = 576000;
extern const int ACL_SUCCESS = 0;

TEST_F(TEST_ACL_RT_COMPILE, aclrtc_aclrtcGetCompileLogSize)
{
    MOCKER(PathCheck).stubs().will(returnValue(true));
    asrtcGetProgramLogSizeFuncPtr originalPtr = asrtcGetProgramLogSizePtr;
    auto mockFunc = [](asrtcProgram, size_t*) -> asrtcResult {
        return ASRTC_ERROR_NOT_IMPLEMENTED;
    };
    asrtcGetProgramLogSizePtr = mockFunc;
    aclrtcProg prog = nullptr;
    size_t actualLogSize = 1024;
    aclError result = aclrtcGetCompileLogSize(prog, &actualLogSize);
    EXPECT_EQ(result, ACL_ERROR_RTC_INVALID_INPUT);
}

TEST_F(TEST_ACL_RT_COMPILE, aclrtc_aclrtcGetCompileLog)
{
    asrtcGetProgramLogFuncPtr originalPtr = asrtcGetProgramLogPtr;
    auto mockFunc = [](asrtcProgram, char*) -> asrtcResult {
        return ASRTC_ERROR_NOT_IMPLEMENTED;
    };
    asrtcGetProgramLogPtr = mockFunc;
    aclrtcProg prog = nullptr;
    char log[32] = "some log info ...";
    aclError result = aclrtcGetCompileLog(prog, log);
    EXPECT_EQ(result, ACL_ERROR_RTC_INVALID_INPUT);
}

TEST_F(TEST_ACL_RT_COMPILE, aclrtc_aclrtcAddNameExpr)
{
    asrtcAddNameExpressionFuncPtr originalPtr = asrtcAddNameExpressionPtr;
    auto mockFunc = [](asrtcProgram, const char* const) -> asrtcResult {
        return ASRTC_ERROR_NOT_IMPLEMENTED;
    };
    asrtcAddNameExpressionPtr = mockFunc;

    aclrtcProg prog = nullptr;
    const char *nameExpr = "hello_world";
    aclError result = aclrtcAddNameExpr(prog, nameExpr);
    EXPECT_EQ(result, ACL_ERROR_RTC_FAILURE);
}

TEST_F(TEST_ACL_RT_COMPILE, aclrtc_aclrtcGetManglingName)
{
    asrtcGetLoweredNameFuncPtr originalPtr = asrtcGetLoweredNamePtr;
    auto mockFunc = [](asrtcProgram, const char*, const char**) -> asrtcResult {
        return ASRTC_ERROR_NOT_IMPLEMENTED;
    };
    asrtcGetLoweredNamePtr = mockFunc;

    aclrtcProg prog = nullptr;
    const char *nameExpr = "hello_world";
    const char *manglingName = "hello_world";
    aclError result = aclrtcGetLoweredName(prog, nameExpr, &manglingName);
    EXPECT_EQ(result, ACL_ERROR_RTC_FAILURE);
}

TEST_F(TEST_ACL_RT_COMPILE, aclrtc_aclrtcCompileProg)
{
    asrtcCompileProgramFuncPtr originalPtr = asrtcCompileProgramPtr;
    auto mockFunc = [](asrtcProgram, int, const char* const*) -> asrtcResult {
        return ASRTC_ERROR_COMPILE;
    };
    asrtcCompileProgramPtr = mockFunc;

    aclrtcProg prog = nullptr;
    const char *options[] = {"--cce-aicore-arch=dav-c220-cube", "-O2"};
    int numOptions = sizeof(options) / sizeof(options[0]);
    MOCKER(PathCheck).stubs().will(returnValue(true));
    aclError result = aclrtcCompileProg(prog, numOptions, options);
    EXPECT_EQ(result, ACL_ERROR_RTC_INVALID_INPUT);
}

TEST_F(TEST_ACL_RT_COMPILE, aclrtc_aclrtcCompileProg_unknown)
{
    asrtcCompileProgramFuncPtr originalPtr = asrtcCompileProgramPtr;
    auto mockFunc = [](asrtcProgram, int, const char* const*) -> asrtcResult {
        return ASRTC_ERROR_UNKNOWN;
    };
    asrtcCompileProgramPtr = mockFunc;

    aclrtcProg prog = nullptr;
    const char *options[] = {"--cce-aicore-arch=dav-c220-cube", "-O2"};
    int numOptions = sizeof(options) / sizeof(options[0]);
    MOCKER(PathCheck).stubs().will(returnValue(true));
    aclError result = aclrtcCompileProg(prog, numOptions, options);
    EXPECT_EQ(result, ACL_ERROR_RTC_INVALID_INPUT);
}

TEST_F(TEST_ACL_RT_COMPILE, aclrtc_aclrtcCreateProg)
{
    asrtcCreateProgramFuncPtr originalPtr = asrtcCreateProgramPtr;
    auto mockFunc =
        [](asrtcProgram*, const char *, const char *, int, const char *const *, const char *const *) -> asrtcResult {
        return ASRTC_SUCCESS;
    };
    asrtcCreateProgramPtr = mockFunc;

    aclrtcProg prog = nullptr;
    const char *src = R""""(
#include "const.h"
#include "kernel_operator.h"
#include "acl/acl.h"

__global__ __aicore__ void add_custom(GM_ADDR x) {*x = 3 + MY_CONST;}
// extern "C" __global__ __aicore__ void add_custom(GM_ADDR x) {*x = 3 + MY_CONST;}
)"""";
    MOCKER(LoadExtraLib).stubs().will(returnValue(ACL_SUCCESS));
    aclError result = aclrtcCreateProg(&prog, src, "test_kernel", 0, nullptr, nullptr);
    EXPECT_EQ(result, ACL_SUCCESS);
}

TEST_F(TEST_ACL_RT_COMPILE, aclrtc_aclrtcCreateProg_fail)
{
    aclrtcProg prog = nullptr;
    const char *src = R""""(
#include "const.h"
#include "kernel_operator.h"
#include "acl/acl.h"

__global__ __aicore__ void add_custom(GM_ADDR x) {*x = 3 + MY_CONST;}
// extern "C" __global__ __aicore__ void add_custom(GM_ADDR x) {*x = 3 + MY_CONST;}
)"""";
    MOCKER(LoadExtraLib).stubs().will(returnValue(ACL_ERROR_RTC_FAILURE));
    aclError result = aclrtcCreateProg(&prog, src, "test_kernel", 0, nullptr, nullptr);
    EXPECT_EQ(result, ACL_ERROR_RTC_FAILURE);
}

TEST_F(TEST_ACL_RT_COMPILE, aclrtc_aclrtcDestroyProg)
{
    asrtcDestroyProgramFuncPtr originalPtr = asrtcDestroyProgramPtr;
    auto mockFunc = [](asrtcProgram*) -> asrtcResult {
        return ASRTC_SUCCESS;
    };
    const char *src = R""""(
#include "const.h"
#include "kernel_operator.h"
#include "acl/acl.h"

__global__ __aicore__ void add_custom(GM_ADDR x) {*x = 3 + MY_CONST;}
// extern "C" __global__ __aicore__ void add_custom(GM_ADDR x) {*x = 3 + MY_CONST;}
)"""";
    asrtcDestroyProgramPtr = mockFunc;
    MOCKER(LoadExtraLib).stubs().will(returnValue(ACL_SUCCESS));
    aclrtcProg prog = nullptr;
    aclrtcCreateProg(&prog, src, "test_kernel", 0, nullptr, nullptr);
    aclError result = aclrtcDestroyProg(&prog);
    EXPECT_EQ(result, ACL_SUCCESS);
}

TEST_F(TEST_ACL_RT_COMPILE, aclrtc_aclrtcGetBinData)
{
    asrtcGetDeviceELFFuncPtr originalPtr = asrtcGetDeviceELFPtr;
    auto mockFunc = [](asrtcProgram, char*) -> asrtcResult {
        return ASRTC_ERROR_COMPILE;
    };
    asrtcGetDeviceELFPtr = mockFunc;

    aclrtcProg prog = nullptr;
    char binData[32] = "some bin data ...";
    aclError result = aclrtcGetBinData(prog, binData);
    EXPECT_EQ(result, ACL_ERROR_RTC_INVALID_INPUT);
}

TEST_F(TEST_ACL_RT_COMPILE, aclrtc_aclrtcGetBinDataSize)
{
    asrtcGetDeviceELFSizeFuncPtr originalPtr = asrtcGetDeviceELFSizePtr;
    auto mockFunc = [](asrtcProgram, size_t*) -> asrtcResult {
        return ASRTC_ERROR_COMPILE;
    };
    asrtcGetDeviceELFSizePtr = mockFunc;

    aclrtcProg prog = nullptr;
    size_t binDataSizeRet = 1024;
    aclError result = aclrtcGetBinDataSize(prog, &binDataSizeRet);
    EXPECT_EQ(result, ACL_ERROR_RTC_INVALID_INPUT);
}

TEST_F(TEST_ACL_RT_COMPILE, aclrtc_EndsWith)
{
    std::string srcStr = "empty";
    std::string suffix = "not empty";
    EXPECT_EQ(EndsWith(srcStr, suffix), false);
}

TEST_F(TEST_ACL_RT_COMPILE, aclrtc_ExtractCannPath)
{
    std::string pluginPath = "/x86_64-linux/lib64/libacl_rtc.so";
    MOCKER(EndsWith).stubs().will(returnValue(true));
    std::string res = ExtractCannPath(pluginPath);
    EXPECT_EQ(res, std::string(""));
}

TEST_F(TEST_ACL_RT_COMPILE, aclrtc_PathCheck)
{
    const char *path = "../../../../tests/tools/aclrtc/stub/asrtc.h";
    bool res = PathCheck(path);
    EXPECT_EQ(res, true);
}

TEST_F(TEST_ACL_RT_COMPILE, aclrtc_UnloadExtraLib)
{
    handle = reinterpret_cast<void*>(0x1234);
    MOCKER(dlclose).expects(exactly(1)).will(returnValue(0));
    UnloadExtraLib();
    ASSERT_EQ(handle, nullptr);
}

TEST_F(TEST_ACL_RT_COMPILE, aclrtc_LoadExtraLib)
{
    MOCKER(GetCannPath).stubs().will(returnValue(std::string("")));
    MOCKER(PathCheck).stubs().will(returnValue(true));
    MOCKER(dlopen).stubs().will(returnValue((void*)("fake_handle")));
    MOCKER(dlsym).stubs().will(returnValue((void*)("fake_handle")));
    aclError ret = LoadExtraLib();
    ASSERT_EQ(ret, ACL_SUCCESS);
}

TEST_F(TEST_ACL_RT_COMPILE, aclrtc_LoadExtraLib_fail)
{
    MOCKER(GetCannPath).stubs().will(returnValue(std::string("")));
    MOCKER(PathCheck).stubs().will(returnValue(true));
    MOCKER(dlopen).stubs().will(returnValue(static_cast<void*>(nullptr)));
    LoadExtraLib();
    ASSERT_EQ(handle, nullptr);
}