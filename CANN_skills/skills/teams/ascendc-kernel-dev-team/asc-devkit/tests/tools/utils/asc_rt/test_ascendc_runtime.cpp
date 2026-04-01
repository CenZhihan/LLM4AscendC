/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cmath>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <string>
#include <unordered_map>
#include <fcntl.h>

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>

#include "ascendc_runtime.h"
#include "securec.h"
#include "runtime/kernel.h"
#include "acl_rt.h"
#include "ascendc_tool_log.h"

#include <iostream>
using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

class TEST_ASCENDC_RUNTIME : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ASCENDC_RUNTIME, ascendcRuntimeProfTest){
    const char *name = "xxx";
    uint32_t numBlocks = 1;
    uint32_t taskType = 0;
    const uint64_t startTime = 0;
    MOCKER(ReportAscendProf).expects(once());
    ReportAscendProf(name, numBlocks, taskType, startTime);
    EXPECT_NO_THROW(GlobalMockObject::verify());
}

TEST_F(TEST_ASCENDC_RUNTIME, ascendcRuntimeProfFixTest){
    const char *name = "xxx";
    uint32_t numBlocks = 1;
    uint32_t taskType = 10;
    const uint64_t startTime = 0;
    ReportAscendProf(name, numBlocks, taskType, startTime);
    EXPECT_NO_THROW(GlobalMockObject::verify());
}

#define ASCENDC_DUMP_SIZE 75 * 1024 * 1024
TEST_F(TEST_ASCENDC_RUNTIME, ascendcRuntimeRegisterTest){
    const char *name = "sss";
    void *handle;
    size_t fileSize = 3;
    uint32_t type = 0;
    uint32_t ret;
    ret = RegisterAscendBinary(name, fileSize, type, &handle);
    EXPECT_EQ(ret, 0);
    type = 1;
    ret = RegisterAscendBinary(name, fileSize, type, &handle);
    EXPECT_EQ(ret, 0);
    type = 2;
    ret = RegisterAscendBinary(name, fileSize, type, &handle);
    EXPECT_EQ(ret, 0);
    GetAscendCoreSyncAddr(&handle);
    AllocAscendMemDevice(&handle, fileSize);
    FreeAscendMemDevice(handle);
    UnregisterAscendBinary(handle);
}

TEST_F(TEST_ASCENDC_RUNTIME, ascendcRuntimeDevBinaryRegisterTest) {
    uint8_t fileBuf[32];
    size_t fileSize = 32;
    void *handle;
    int32_t ret;
    ret = AscendDevBinaryRegister(fileBuf, fileSize, &handle);
    EXPECT_EQ(ret, 0);
    const char* stubFunc = "hello_world";
    ret = AscendFunctionRegister(handle, stubFunc);
    EXPECT_EQ(ret, 0);
    MOCKER(rtKernelLaunchWithFlagV2).stubs().will(returnValue(0));
    uint32_t numBlocks;
    void **args = nullptr;
    uint32_t size;
    rtStream_t stream = nullptr;
    ret = AscendKernelLaunchWithFlagV2(stubFunc, numBlocks, args, size, stream, 0);
    EXPECT_EQ(ret, 0);
    ret = AscendKernelLaunchWithFlagV2(stubFunc, numBlocks, args, size, stream, 2048);
    EXPECT_EQ(ret, 0);
}

TEST_F(TEST_ASCENDC_RUNTIME, ascendcRuntimeMemoryFailedTest){
    size_t bufsize = 16;
    uint32_t ret;
    MOCKER(aclrtMalloc).expects(once()).will(returnValue(1));
    ret = AllocAscendMemDevice(nullptr, bufsize);
    EXPECT_NE(ret, 0);
    ret = FreeAscendMemDevice(nullptr);
    EXPECT_NE(ret, 0);
}

TEST_F(TEST_ASCENDC_RUNTIME, ascendcRuntimeGetProfStatusTest){
    MOCKER(GetAscendProfStatus).expects(once()).will(returnValue(true));
    MOCKER(AscendProfRegister).expects(once());
    GetAscendProfStatus();
    AscendProfRegister();
    EXPECT_NO_THROW(GlobalMockObject::verify());
}

TEST_F(TEST_ASCENDC_RUNTIME, GetCoreNumForMixVectorCore){
    uint32_t aiCoreNum = 0;
    uint32_t vectorCoreNum = 0;
    GetCoreNumForMixVectorCore(&aiCoreNum, &vectorCoreNum);
    EXPECT_EQ(aiCoreNum, 0);
    EXPECT_EQ(vectorCoreNum, 0);
}

typedef void *rtEvent_t;
typedef void *rtStream_t;
typedef void *rtContext_t;

typedef struct {
    rtStream_t stream;
    rtEvent_t eventA;
    rtEvent_t eventB;
} AscendCStreamForVectorCore;

extern "C" uint32_t LaunchAscendKernelForVectorCore(const char *opType, void *handle, const uint64_t key, void **args, uint32_t size,
    const void *stream, bool enableProf, uint32_t aicNumBlocks, uint32_t aivNumBlocks, uint32_t aivNumBlocksOffset);
extern "C" void AscendCDestroyStreamCallBack(rtStream_t stream, const bool isCreate);
extern std::unordered_map<const void *, AscendCStreamForVectorCore> g_ascStreamMap;
extern "C" uint32_t LaunchAscendKernel(void *handle, const uint64_t key, const uint32_t numBlocks, void **args, uint32_t size,
    const rtStream_t stream);


TEST_F(TEST_ASCENDC_RUNTIME, LaunchVectorCore){
    const char* opType = "AddCustom";
    void * handle = nullptr;
    void **args = nullptr;
    void *stream = nullptr;
    AscendCStreamForVectorCore streamForVectorCore;
    AscendCDestroyStreamCallBack(stream, true);
    g_ascStreamMap[stream] = streamForVectorCore;
    AscendCDestroyStreamCallBack(stream, false);
    uint32_t ret = LaunchAscendKernelForVectorCore(opType, handle, 0, args, 0, stream, false, 1, 1, 0);
    EXPECT_EQ(ret, 0);
    ret = LaunchAscendKernelForVectorCore(opType, handle, 0, args, 0, stream, true, 1, 1, 0);
    EXPECT_EQ(ret, 0);
}

const char *fake_rtGetSocVersion()
{
    return "ascend910b2";
}

const char *fake_rtGetSocVersion1()
{
    return "ascend910b22";
}

TEST_F(TEST_ASCENDC_RUNTIME, TestAscendCheckSoCVersion)
{
    char errMsg[1024]; // max err msg length is 1024
    const char* socVersion = "ascend910b1";
    MOCKER(aclrtGetSocName).stubs().will(invoke(fake_rtGetSocVersion));
    bool ret = AscendCheckSoCVersion(socVersion, errMsg);
    EXPECT_EQ(ret, true);
    const char* socVersion1 = "ascend310p1";
    ret = AscendCheckSoCVersion(socVersion1, errMsg);
    EXPECT_EQ(ret, false);
    GlobalMockObject::verify();
    MOCKER(aclrtGetSocName).stubs().will(invoke(fake_rtGetSocVersion1));
    ret = AscendCheckSoCVersion(socVersion1, errMsg);
    EXPECT_EQ(ret, false);
    GlobalMockObject::verify();
    const char* socVersion2 = "ascend310p1xx";
    ret = AscendCheckSoCVersion(socVersion2, errMsg);
    EXPECT_EQ(ret, false);
}

TEST_F(TEST_ASCENDC_RUNTIME, LaunchAscendKernelA){
    void * handle = nullptr;
    void *stream = nullptr;
    void **args = nullptr;
    const uint32_t numBlocks = 0;
    uint32_t size = 0;
    const uint64_t key = 0;
    aclrtLaunchKernelCfg *cfg = nullptr;
    aclrtLaunchKernelWithConfig(handle, numBlocks, stream, cfg, nullptr, nullptr);
    uint32_t ret = LaunchAscendKernel(handle, key, numBlocks, args, size, stream);
    EXPECT_EQ(ret, 0);
}

TEST_F(TEST_ASCENDC_RUNTIME, AscendDevBinaryLazyRegister) {
    const char* binBuf = "test_binary";
    size_t binSize = strlen(binBuf);
    void** handle = nullptr;
    int32_t ret = AscendDevBinaryLazyRegister(binBuf, binSize, handle);
    EXPECT_EQ(ret, 0);
}

TEST_F(TEST_ASCENDC_RUNTIME, AscendGetFuncFromBinary) {
    void* const binHandle = nullptr;
    const char* kernelName = "test_kernel";
    void** funcHandle = nullptr;
    int32_t ret = AscendGetFuncFromBinary(binHandle, kernelName, funcHandle);
    EXPECT_EQ(ret, 0);
}

typedef enum KernelType : unsigned int {
    K_TYPE_AICORE = 1,
    K_TYPE_AIC = 2,
    K_TYPE_AIV = 3,
    K_TYPE_MIX_AIC_MAIN = 4,
    K_TYPE_MIX_AIV_MAIN = 5,
    K_TYPE_AIC_ROLLBACK = 6,
    K_TYPE_AIV_ROLLBACK = 7,
    K_TYPE_MAX
} KernelTypeUt;

uint32_t AscendCFunctionGetMetaInfoKtypeStubMax(void *funcHandle, unsigned int *curKernelType)
{
    *curKernelType = K_TYPE_MAX;
    return 0;
}

uint32_t AscendCFunctionGetMetaInfoKtypeStubAicore(void *funcHandle, unsigned int *curKernelType)
{
    *curKernelType = K_TYPE_AICORE;
    return 0;
}

uint32_t AscendCFunctionGetMetaInfoKtypeStubAic(void *funcHandle, unsigned int *curKernelType)
{
    *curKernelType = K_TYPE_AIC;
    return 0;
}

uint32_t AscendCFunctionGetMetaInfoKtypeStubAiv(void *funcHandle, unsigned int *curKernelType)
{
    *curKernelType = K_TYPE_AIV;
    return 0;
}

uint32_t AscendCFunctionGetMetaInfoKtypeStubMixAic(void *funcHandle, unsigned int *curKernelType)
{
    *curKernelType = K_TYPE_MIX_AIC_MAIN;
    return 0;
}

uint32_t AscendCFunctionGetMetaInfoKtypeStubMixAiv(void *funcHandle, unsigned int *curKernelType)
{
    *curKernelType = K_TYPE_MIX_AIV_MAIN;
    return 0;
}

uint32_t AscendCFunctionGetMetaInfoKCoreRation0_1(void *funcHandle, unsigned short *coreAicRation,
    unsigned short *coreAivRation)
{
    *coreAicRation = 0;
    *coreAivRation = 1;
    return 0;
}

uint32_t AscendCFunctionGetMetaInfoKCoreRation1_0(void *funcHandle, unsigned short *coreAicRation,
    unsigned short *coreAivRation)
{
    *coreAicRation = 1;
    *coreAivRation = 0;
    return 0;
}

uint32_t AscendCFunctionGetMetaInfoKCoreRation1_1(void *funcHandle, unsigned short *coreAicRation,
    unsigned short *coreAivRation)
{
    *coreAicRation = 1;
    *coreAivRation = 1;
    return 0;
}

uint32_t AscendCFunctionGetMetaInfoKCoreRation1_2(void *funcHandle, unsigned short *coreAicRation,
    unsigned short *coreAivRation)
{
    *coreAicRation = 1;
    *coreAivRation = 2;
    return 0;
}

TEST_F(TEST_ASCENDC_RUNTIME, AscendLaunchKernelWithHostArgs_unknown_soc) {
    void* funcHandle = nullptr;
    uint32_t numBlocks = 1;
    void* stream = nullptr;
    void* hostArgs = nullptr;
    size_t argsSize = 0;
    uint32_t ubufDynamicSize = 0;
    EXPECT_EXIT({
        MOCKER(aclrtGetSocName).stubs().will(returnValue(static_cast<char const*>(nullptr)));
        int32_t ret = AscendLaunchKernelWithHostArgs(funcHandle, numBlocks, stream, hostArgs, argsSize, ubufDynamicSize);
        exit(ret == 0 ? 0 : 1);
    }, ::testing::ExitedWithCode(1), "");
}

TEST_F(TEST_ASCENDC_RUNTIME, AscendLaunchKernelWithHostArgs_unsupport_soc) {
    void* funcHandle = nullptr;
    uint32_t numBlocks = 1;
    void* stream = nullptr;
    void* hostArgs = nullptr;
    size_t argsSize = 0;
    uint32_t ubufDynamicSize = 0;
    EXPECT_EXIT({
        MOCKER(aclrtGetSocName).stubs().will(returnValue((const char*)"UnsupportedSoc"));
        int32_t ret = AscendLaunchKernelWithHostArgs(funcHandle, numBlocks, stream, hostArgs, argsSize, ubufDynamicSize);
        exit(ret == 0 ? 0 : 1);
    }, ::testing::ExitedWithCode(1), "");
}

TEST_F(TEST_ASCENDC_RUNTIME, AscendLaunchKernelWithHostArgs_910B) {
    void* funcHandle = nullptr;
    uint32_t numBlocks = 1;
    void* stream = nullptr;
    void* hostArgs = nullptr;
    size_t argsSize = 0;
    uint32_t ubufDynamicSize = 0;
    EXPECT_EXIT({
        MOCKER(aclrtGetSocName).stubs().will(returnValue((const char*)"Ascend910B"));
        int32_t ret = AscendLaunchKernelWithHostArgs(funcHandle, numBlocks, stream, hostArgs, argsSize, 0);
        exit(ret == 0 ? 0 : 1);
    }, ::testing::ExitedWithCode(0), "");
}

TEST_F(TEST_ASCENDC_RUNTIME, AscendLaunchKernelWithHostArgs_310P_aicore) {
    void* funcHandle = nullptr;
    uint32_t numBlocks = 1;
    void* stream = nullptr;
    void* hostArgs = nullptr;
    size_t argsSize = 0;
    uint32_t ubufDynamicSize = 0;
    EXPECT_EXIT({
        MOCKER(aclrtGetSocName).stubs().will(returnValue((const char*)"Ascend310P"));
        MOCKER(AscendCFunctionGetMetaInfoKtype).stubs().will(invoke(AscendCFunctionGetMetaInfoKtypeStubAicore));
        int32_t ret = AscendLaunchKernelWithHostArgs(funcHandle, numBlocks, stream, hostArgs, argsSize, ubufDynamicSize);
        exit(ret == 0 ? 0 : 1);
    }, ::testing::ExitedWithCode(0), "");
}

TEST_F(TEST_ASCENDC_RUNTIME, AscendLaunchKernelWithHostArgs_310P_mix_vector_core) {
    void* funcHandle = nullptr;
    uint32_t numBlocks = 1;
    void* stream = nullptr;
    void* hostArgs = nullptr;
    size_t argsSize = 0;
    uint32_t ubufDynamicSize = 0;
    EXPECT_EXIT({
        MOCKER(aclrtGetSocName).stubs().will(returnValue((const char*)"Ascend310P"));
        MOCKER(AscendCFunctionGetMetaInfoKtype).stubs().will(invoke(AscendCFunctionGetMetaInfoKtypeStubAiv));
        int32_t ret = AscendLaunchKernelWithHostArgs(funcHandle, numBlocks, stream, hostArgs, argsSize, ubufDynamicSize);
        exit(ret == 0 ? 0 : 1);
    }, ::testing::ExitedWithCode(1), "");
}

TEST_F(TEST_ASCENDC_RUNTIME, AscendLaunchKernelWithHostArgs_950) {
    void* funcHandle = nullptr;
    uint32_t numBlocks = 1;
    void* stream = nullptr;
    void* hostArgs = nullptr;
    size_t argsSize = 0;
    uint32_t ubufDynamicSize = 512;
    EXPECT_EXIT({
        MOCKER(aclrtGetSocName).stubs().will(returnValue((const char*)"Ascend950"));
        int32_t ret = AscendLaunchKernelWithHostArgs(funcHandle, numBlocks, stream, hostArgs, argsSize, ubufDynamicSize);
        exit(ret == 0 ? 0 : 1);
    }, ::testing::ExitedWithCode(0), "");
}

TEST_F(TEST_ASCENDC_RUNTIME, AscendCFunctionGetMetaInfoKtype) {
    void* funcHandle = nullptr;
    unsigned int kernelType = 5;
    uint32_t ret = AscendCFunctionGetMetaInfoKtype(funcHandle, &kernelType);
    EXPECT_EQ(ret, 0);
}

TEST_F(TEST_ASCENDC_RUNTIME, AscendCFunctionGetMetaInfoCoreRation) {
    void* funcHandle = nullptr;
    unsigned short aicRation;
    unsigned short aivRation;
    uint32_t ret = AscendCFunctionGetMetaInfoCoreRation(funcHandle, &aicRation, &aivRation);
    EXPECT_EQ(ret, 0);
}

TEST_F(TEST_ASCENDC_RUNTIME, AscendCFunctionGetMetaInfoKtypeFailed) {
    void* funcHandle = nullptr;
    unsigned int kernelType = 5;
    MOCKER(rtFunctionGetMetaInfo).stubs().will(returnValue(1));
    uint32_t ret = AscendCFunctionGetMetaInfoKtype(funcHandle, &kernelType);
    EXPECT_EQ(ret, 1);
}

TEST_F(TEST_ASCENDC_RUNTIME, AscendCFunctionGetMetaInfoCoreRationFailed) {
    void* funcHandle = nullptr;
    unsigned short aicRation;
    unsigned short aivRation;
    MOCKER(rtFunctionGetMetaInfo).stubs().will(returnValue(1));
    uint32_t ret = AscendCFunctionGetMetaInfoCoreRation(funcHandle, &aicRation, &aivRation);
    EXPECT_EQ(ret, 1);
}

TEST_F(TEST_ASCENDC_RUNTIME, AscendCGetProfkTypeImplGetKtypeFailed) {
    void* funcHandle = nullptr;
    MOCKER(AscendCFunctionGetMetaInfoKtype).stubs().will(returnValue(1));
    uint32_t ret = AscendCGetProfkTypeImpl(funcHandle);
    EXPECT_EQ(ret, 5);
}

TEST_F(TEST_ASCENDC_RUNTIME, AscendCGetProfkTypeImplGetCoreRationFailed) {
    void* funcHandle = nullptr;
    MOCKER(AscendCFunctionGetMetaInfoKtype).stubs().will(invoke(AscendCFunctionGetMetaInfoKtypeStubMixAic));
    MOCKER(AscendCFunctionGetMetaInfoCoreRation).stubs().will(returnValue(1));
    uint32_t ret = AscendCGetProfkTypeImpl(funcHandle);
    EXPECT_EQ(ret, 5);
}

TEST_F(TEST_ASCENDC_RUNTIME, AscendCGetProfkTypeImplMixAiv1_0) {
    void* funcHandle = nullptr;
    MOCKER(AscendCFunctionGetMetaInfoKtype).stubs().will(invoke(AscendCFunctionGetMetaInfoKtypeStubMixAiv));
    MOCKER(AscendCFunctionGetMetaInfoCoreRation).stubs().will(invoke(AscendCFunctionGetMetaInfoKCoreRation0_1));
    uint32_t ret = AscendCGetProfkTypeImpl(funcHandle);
    EXPECT_EQ(ret, 7);
}

TEST_F(TEST_ASCENDC_RUNTIME, AscendCGetProfkTypeImplMixAic1_0) {
    void* funcHandle = nullptr;
    MOCKER(AscendCFunctionGetMetaInfoKtype).stubs().will(invoke(AscendCFunctionGetMetaInfoKtypeStubMixAic));
    MOCKER(AscendCFunctionGetMetaInfoCoreRation).stubs().will(invoke(AscendCFunctionGetMetaInfoKCoreRation1_0));
    uint32_t ret = AscendCGetProfkTypeImpl(funcHandle);
    EXPECT_EQ(ret, 8);
}

TEST_F(TEST_ASCENDC_RUNTIME, AscendCGetProfkTypeImplMixAic1_1) {
    void* funcHandle = nullptr;
    MOCKER(AscendCFunctionGetMetaInfoKtype).stubs().will(invoke(AscendCFunctionGetMetaInfoKtypeStubMixAic));
    MOCKER(AscendCFunctionGetMetaInfoCoreRation).stubs().will(invoke(AscendCFunctionGetMetaInfoKCoreRation1_1));
    uint32_t ret = AscendCGetProfkTypeImpl(funcHandle);
    EXPECT_EQ(ret, 9);
}

TEST_F(TEST_ASCENDC_RUNTIME, AscendCGetProfkTypeImplMixAic1_2) {
    void* funcHandle = nullptr;
    MOCKER(AscendCFunctionGetMetaInfoKtype).stubs().will(invoke(AscendCFunctionGetMetaInfoKtypeStubMixAic));
    MOCKER(AscendCFunctionGetMetaInfoCoreRation).stubs().will(invoke(AscendCFunctionGetMetaInfoKCoreRation1_2));
    uint32_t ret = AscendCGetProfkTypeImpl(funcHandle);
    EXPECT_EQ(ret, 10);
}

TEST_F(TEST_ASCENDC_RUNTIME, AscendCGetProfkTypeImplAic) {
    void* funcHandle = nullptr;
    MOCKER(AscendCFunctionGetMetaInfoKtype).stubs().will(invoke(AscendCFunctionGetMetaInfoKtypeStubAic));
    uint32_t ret = AscendCGetProfkTypeImpl(funcHandle);
    EXPECT_EQ(ret, 6);
}

TEST_F(TEST_ASCENDC_RUNTIME, AscendCGetProfkTypeImplAiv) {
    void* funcHandle = nullptr;
    MOCKER(AscendCFunctionGetMetaInfoKtype).stubs().will(invoke(AscendCFunctionGetMetaInfoKtypeStubAiv));
    uint32_t ret = AscendCGetProfkTypeImpl(funcHandle);
    EXPECT_EQ(ret, 5);
}

TEST_F(TEST_ASCENDC_RUNTIME, AscendCGetProfkTypeImplAicore) {
    void* funcHandle = nullptr;
    MOCKER(AscendCFunctionGetMetaInfoKtype).stubs().will(invoke(AscendCFunctionGetMetaInfoKtypeStubAicore));
    uint32_t ret = AscendCGetProfkTypeImpl(funcHandle);
    EXPECT_EQ(ret, 2);
}

TEST_F(TEST_ASCENDC_RUNTIME, AscendCGetProfkTypeImplothers) {
    void* funcHandle = nullptr;
    MOCKER(AscendCFunctionGetMetaInfoKtype).stubs().will(invoke(AscendCFunctionGetMetaInfoKtypeStubMax));
    uint32_t ret = AscendCGetProfkTypeImpl(funcHandle);
    EXPECT_EQ(ret, 5);
}