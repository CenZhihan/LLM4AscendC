/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <dlfcn.h>
#include <iostream>

#include "asc_platform.h"

extern "C" int ASCInitSocSpec(const char* soc_version, const char* core_type, const char* aicore_num, const char* l1FusionFlag);

extern "C" int ASCSetSocSpec(const char* spec);

extern "C" void ASCGetSocSpec(const char* spec, char* buffer, int buffer_size);

extern "C" int ASCTeUpdateVersion(const char* soc_version, const char* core_type, const char* aicore_num, const char* l1_fusion);

extern "C" void* CreateStrStrMap();

extern "C" void MapInsert(void* mapPoint, const char* key, const char* value);

extern "C" void MapDelete(void* mapPoint);

extern "C" bool ASCSetPlatformInfoRes(const int32_t device_id, void* mapPoint);

extern "C" bool ASCSetCoreNumByCoreType(const char* core_type);

class TEST_ASC_PLATFORM_API : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() 
    {
        GlobalMockObject::verify();
    }
};

std::string core_type_stub;

// stub GetPlatformRes
bool MockGetPlatformRes(fe::PlatFormInfos *This, const std::string &label, const std::string &key, std::string &val) {
    val = "";
    static const std::map<std::string, std::string> info_map_stub = {
        {"Short_SoC_version", "Ascend910B"},
        {"COMPILER_ARCH", "dav-c220-vec"},
        {"cube_vector_combine", "split"},
        {"cube_freq", "66"},
        {"cube_core_cnt", "4"},
        {"vector_core_cnt", "2"},
        {"ub_size", "1024"},
        {"fb_size", "512"},
        {"fb0_size", "128"},
        {"fb1_size", "128"},
        {"fb2_size", "128"},
        {"fb3_size", "128"},
        {"l0_a_size", "32"},
        {"l0_b_size", "32"},
        {"l0_c_size", "32"},
        {"l2_size", "256"},
        {"l1_size", "64"},
        {"bt_size", "2048"},
        {"ai_core_cnt", "6"},
        {"vector_reg_width", "4"},
        {"cube_vector_split", "true"},
    };

    auto it = info_map_stub.find(key);
    if (it != info_map_stub.end()) {
        val = it->second;
    } else {
        return false;
    }

    return true;
}

std::string MockGetCoreType(fe::OptionalInfos *This) {
    return core_type_stub;
}

void MockSetSocSpec(const std::string& key) {
    core_type_stub = key;
}

TEST_F(TEST_ASC_PLATFORM_API, test_ASCInitSocSpec)
{
    char b_soc_version[16] = "Ascend910B1";
    char b_core_type[16] = "AiCore";
    char b_aicore_num[16] = "";
    char b_l1_fusion_flag[16] = "";

    EXPECT_EQ(ASCInitSocSpec("Ascend310P1", "AiCore", "", "false"), 0);
    int result = ASCInitSocSpec("Ascend910", b_core_type, b_aicore_num, b_l1_fusion_flag);
    result = ASCInitSocSpec(b_soc_version, b_core_type, b_aicore_num, b_l1_fusion_flag);
    EXPECT_EQ(result, 0);
}

TEST_F(TEST_ASC_PLATFORM_API, test_ASCGetSocSpec)
{
    core_type_stub = "VectorCore";
    int buffer_size = 100;
    char buffer[buffer_size];

    MOCKER_CPP(&fe::PlatFormInfos::GetPlatformRes, bool(fe::PlatFormInfos::*)(const std::string&, const std::string&, std::string&)).stubs().will(invoke(MockGetPlatformRes));
    MOCKER_CPP(&fe::OptionalInfos::GetCoreType, std::string(fe::OptionalInfos::*)()).stubs().will(invoke(MockGetCoreType));

    ASCGetSocSpec("SOC_VERSION", buffer, buffer_size);
    EXPECT_STREQ(buffer, "Ascend910B1");
    ASCGetSocSpec("SHORT_SOC_VERSION", buffer, buffer_size);
    EXPECT_STREQ(buffer, "Ascend910B");
    ASCGetSocSpec("COMPILER_ARCH", buffer, buffer_size);
    EXPECT_STREQ(buffer, "dav-c220-vec");
    ASCGetSocSpec("cube_vector_combine", buffer, buffer_size);
    EXPECT_STREQ(buffer, "split");
    ASCGetSocSpec("AICORE_TYPE", buffer, buffer_size);
    EXPECT_STREQ(buffer, "VectorCore");

    ASCGetSocSpec("Compiler_arch", buffer, buffer_size);
    EXPECT_STREQ(buffer, "unknown");
    ASCGetSocSpec("Compiler_aicpu_support_os", buffer, buffer_size);
    EXPECT_STREQ(buffer, "unknown");
}

TEST_F(TEST_ASC_PLATFORM_API, test_ASCSetSocSpec)
{
    int buffer_size = 100;
    char buffer[buffer_size];

    MOCKER_CPP(&fe::PlatFormInfos::GetPlatformRes, bool(fe::PlatFormInfos::*)(const std::string&, const std::string&, std::string&)).stubs().will(invoke(MockGetPlatformRes));
    MOCKER_CPP(&fe::OptionalInfos::GetCoreType, std::string(fe::OptionalInfos::*)()).stubs().will(invoke(MockGetCoreType));

    ASCGetSocSpec("AICORE_TYPE", buffer, buffer_size);
    EXPECT_STREQ(buffer, "VectorCore");

    EXPECT_EQ(ASCSetSocSpec("16"), 0);
    EXPECT_EQ(ASCSetSocSpec("true"), 0);
    EXPECT_EQ(ASCSetSocSpec("error"), -1);
    MockSetSocSpec("AiCore");
    int result = ASCSetSocSpec("Ascend910");
    result = ASCSetSocSpec("AiCore");
    EXPECT_EQ(result, 0);
    
    ASCGetSocSpec("Compiler_arch", buffer, buffer_size);
    EXPECT_STREQ(buffer, "unknown");
    ASCGetSocSpec("SOC_VERSION", buffer, buffer_size);
    EXPECT_STREQ(buffer, "Ascend910B1");
    ASCGetSocSpec("AICORE_TYPE", buffer, buffer_size);
    EXPECT_STREQ(buffer, "AiCore");
}

TEST_F(TEST_ASC_PLATFORM_API, test_CceConfBase_1)
{
    tvm::cceconf::CceConfBase* conf = tvm::cceconf::CceConfBase::GetInstance();

    MOCKER_CPP(&fe::PlatFormInfos::GetPlatformRes, bool(fe::PlatFormInfos::*)(const std::string&, const std::string&, std::string&)).stubs().will(invoke(MockGetPlatformRes));
    MOCKER_CPP_VIRTUAL(conf, &tvm::cceconf::CceConfBase::IsC220).stubs().will(returnValue(true));

    conf->SetOptionalCoreType("DumpCore");
    conf->SetOptionalSocVersion("Ascend310P");
    EXPECT_EQ(conf->SetOptionalCoreType("AiCore"), true);
    conf->SetOptionalAicoreNum("16");
    conf->SetOptionalL1FusionFlag("");
    
    EXPECT_EQ(conf->SetOptionalCoreType("VectorCore"), false);
    conf->SetOptionalSocVersion("Ascend910B1");
    conf->SetOptionalAicoreNum("");
    conf->SetOptionalL1FusionFlag("false");

    EXPECT_EQ(conf->SetCoreNumByCoreType("AiCore"), true);
    EXPECT_STREQ(conf->GetSocVersion().c_str(), "Ascend910B1");
    
    EXPECT_EQ(conf->AcquireCubeCoreCnt(), 4);
    EXPECT_EQ(conf->AcquireCubeFreq(), 66);
    EXPECT_EQ(conf->AcquireVectorCoreCnt(), 2);
    EXPECT_EQ(conf->AcquireUBSize(), 1024);
    EXPECT_EQ(conf->AcquireFB0Size(), 128);
    EXPECT_EQ(conf->AcquireFB1Size(), 128);
    EXPECT_EQ(conf->AcquireFB2Size(), 128);
    EXPECT_EQ(conf->AcquireFB3Size(), 128);
    EXPECT_EQ(conf->AcquireFBSize(), 512);
    EXPECT_EQ(conf->AcquireCoreNum(), 16);
    EXPECT_EQ(conf->AcquireL2Size(), 256);
    EXPECT_EQ(conf->AcquireL1Size(), 64);
    EXPECT_EQ(conf->AcquireBTSize(), 2048);
    EXPECT_EQ(conf->AcquireL0ASize(), 32);
    EXPECT_EQ(conf->AcquireL0BSize(), 32);
    EXPECT_EQ(conf->AcquireL0CSize(), 32);
    EXPECT_EQ(conf->AcquireCubeVectorSplitFlag(), "true");
}

TEST_F(TEST_ASC_PLATFORM_API, test_CceConfBase_2)
{
    tvm::cceconf::CceConfBase* conf = tvm::cceconf::CceConfBase::GetInstance();

    MOCKER_CPP(&fe::PlatFormInfos::GetPlatformRes, bool(fe::PlatFormInfos::*)(const std::string&, const std::string&, std::string&)).stubs().will(invoke(MockGetPlatformRes));
    MOCKER_CPP_VIRTUAL(conf, &tvm::cceconf::CceConfBase::IsM310).stubs().will(returnValue(true));

    EXPECT_EQ(conf->AcquireCubeCoreCnt(), 4);
    EXPECT_EQ(conf->AcquireL1Size(), 64);
    EXPECT_EQ(conf->AcquireUBSize(), 1024);
    EXPECT_EQ(conf->AcquireL0ASize(), 32);
    EXPECT_EQ(conf->AcquireVectorRegBytesWidth(), 4);
    EXPECT_EQ(conf->AcquirePredicateRegBytesWidth(), 4);
    EXPECT_EQ(conf->AcquireWideRegBytesWidth(), 4);
}

TEST_F(TEST_ASC_PLATFORM_API, test_NewAPIs)
{
    char b_soc_version[16] = "Ascend910B1";
    char b_core_type[16] = "AiCore";
    char b_aicore_num[16] = "";
    char b_l1_fusion_flag[16] = "";

    int result = ASCTeUpdateVersion("Ascend910", "VectorCore", "16", "or");
    result = ASCTeUpdateVersion(b_soc_version, b_core_type, b_aicore_num, b_l1_fusion_flag);
    EXPECT_EQ(result, 0);

    void* mapPoint = CreateStrStrMap();
    MapInsert(mapPoint, "AiCore", "Ascend310P");
    MOCKER_CPP(&fe::PlatformInfoManager::GetPlatformInstanceByDevice, uint32_t(fe::PlatformInfoManager::*)(const uint32_t&, fe::PlatFormInfos&)).stubs().will(returnValue((uint32_t)0));
    MOCKER_CPP(&fe::PlatformInfoManager::GetPlatformInfos, uint32_t(fe::PlatformInfoManager::*)(const std::string, fe::PlatFormInfos&, fe::OptionalInfos&)).stubs().will(returnValue((uint32_t)0));
    EXPECT_EQ(ASCSetPlatformInfoRes(0, mapPoint), true);
    EXPECT_EQ(ASCSetCoreNumByCoreType("VectorCore"), true);
    MapDelete(mapPoint);
}