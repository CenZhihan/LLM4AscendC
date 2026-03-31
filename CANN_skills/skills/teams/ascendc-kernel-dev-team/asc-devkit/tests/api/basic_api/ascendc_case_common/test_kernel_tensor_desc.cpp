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
#include "kernel_tensor_impl.h"
#include <vector>
#include "test_utils.h"

using namespace AscendC;
using namespace std;

/* **************************** LocalTensor Print ****************************** */
struct PrintTensorDescParams {
    string dataType;
    int32_t printSize;
    string goldenStr;
};

template <typename T>
struct SelfTensorDesc : public TensorTrait<T> {
    static constexpr int32_t tensorLen = 256;
};

class TEST_TENSOR_DESC_PRINT : public testing::Test, public testing::WithParamInterface<PrintTensorDescParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TENSOR_DESC_PRINT, TEST_TENSOR_DESC_PRINT,
    ::testing::Values(PrintTensorDescParams { "uint32_t", 16, "0000 : 0 1 2 3 4 5 6 7 \n0008 : 8 9 10 11 12 13 14 15 \n" },
    PrintTensorDescParams { "uint32_t", 7, "0000 : 0 1 2 3 4 5 6 \n" },
    PrintTensorDescParams { "int32_t", 16, "0000 : 0 1 2 3 4 5 6 7 \n0008 : 8 9 10 11 12 13 14 15 \n" },
    PrintTensorDescParams { "int32_t", 7, "0000 : 0 1 2 3 4 5 6 \n" },
    PrintTensorDescParams { "half", 33,
    "0000 : 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \n0016 : 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \n0032 : 32 "
    "\n" },
    PrintTensorDescParams { "half", 17, "0000 : 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \n0016 : 16 \n" },
    PrintTensorDescParams { "half", 7, "0000 : 0 1 2 3 4 5 6 \n" },
    PrintTensorDescParams { "float", 17, "0000 : 0 1 2 3 4 5 6 7 \n0008 : 8 9 10 11 12 13 14 15 \n0016 : 16 \n" },
    PrintTensorDescParams { "float", 7, "0000 : 0 1 2 3 4 5 6 \n" },
    PrintTensorDescParams { "uint16_t", 33,
    "0000 : 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \n0016 : 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \n0032 : 32 "
    "\n" },
    PrintTensorDescParams { "uint16_t", 17, "0000 : 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \n0016 : 16 \n" },
    PrintTensorDescParams { "uint16_t", 7, "0000 : 0 1 2 3 4 5 6 \n" },
    PrintTensorDescParams { "uint8_t", 33,
    "0000 : 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \n0032 : 32 \n" },
    PrintTensorDescParams { "int8_t", 33,
    "0000 : 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \n0032 : 32 \n" },
    PrintTensorDescParams { "bool", 33,
    "0000 : 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 \n0032 : 1 \n" }));

template <typename T> void GetPrintString(const string& dataType, const int32_t printSize, string& retStr)
{
    const int32_t maxLen = 64;
    TBuffAddr addr;
    addr.logicPos = 1;
    LocalTensor<SelfTensorDesc<T>> tensor(addr);
    tensor.InitBuffer(0, maxLen);
    for (int32_t i = 0; i < maxLen; i++) {
        tensor.SetValue(i, T(i));
    }
    tensor.Print(printSize);
    retStr = tensor.os_.str();
}

TEST_P(TEST_TENSOR_DESC_PRINT, TensorDescPrintTest)
{
    auto param = GetParam();
    string curStr;

    if (param.dataType == "uint32_t") {
        GetPrintString<uint32_t>(param.dataType, param.printSize, curStr);
    } else if (param.dataType == "int32_t") {
        GetPrintString<int32_t>(param.dataType, param.printSize, curStr);
    } else if (param.dataType == "half") {
        GetPrintString<half>(param.dataType, param.printSize, curStr);
    } else if (param.dataType == "float") {
        GetPrintString<float>(param.dataType, param.printSize, curStr);
    } else if (param.dataType == "uint16_t") {
        GetPrintString<uint16_t>(param.dataType, param.printSize, curStr);
    } else if (param.dataType == "uint8_t") {
        GetPrintString<uint8_t>(param.dataType, param.printSize, curStr);
    } else if (param.dataType == "int8_t") {
        GetPrintString<int8_t>(param.dataType, param.printSize, curStr);
    } else if (param.dataType == "bool") {
        GetPrintString<bool>(param.dataType, param.printSize, curStr);
    }
    EXPECT_EQ(curStr, param.goldenStr);
}

class TEST_TENSOR_DESC : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};
/* **************************** LocalTensor Operator[] ****************************** */
TEST_F(TEST_TENSOR_DESC, TensorBracketWithoutInitBuffer)
{
    const int32_t maxLen = 32;
    const int32_t offsetIndex = 16;
    int32_t* data = new int32_t[maxLen];
    for (int32_t i = 0; i < maxLen; i++) {
        data[i] = i;
    }

    TBuffAddr addr;
    addr.logicPos = 1;
    addr.bufferHandle = nullptr;
    addr.dataLen = maxLen * sizeof(int32_t);
    addr.bufferAddr = 0;
    addr.absAddr = (uint8_t*)data;

    LocalTensor<SelfTensorDesc<int32_t>> tensor(addr);
    tensor.Print();
    LocalTensor<SelfTensorDesc<int32_t>> tensorPart = tensor[offsetIndex];
    string str1 = tensor.os_.str();
    string str2 = tensorPart.os_.str();
    string str1Golden = "0000 : 0 1 2 3 4 5 6 7 \n0008 : 8 9 10 11 12 13 14 15 \n0016 : 16 17 18 19 20 21 22 23 \n0024 "
        ": 24 25 26 27 28 29 30 31 \n";
    string str2Golden = "";
    EXPECT_EQ(str1, str1Golden);
    EXPECT_EQ(str2, str2Golden);
    EXPECT_EQ(tensorPart.GetSize(), maxLen - offsetIndex);
    EXPECT_EQ(tensorPart.GetLength(), (maxLen - offsetIndex) * sizeof(int32_t));
    EXPECT_EQ(tensor.address_.absAddr, (uint8_t*)data);
    EXPECT_EQ(tensorPart.address_.absAddr, (uint8_t*)data + offsetIndex * sizeof(int32_t));
    EXPECT_EQ(tensor.address_.bufferAddr, 0);
    EXPECT_EQ(tensorPart.address_.bufferAddr, 0 + +offsetIndex * sizeof(int32_t));

    delete[] data;
}

TEST_F(TEST_TENSOR_DESC, TensorBracketWithInitBuffer)
{
    const int32_t maxLen = 32;
    const int32_t offsetIndex = 24;

    TBuffAddr addr;
    addr.logicPos = 1;
    LocalTensor<SelfTensorDesc<int32_t>> tensor(addr);
    tensor.InitBuffer(0, maxLen);
    for (int32_t i = 0; i < maxLen; i++) {
        tensor.SetValue(i, i);
    };
    tensor.Print();
    LocalTensor<SelfTensorDesc<int32_t>> tensorPart = tensor[offsetIndex];
    string str1 = tensor.os_.str();
    string str2 = tensorPart.os_.str();
    string str1Golden = "0000 : 0 1 2 3 4 5 6 7 \n0008 : 8 9 10 11 12 13 14 15 \n0016 : 16 17 18 19 20 21 22 23 \n0024 "
        ": 24 25 26 27 28 29 30 31 \n";
    string str2Golden = "";
    EXPECT_EQ(str1, str1Golden);
    EXPECT_EQ(str2, str2Golden);
    EXPECT_EQ(tensorPart.GetSize(), maxLen - offsetIndex);
    EXPECT_EQ(tensorPart.GetLength(), (maxLen - offsetIndex) * sizeof(int32_t));
}

/* **************************** LocalTensor Operator= ****************************** */
TEST_F(TEST_TENSOR_DESC, TensorEqual)
{
    const int32_t maxLen = 32;
    int32_t* data = new int32_t[maxLen];
    for (int32_t i = 0; i < maxLen; i++) {
        data[i] = i;
    }

    TBuffAddr addr;
    addr.logicPos = 1;
    addr.bufferHandle = nullptr;
    addr.dataLen = maxLen * sizeof(int32_t);
    addr.bufferAddr = 0;
    addr.absAddr = (uint8_t*)data;

    LocalTensor<SelfTensorDesc<int32_t>> tensor1(addr);
    tensor1.Print();
    LocalTensor<SelfTensorDesc<int32_t>> tensor2 = tensor1;
    string str1 = tensor1.os_.str();
    string str2 = tensor2.os_.str();
    string strGolden = "0000 : 0 1 2 3 4 5 6 7 \n0008 : 8 9 10 11 12 13 14 15 \n0016 : 16 17 18 19 20 21 22 23 \n0024 "
        ": 24 25 26 27 28 29 30 31 \n";
    EXPECT_EQ(str1, strGolden);
    EXPECT_EQ(str2, "");
    EXPECT_EQ(tensor1.GetSize(), maxLen);
    EXPECT_EQ(tensor2.GetSize(), maxLen);
    EXPECT_EQ(tensor1.GetLength(), maxLen * sizeof(int32_t));
    EXPECT_EQ(tensor2.GetLength(), maxLen * sizeof(int32_t));
    EXPECT_EQ(tensor1.address_.absAddr, (uint8_t*)data);
    EXPECT_EQ(tensor2.address_.absAddr, (uint8_t*)data);
    EXPECT_EQ(tensor1.address_.bufferAddr, 0);
    EXPECT_EQ(tensor2.address_.bufferAddr, 0);

    delete[] data;
}

/* **************************** LocalTensor CopyConstruct ****************************** */
TEST_F(TEST_TENSOR_DESC, TensorConpyConstrcutWithInitBuffer)
{
    const int32_t maxLen = 32;
    TBuffAddr addr;
    addr.logicPos = 1;
    LocalTensor<SelfTensorDesc<int32_t>> tensor1(addr);
    tensor1.InitBuffer(0, maxLen);
    for (int32_t i = 0; i < maxLen; i++) {
        tensor1.SetValue(i, i);
    };
    EXPECT_EQ(tensor1.os_.str(), "");
    tensor1.Print();
    LocalTensor<SelfTensorDesc<int32_t>> tensor2(tensor1);
    string str1 = tensor1.os_.str();
    string str2 = tensor2.os_.str();
    string strGolden = "0000 : 0 1 2 3 4 5 6 7 \n0008 : 8 9 10 11 12 13 14 15 \n0016 : 16 17 18 19 20 21 22 23 \n0024 "
        ": 24 25 26 27 28 29 30 31 \n";
    EXPECT_EQ(str1, strGolden);
    EXPECT_EQ(str2, "");
    EXPECT_EQ(tensor2.GetSize(), maxLen);
    EXPECT_EQ(tensor2.GetLength(), maxLen * sizeof(int32_t));
}

TEST_F(TEST_TENSOR_DESC, TensorCopyConstrcutWithoutInitBuffer)
{
    const int32_t maxLen = 32;
    int32_t* data = new int32_t[maxLen];
    for (int32_t i = 0; i < maxLen; i++) {
        data[i] = i;
    }

    TBuffAddr addr;
    addr.logicPos = 1;
    addr.bufferHandle = nullptr;
    addr.dataLen = maxLen * sizeof(int32_t);
    addr.bufferAddr = 0;
    addr.absAddr = (uint8_t*)data;

    LocalTensor<SelfTensorDesc<int32_t>> tensor1(addr);
    tensor1.Print();
    LocalTensor<SelfTensorDesc<int32_t>> tensor2(tensor1);
    string str1 = tensor1.os_.str();
    string str2 = tensor2.os_.str();
    string str1Golden = "0000 : 0 1 2 3 4 5 6 7 \n0008 : 8 9 10 11 12 13 14 15 \n0016 : 16 17 18 19 20 21 22 23 \n0024 "
        ": 24 25 26 27 28 29 30 31 \n";
    string str2Golden = "";
    EXPECT_EQ(str1, str1Golden);
    EXPECT_EQ(str2, str2Golden);
    EXPECT_EQ(tensor2.GetSize(), maxLen);
    EXPECT_EQ(tensor2.GetLength(), maxLen * sizeof(int32_t));
    EXPECT_EQ(tensor1.address_.absAddr, (uint8_t*)data);
    EXPECT_EQ(tensor2.address_.absAddr, (uint8_t*)data);
    EXPECT_EQ(tensor1.address_.bufferAddr, 0);
    EXPECT_EQ(tensor2.address_.bufferAddr, 0);

    delete[] data;
}

#define USER_TAG 10
TEST_F(TEST_TENSOR_DESC, TensorSetUserTagTest)
{
    TTagType tagAddr = 0;
    TBuffAddr addr = {0};
    addr.bufferHandle = (TBufHandle)malloc(sizeof(TBufType));
    LocalTensor<SelfTensorDesc<int32_t>> tensor(addr);
    // 传入数值
    int32_t userTag = USER_TAG;
    tensor.SetUserTag((TTagType)userTag);
    EXPECT_EQ(USER_TAG, tensor.GetUserTag());

    // 将指针转成TTagType作为userTag
    tensor.SetUserTag((TTagType)(uintptr_t)&userTag);
    EXPECT_EQ((TTagType)(uintptr_t)&userTag, tensor.GetUserTag());

    free(addr.bufferHandle);
}

TEST_F(TEST_TENSOR_DESC, TensorReinterpretCastTest)
{
    const int32_t maxLen = 64;
    TBuffAddr addr;
    addr.logicPos = 1;
    LocalTensor<SelfTensorDesc<int32_t>> tensor1(addr);
    tensor1.InitBuffer(0, maxLen);

    LocalTensor<SelfTensorDesc<int16_t>> tensorOut = tensor1.ReinterpretCast<SelfTensorDesc<int16_t>>();
    EXPECT_EQ(tensorOut.address_.logicPos, tensor1.address_.logicPos);
    EXPECT_EQ(tensorOut.address_.bufferHandle, tensor1.address_.bufferHandle);
    EXPECT_EQ(tensorOut.address_.dataLen, 256);
    EXPECT_EQ(tensorOut.address_.bufferAddr, tensor1.address_.bufferAddr);
    EXPECT_EQ(tensorOut.address_.absAddr, tensor1.address_.absAddr);
}

TEST_F(TEST_TENSOR_DESC, LocalTensorSupportInt4Test)
{
    const int32_t num = 16;
    uint64_t data = 0xfedcba9876543210;

    TBuffAddr addr;
    addr.logicPos = 1;
    addr.bufferHandle = nullptr;
    addr.dataLen = num / INT4_TWO;
    addr.bufferAddr = 0;
    addr.absAddr = (uint8_t *)&data;

    LocalTensor<SelfTensorDesc<int4b_t>> tensor1(addr);
    EXPECT_EQ(tensor1.GetSize(), 16);
    for (int i = 0; i < num; i++) {
        EXPECT_EQ(tensor1.GetValue(i), (int4b_t)i);
    }

    LocalTensor<SelfTensorDesc<int4b_t>> tensor2 = tensor1[num / 2];
    for (int i = 0; i < num / 2; i++) {
        EXPECT_EQ(tensor2.GetValue(i), (int4b_t)(-8 + i));
    }

    for (int i = 0; i < num; i++) {
        tensor1.SetValue(i, (int4b_t)(0xf - i));
    }
    EXPECT_EQ(data, 0x123456789abcdef);

    LocalTensor<SelfTensorDesc<uint8_t>> tensor3 = tensor1.ReinterpretCast<SelfTensorDesc<uint8_t>>();
    EXPECT_EQ(tensor3.GetSize(), sizeof(data) / sizeof(uint8_t));
}

TEST_F(TEST_TENSOR_DESC, GlobalTensorSupportInt4Test)
{
    const int32_t num = 16;
    uint64_t data = 0xfedcba9876543210;
    uint8_t *dataPtr = (uint8_t *)&data;
    __gm__ int4b_t *gmAddr = (__gm__ int4b_t *)&data;

    GlobalTensor<SelfTensorDesc<int4b_t>> glbTensor1;
    GlobalTensor<SelfTensorDesc<int4b_t>> glbTensor2;
    glbTensor1.SetGlobalBuffer(gmAddr, num);
    glbTensor2 = glbTensor1[num / 2];

    EXPECT_EQ(glbTensor1.GetSize(), num);
    EXPECT_EQ(glbTensor1.GetPhyAddr(), (int4b_t *)&data);

    for (int i = 0; i < num / 2; i++) {
        EXPECT_EQ(glbTensor1.GetPhyAddr(i * 2), (int4b_t *)&dataPtr[i]);
    }

    for (int i = 0; i < num; i++) {
        EXPECT_EQ(glbTensor1.GetValue(i), (int4b_t)i);
    }

    for (int i = 0; i < num / 2; i++) {
        EXPECT_EQ(glbTensor2.GetValue(i), (int4b_t)(-8 + i));
    }

    for (int i = 0; i < num; i++) {
        glbTensor1.SetValue(i, (int4b_t)(0xf - i));
    }
    EXPECT_EQ(data, 0x123456789abcdef);
}

TEST_F(TEST_TENSOR_DESC, GlobalTensorSetValueTest)
{
    GlobalTensor<SelfTensorDesc<int32_t>> glbTensor1;
    int32_t glbBuffer[32] = {0};
    glbTensor1.SetGlobalBuffer(glbBuffer, 32);
    for (int i = 0; i < 32; ++i) {
        glbTensor1.SetValue(i, i);
    }
    for (int i = 0; i < 32; i++) {
        EXPECT_EQ(glbTensor1.GetValue(i), i);
    }
}