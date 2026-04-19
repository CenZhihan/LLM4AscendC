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
#include <memory.h>
#include "kernel_operator_list_tensor_intf.h"

using namespace AscendC;

class TEST_LIST_TENSOR : public testing::Test{
protected:
    void SetUp() {}
    void TearDown() {}
};

class ListTensor {
public:
    inline void PushTensor(__gm__ void* dataAddr)
    {
        dim1 = 0;
        count = 1;
        shapeList = {0};
        dataList.push_back((int64_t)dataAddr);
        index++;
    }

    inline void PushTensor(__gm__ void* dataAddr, uint32_t dim, uint64_t* shapes)
    {
        if ((dim == 0) && (*shapes == 0xffffffff)) {
            PushTensor(dataAddr);
            return;
        }
        if (index != 0) {
            ASSERT(dim1 == dim);
        } else {
            dim1 = dim;
        }
        std::vector<int64_t> shape(shapes, shapes + dim1);
        dataList.push_back((int64_t)dataAddr);
        if (shape != curShape) {
            curShape.assign(shape.begin(), shape.end());
            count++;
            shapeList.push_back(static_cast<uint64_t>(dim1) + ((static_cast<uint64_t>(index) << 32)));
            shapeList.insert(shapeList.end(), curShape.begin(), curShape.end());
        }
        index++;
    }

    inline bool GetTensorInfo(void* addr, uint64_t& size)
    {
        ASSERT(addr != nullptr);
        if (count == 1) {
            shapeList[0] = static_cast<uint64_t>(dim1) + (static_cast<uint64_t>(index) << 32);  // 高 32 比特为offset
            if (dim1 == 0) {
                shapeList.push_back(0xffffffff);
            }
        }
        uint64_t len = (1 + shapeList.size()) * sizeof(int64_t);
        list1.push_back(len);
        list1.insert(list1.end(), shapeList.begin(), shapeList.end());
        list1.insert(list1.end(), dataList.begin(), dataList.end());

        std::copy(list1.begin(), list1.end(), (int64_t *)addr);
        size = len + dataList.size() * sizeof(int64_t);
        return true;
    }

private:
    std::vector<int64_t> list1;
    std::vector<int64_t> shapeList;
    std::vector<int64_t> dataList;
    std::vector<int64_t> curShape;
    uint32_t dim1 = 0;
    uint32_t index = 0;
    uint32_t count = 0;
};

/* 传入shape, dim和数据指针，生成对应内存排布并解析 */
TEST_F(TEST_LIST_TENSOR, testListTensor0a)
{
    ListTensor list;
    int data[10] = {0,1,2,3,4,5,6,7,8,9};
    uint64_t shape[10][2] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10},
                            {11, 12}, {13, 14}, {15, 16}, {17, 18}, {19, 20}};
    uint32_t dim = 2;
    for (int i = 0; i < 10; i++) {
        list.PushTensor((void *)&data[i], dim, &shape[i][0]);
    }

    uint64_t size = 0;
    uint64_t output[100] = {0};
    list.GetTensorInfo(output, size);
    for (int i = 0; i < size/8; i++) {
        printf("output[%d] = %ld \n", i, output[i]);
    }
    EXPECT_EQ(size, ((dim + 1) * 10 + 1 + 10) * sizeof(uint64_t));
    EXPECT_EQ(output[0], ((dim + 1) * 10 + 1) * sizeof(uint64_t));
    for (int i = 1; i < output[0]/8; i += (dim + 1)) {
        EXPECT_EQ(output[i] & 0xffffffff, dim);
        EXPECT_EQ(output[i] >> 32, (i - 1)/(dim + 1));
        for(int j = 0; j < dim; j++) {
            EXPECT_EQ(output[i + j + 1], shape[(i - 1)/(dim + 1)][j]);
        }
    }

    for (int i = output[0]/8; i < size / 8; i++) {
        EXPECT_EQ((void *)output[i], &data[i - output[0]/8]);
    }
    ListTensorDesc listTensorDesc((__gm__ void *)output);

    uint64_t tmp[2] = {0};
    TensorDesc<int> desc;
    desc.SetShapeAddr(&tmp[0]);

    for (int i = 0; i < 10; i++) {
        listTensorDesc.GetDesc(desc, i);

        EXPECT_EQ(desc.GetDim(), dim);
        EXPECT_EQ(desc.GetIndex(), i);
        int totalLen = sizeof(int);
        for(int j = 0; j < dim; j++) {
            EXPECT_EQ(desc.GetShape(j), shape[i][j]);
            totalLen *= shape[i][j];
        }
        EXPECT_EQ(((uint64_t *)desc.GetDataPtr()), (uint64_t *)&data[i]);
        EXPECT_EQ(*(desc.GetDataPtr()), data[i]);
        GlobalTensor<int> descObj = desc.GetDataObj();
        EXPECT_EQ(descObj.GetPhyAddr(), &data[i]);
        EXPECT_EQ(descObj.GetSize(), totalLen);
    }
}

/* 传入单组shape, dim和数据指针，生成对应内存排布并解析 */
TEST_F(TEST_LIST_TENSOR, testListTensor0b)
{
    ListTensor list;
    int data[10] = {0,1,2,3,4,5,6,7,8,9};
    uint64_t shape[1][2] = {{2, 5}};
    uint32_t dim = 2;
    list.PushTensor((void *)&data[0], dim, &shape[0][0]);

    uint64_t size = 0;
    uint64_t output[100] = {0};
    list.GetTensorInfo(output, size);
    for (int i = 0; i < size / sizeof(uint64_t); i++) {
        printf("output[%d] = %ld \n", i, output[i]);
    }
    EXPECT_EQ(size, ((dim + 1) + 1 + 1) * sizeof(uint64_t));
    EXPECT_EQ(output[0], ((dim + 1) + 1) * sizeof(uint64_t));
    for (int i = 1; i < output[0] / sizeof(uint64_t); i += (dim + 1)) {
        EXPECT_EQ(output[i] & 0xffffffff, dim);
        EXPECT_EQ(output[i] >> 32, 1);
        for(int j = 0; j < dim; j++) {
            EXPECT_EQ(output[i + j + 1], shape[(i - 1)/(dim + 1)][j]);
        }
    }

    for (int i = output[0] / sizeof(uint64_t); i < size / sizeof(uint64_t); i++) {
        EXPECT_EQ((void *)output[i], &data[i - output[0]/8]);
    }
    ListTensorDesc listTensorDesc((__gm__ void *)output);
    EXPECT_EQ(listTensorDesc.GetSize(), 1);
    uint64_t tmp[2] = {0};
    TensorDesc<int> desc;
    desc.SetShapeAddr(&tmp[0]);

    listTensorDesc.GetDesc(desc, 0);

    EXPECT_EQ(desc.GetDim(), dim);
    EXPECT_EQ(desc.GetIndex(), 0);
    for(int j = 0; j < dim; j++) {
        EXPECT_EQ(desc.GetShape(j), shape[0][j]);
    }
    EXPECT_EQ(((uint64_t *)desc.GetDataPtr()), (uint64_t *)&data[0]);
}

/* 只传入数据指针，生成对应内存排布并获取相应索引的数据指针 */
TEST_F(TEST_LIST_TENSOR, testListTensor1a)
{
    ListTensor list;
    int data[10][2] = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4},
                       {5, 5}, {6, 6}, {7, 7}, {8, 8}, {9, 9}};
    for (int i = 0; i < 10; i++) {
        list.PushTensor((void *)&data[i]);
    }

    uint64_t size = 0;
    uint64_t output[100] = {0};
    list.GetTensorInfo(output, size);

    for (int i = 0; i < size/8; i++) {
        printf("output[%d] = %ld \n", i, output[i]);
    }
    ListTensorDesc listTensorDesc((__gm__ void *)output);
    size = listTensorDesc.GetSize();
    EXPECT_EQ(size, 10);
    for (int i = 0; i < size; i++) {
        auto ptr = listTensorDesc.GetDataPtr<int>(i);
        EXPECT_EQ((void *)ptr, (void *)&data[i]);
    }
}

/* 传入dim, shape和数据指针. 当dim为0或shape为-1时, 按照只传入数据指针处理 */
TEST_F(TEST_LIST_TENSOR, testListTensor1b)
{
    ListTensor list;
    int data[10][2] = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4},
                       {5, 5}, {6, 6}, {7, 7}, {8, 8}, {9, 9}};
    uint64_t shape[1][1] = {{0xffffffff}};
    for (int i = 0; i < 10; i++) {
        list.PushTensor((void *)&data[i], 0, &shape[0][0]);
    }

    uint64_t size = 0;
    uint64_t output[100] = {0};
    list.GetTensorInfo(output, size);

    for (int i = 0; i < size/8; i++) {
        printf("output[%d] = %ld \n", i, output[i]);
    }
    ListTensorDesc listTensorDesc((__gm__ void *)output);
    size = listTensorDesc.GetSize();
    EXPECT_EQ(size, 10);
    for (int i = 0; i < size; i++) {
        auto ptr = listTensorDesc.GetDataPtr<int>(i);
        EXPECT_EQ((void *)ptr, (void *)&data[i]);
    }
}

/* 传入dim, shape和数据指针. 只获取数据指针时, 调用ListTensorDesc中的GetDataPtr方法, 而非TensorDesc中的GetDataPtr */
TEST_F(TEST_LIST_TENSOR, testListTensor1c)
{
    ListTensor list;
    int data[10] = {0,1,2,3,4,5,6,7,8,9};
    uint64_t shape[10][2] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10},
                            {11, 12}, {13, 14}, {15, 16}, {17, 18}, {19, 20}};
    uint32_t dim = 2;
    for (int i = 0; i < 10; i++) {
        list.PushTensor((void *)&data[i], dim, &shape[i][0]);
    }

    uint64_t size = 0;
    uint64_t output[100] = {0};
    list.GetTensorInfo(output, size);
    for (int i = 0; i < size/8; i++) {
        printf("output[%d] = %ld \n", i, output[i]);
    }
    EXPECT_EQ(size, ((dim + 1) * 10 + 1 + 10) * sizeof(uint64_t));
    EXPECT_EQ(output[0], ((dim + 1) * 10 + 1) * sizeof(uint64_t));
    for (int i = 1; i < output[0]/8; i += (dim + 1)) {
        EXPECT_EQ(output[i] & 0xffffffff, dim);
        EXPECT_EQ(output[i] >> 32, (i - 1)/(dim + 1));
        for(int j = 0; j < dim; j++) {
            EXPECT_EQ(output[i + j + 1], shape[(i - 1)/(dim + 1)][j]);
        }
    }

    for (int i = output[0]/8; i < size / 8; i++) {
        EXPECT_EQ((void *)output[i], &data[i - output[0]/8]);
    }
    ListTensorDesc listTensorDesc((__gm__ void *)output);
    EXPECT_EQ(listTensorDesc.GetSize(), 10);
    for (int i = 0; i < listTensorDesc.GetSize(); i++) {
        auto ptr = listTensorDesc.GetDataPtr<int>(i);
        EXPECT_EQ((void *)ptr, (void *)&data[i]);
    }
}
