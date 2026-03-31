/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <inttypes.h>
#include <algorithm>
#include "graph/tensor.h"
#include "debug/stub_util.h"
namespace {
const int64_t UNKNOWN_DIM_SIZE = -1;
}  // namespace

namespace ge {
// If not overflow return true
static bool Int64MulNotOverflow(const int64_t a, const int64_t b)
{
    if (a > 0) {
        if (b > 0) {
            if (a > (INT64_MAX / b)) {
                return false;
            }
        } else {
            if (b < (INT64_MIN / a)) {
                return false;
            }
        }
    } else {
        if (b > 0) {
            if (a < (INT64_MIN / b)) {
                return false;
            }
        } else {
            if ((a != 0) && (b < (INT64_MAX / a))) {
                return false;
            }
        }
    }
    return true;
}

class ShapeImpl {
public:
    ShapeImpl() = default;
    ~ShapeImpl() = default;
    explicit ShapeImpl(const std::vector<int64_t> &dims)
    {
        bool isUnknownDimNum = false;
        for (const auto &dim : dims) {
            if (dim == UNKNOWN_DIM_NUM) {
                isUnknownDimNum = true;
                break;
            }
        }
        dims_ = isUnknownDimNum ? std::vector<int64_t>({UNKNOWN_DIM_NUM}) : dims;
    }

private:
    std::vector<int64_t> dims_;
    friend class Shape;
};

Shape::Shape()
{
    impl_ = ComGraphMakeShared<ShapeImpl>();
}

Shape::Shape(const std::vector<int64_t> &dims)
{
    impl_ = ComGraphMakeShared<ShapeImpl>(dims);
}

size_t Shape::GetDimNum() const
{
    if (impl_ != nullptr) {
        const bool isDimUnknown = std::any_of(std::begin(impl_->dims_), std::end(impl_->dims_),
            [](const int64_t i) { return i == UNKNOWN_DIM_NUM; });
        if (isDimUnknown) {
            GELOGI("Dim num is unknown, return 0U.");
            return 0U;
        }
        return impl_->dims_.size();
    }
    return 0U;
}

int64_t Shape::GetDim(size_t idx) const
{
    if (impl_ != nullptr) {
        if (idx >= impl_->dims_.size()) {
            return 0;
        }
        return impl_->dims_[idx];
    }
    return 0;
}

std::vector<int64_t> Shape::GetDims() const
{
    const std::vector<int64_t> dims;
    if (impl_ != nullptr) {
        return impl_->dims_;
    }
    return dims;
}

int64_t Shape::GetShapeSize() const
{
    if (impl_ != nullptr) {
        if (impl_->dims_.empty()) {
            return 0;
        }
        int64_t size = 1;
        for (const auto i : impl_->dims_) {
            if ((i == UNKNOWN_DIM_NUM) || (i == UNKNOWN_DIM)) {
                return UNKNOWN_DIM_SIZE;
            }

            if (!Int64MulNotOverflow(size, i)) {
                GELOGE(GRAPH_FAILED, "[Check][Overflow] mul overflow: %" PRId64 ", %" PRId64, size, i);
                size = 0;
                return size;
            }
            size *= i;
        }
        return size;
    }
    return 0;
}

}