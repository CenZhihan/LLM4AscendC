/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <google/protobuf/util/json_util.h>
#include <iostream>
#include <fstream>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/dynamic_message.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include "ge_ir.pb.h"

extern "C" {
    bool ReadModelFromText(const char *model_path, char **json_out, size_t *json_len) {
        std::cout << "Start to read model file: " << model_path << std::endl;
        // 检查文件是否存在
        std::ifstream fs(model_path, std::ifstream::in);
        if (!fs.is_open()) {
            std::cerr << "Failed to open file: " << model_path << std::endl;
            return false;
        }
        google::protobuf::io::IstreamInputStream input(&fs);
        google::protobuf::TextFormat::Parser model_parser;
        ge::proto::ModelDef message;
        try {
            if (!model_parser.Parse(&input, &message)) {
                std::cerr << "Failed to parse model from file: " << model_path << std::endl;
                fs.close();
                return false;
            }
        } catch (const std::exception &e) {
            std::cerr << "Exception occurred while parsing model: " << e.what() << std::endl;
            fs.close();
            return false;
        }
        fs.close();

        google::protobuf::util::JsonPrintOptions options;
        options.add_whitespace = true;
        options.always_print_primitive_fields = true;
        std::string json;
        try {
            if (!google::protobuf::util::MessageToJsonString(message, &json, options).ok()) {
                std::cerr << "Failed to convert message to JSON" << std::endl;
                return false;
            }
        } catch (const std::exception &e) {
            std::cerr << "Exception occurred while converting message to JSON: " << e.what() << std::endl;
            return false;
        }
        *json_out = new char[json.size() + 1];
        strcpy(*json_out, json.c_str());
        *json_len = json.size();

        return true;
    }

    void FreeJson(char *json) {
        delete[] json;
    }
}
