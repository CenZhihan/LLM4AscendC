/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ASCENDC_CODE_HIDDEN_H
#define ASCENDC_CODE_HIDDEN_H
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <experimental/filesystem>

namespace CodeHidden {

using namespace std;
namespace fs = std::experimental::filesystem;

constexpr int32_t TEMP_BUFFER_SIZE = 1024 * 1024;
constexpr uint64_t MAGIC_NUMBER = 0xfafbfcfdfefff1f2;
constexpr uint32_t VERSION = 1;
constexpr int32_t FAILED = 1;
constexpr int32_t SUCCESS = 0;

class AscendcCodeHidden {
public:
    AscendcCodeHidden(string srcDirPath, string resultFilePath)
        : srcDirPath_(srcDirPath), resultFilePath_(resultFilePath) {};
    ~AscendcCodeHidden();

    int32_t Init();
    int32_t Run();

private:
    int32_t WriteHeader();
    int32_t WriteIndex();
    int32_t WriteFileContent();
    int32_t GetAllFiles(string dirPath);
    bool IsLittleEndian();

    int32_t WriteFile(string filePath);
    template <typename T>
    int32_t WriteInt(T data);

private:
    bool isLittleEndian_ = true;
    string srcDirPath_;
    string resultFilePath_;
    ofstream resultOfs_;
    vector<string> srcFiles_;
    uint64_t totalPathLen = 0;
    char fileData_[TEMP_BUFFER_SIZE];
};

}  // namespace CodeHidden
#endif