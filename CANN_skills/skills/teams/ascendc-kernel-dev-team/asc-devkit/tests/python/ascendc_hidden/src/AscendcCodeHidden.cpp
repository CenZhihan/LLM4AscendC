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

#include "AscendcCodeHidden.h"

namespace CodeHidden {

AscendcCodeHidden::~AscendcCodeHidden() {
    if (resultOfs_.is_open()) {
        resultOfs_.close();
    }
}

int32_t AscendcCodeHidden::Init() {
    if (srcDirPath_.empty()) {
        cout << "[ERROR](Init) srcDirPath_ is empty." << endl;
        return FAILED;
    }
    if (srcDirPath_.back() == '/') {
        srcDirPath_.pop_back();
    }
    resultOfs_.open(resultFilePath_, ios::trunc | ios::binary);
    if (resultOfs_.is_open() == false) {
        cout << "[ERROR](Init) Open file [" << resultFilePath_ << "] failed." << endl;
        return FAILED;
    }
    srcFiles_.clear();
    if (GetAllFiles(srcDirPath_) != SUCCESS) {
        cout << "[ERROR](GetAllFiles) failed." << endl;
        return FAILED;
    };

    isLittleEndian_ = IsLittleEndian();
    cout << "[INFO](Init) Is little endian: " << (isLittleEndian_ ? "true" : "false") << "." << endl;
    return SUCCESS;
}

int32_t AscendcCodeHidden::Run() {
    if (WriteHeader() != SUCCESS) {
        cout << "[ERROR](Run) WriteHeader failed." << endl;
        return FAILED;
    };

    if (WriteIndex() != SUCCESS) {
        cout << "[ERROR](Run) WriteIndex failed." << endl;
        return FAILED;
    };

    if (WriteFileContent() != SUCCESS) {
        cout << "[ERROR](Run) WriteFileContent failed." << endl;
        return FAILED;
    };

    return SUCCESS;
}

int32_t AscendcCodeHidden::WriteHeader() {
    uint64_t magicNumber = MAGIC_NUMBER;
    if (WriteInt(magicNumber) != SUCCESS) {
        cout << "[ERROR](WriteHeader) Write magicNumber failed." << endl;
        return FAILED;
    }
    uint32_t version = VERSION;
    if (WriteInt(version) != SUCCESS) {
        cout << "[ERROR](WriteHeader) Write version failed." << endl;
        return FAILED;
    }
    uint32_t indexRegionLen = totalPathLen + srcFiles_.size() * 3 * sizeof(uint32_t);
    if (WriteInt(indexRegionLen) != SUCCESS) {
        cout << "[ERROR](WriteHeader) Write indexRegionLen failed." << endl;
        return FAILED;
    }
    return SUCCESS;
}

int32_t AscendcCodeHidden::WriteIndex() {
    uint32_t fileOffset = 0;
    for (size_t i = 0; i < srcFiles_.size(); i++) {
        if (WriteInt(fileOffset) != SUCCESS) {
            cout << "[ERROR](WriteIndex) Write fileOffset of the file[" << srcDirPath_ + srcFiles_[i] << "] failed."
                 << endl;
            return FAILED;
        }
        uint32_t fileLen = fs::file_size(srcDirPath_ + srcFiles_[i]);
        if (WriteInt(fileLen) != SUCCESS) {
            cout << "[ERROR](WriteIndex) Write fileLen of the file[" << srcDirPath_ + srcFiles_[i] << "] failed."
                 << endl;
            return FAILED;
        }
        uint32_t filePathLen = srcFiles_[i].length();
        if (WriteInt(filePathLen) != SUCCESS) {
            cout << "[ERROR](WriteIndex) Write filePathLen of the file[" << srcDirPath_ + srcFiles_[i] << "] failed."
                 << endl;
            return FAILED;
        }
        try {
            resultOfs_.write(srcFiles_[i].c_str(), filePathLen);
            fileOffset = fileOffset + fileLen + 1;
        } catch (fs::filesystem_error& e) {
            cout << "[ERROR](WriteIndex) Catch error, " << e.what() << endl;
            return FAILED;
        }
    }
    return SUCCESS;
}

int32_t AscendcCodeHidden::WriteFileContent() {
    for (size_t i = 0; i < srcFiles_.size(); i++) {
        if (WriteFile(srcDirPath_ + srcFiles_[i]) != SUCCESS) {
            cout << "[ERROR](WriteFileContent) Write file content of the file[" << srcDirPath_ + srcFiles_[i]
                 << "] failed." << endl;
            return FAILED;
        }
    }
    return SUCCESS;
}

int32_t AscendcCodeHidden::GetAllFiles(string dirPath) {
    try {
        if (fs::exists(dirPath) == false) {
            cout << "[ERROR](GetAllFiles) dirPath[" << dirPath << "] not exists." << endl;
            return FAILED;
        }
        if (fs::is_directory(dirPath) == false) {
            cout << "[ERROR](GetAllFiles) dirPath[" << dirPath << "] not a directory." << endl;
            return FAILED;
        }
        for (const auto& entry : fs::directory_iterator(dirPath)) {
            if (fs::is_regular_file(entry.status())) {
                string tempPath = entry.path();
                tempPath = tempPath.substr(srcDirPath_.length());
                srcFiles_.push_back(tempPath);
                totalPathLen += tempPath.length();
            } else if (fs::is_directory(entry.status())) {
                if (GetAllFiles(entry.path()) != SUCCESS) {
                    return FAILED;
                };
            }
        }
    } catch (fs::filesystem_error& e) {
        cout << "[ERROR](GetAllFiles) Catch error, " << e.what() << endl;
        return FAILED;
    }
    return SUCCESS;
}

bool AscendcCodeHidden::IsLittleEndian() {
    uint32_t i = 0x12345678;
    uint8_t* c = reinterpret_cast<uint8_t*>(&i);
    return *c == 0x78;
}

int32_t AscendcCodeHidden::WriteFile(string filePath) {
    ifstream srcFile(filePath, std::ios::in | std::ios::binary);

    if (srcFile.is_open() == false) {
        cout << "[ERROR](WriteFile) Open file[" << filePath << "] failed." << endl;
        return FAILED;
    }
    try {
        while (!srcFile.eof()) {
            srcFile.read(fileData_, TEMP_BUFFER_SIZE);
            size_t numRead = srcFile.gcount();
            resultOfs_.write(fileData_, numRead);
        }
        fileData_[0] = '\0';
        resultOfs_.write(fileData_, 1);
    } catch (fs::filesystem_error& e) {
        cout << "[ERROR](WriteFile) Catch error, " << e.what() << endl;
        return FAILED;
    }
    return SUCCESS;
}

template <typename T>
int32_t AscendcCodeHidden::WriteInt(T data) {
    try {
        if (isLittleEndian_) {
            resultOfs_.write(reinterpret_cast<char*>(&data), sizeof(T));
        } else {
            T tempData = 0;
            char* littlePtr = reinterpret_cast<char*>(&tempData);
            char* bigPtr = reinterpret_cast<char*>(&data) + sizeof(T);
            for (size_t i = 0; i < sizeof(T); i++) {
                *(littlePtr + i) = *(bigPtr - 1 - i);
            }
            resultOfs_.write(reinterpret_cast<char*>(&tempData), sizeof(T));
        }
    } catch (fs::filesystem_error& e) {
        cout << "[ERROR](WriteInt) Catch error, " << e.what() << endl;
        return FAILED;
    }
    return SUCCESS;
}

}  // namespace CodeHidden