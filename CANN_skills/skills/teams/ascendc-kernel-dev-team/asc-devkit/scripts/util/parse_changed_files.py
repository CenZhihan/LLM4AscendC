#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import os
import sys


def file_filter(path):
    """
    Determine if the file contains files that do not require compilation.
    """
    key_words = ['.md', 'OWNERS', 'LICENSE', 'classify_rule.yaml', 'examples', 'docs']
    for key_word in key_words:
        if key_word in path:
            return True
    return False


def main():
    """"
    get change info from ci, CI_MODE=True, ci will write 'get diff >/change_file.txt'
    Return whether to skip compilation.
    """
    if len(sys.argv) < 2:
        return "FALSE"

    changed_files_arg = sys.argv[1]
    try:
        # 读取修改的文件列表
        if os.path.isfile(changed_files_arg) and changed_files_arg.endswith('.txt'):
            with open(changed_files_arg, 'r') as f:
                changed_files = [line.strip() for line in f.readlines() if line.strip()]
        elif ',' in changed_files_arg:
            changed_files = [f.strip() for f in changed_files_arg.split(',') if f.strip()]
        else:
            changed_files = [changed_files_arg.strip()]
    except BaseException as err:
        return "FALSE"
    finally:
        pass

    if not changed_files:
        return "FALSE"

    need_skip = True
    for file_path in changed_files:
        need_skip = need_skip and file_filter(file_path)
        if not need_skip:
            return "FALSE"

    return "TRUE"


if __name__ == '__main__':
    sys.stdout.write(main())
