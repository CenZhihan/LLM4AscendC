#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
import os
import inspect
FILE_PATH = os.path.dirname(os.path.realpath(__file__))

def debug(log_msg, *log_paras):
    line_no = inspect.currentframe().f_back.f_lineno
    funcname = inspect.currentframe().f_back.f_code.co_name
    co_filename = inspect.currentframe().f_back.f_code.co_filename
    filename = os.path.relpath(co_filename, FILE_PATH)
    log_str = '[Debug][%s:%d][%s] ' % (co_filename, line_no, funcname)
    log_all_msg = log_str + log_msg % log_paras
    print(log_all_msg)


def info(log_msg, *log_paras):
    line_no = inspect.currentframe().f_back.f_lineno
    funcname = inspect.currentframe().f_back.f_code.co_name
    co_filename = inspect.currentframe().f_back.f_code.co_filename
    filename = os.path.relpath(co_filename, FILE_PATH)
    log_str = '[Info][%s:%d][%s] ' % (co_filename, line_no, funcname)
    log_all_msg = log_str + log_msg % log_paras
    print(log_all_msg)


def warn(log_msg, *log_paras):
    line_no = inspect.currentframe().f_back.f_lineno
    funcname = inspect.currentframe().f_back.f_code.co_name
    co_filename = inspect.currentframe().f_back.f_code.co_filename
    filename = os.path.relpath(co_filename, FILE_PATH)
    log_str = '[Warning][%s:%d][%s] ' % (co_filename, line_no, funcname)
    log_all_msg = log_str + log_msg % log_paras
    print(log_all_msg)


def error(log_msg, *log_paras):
    line_no = inspect.currentframe().f_back.f_lineno
    funcname = inspect.currentframe().f_back.f_code.co_name
    co_filename = inspect.currentframe().f_back.f_code.co_filename
    filename = os.path.relpath(co_filename, FILE_PATH)
    log_str = '[Error][%s:%d][%s] ' % (co_filename, line_no, funcname)
    log_all_msg = log_str + log_msg % log_paras
    print(log_all_msg)

