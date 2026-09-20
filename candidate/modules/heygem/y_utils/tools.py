# -*- coding: utf-8 -*-
# 出处:candidate/modules/heygem/y_utils/tools.py ← y_utils/tools.pyc
# 反汇编:evidence/modules/_heygem_orphans/disasm/y_utils_tools.pyc.dis(py3.8)
# 签名/常量按 .dis 对齐:check_dir(dir_path, flag=True)(MAKE_FUNCTION flags=1,defaults=(True,))
#   print_content(content) / read_file(path) / write_file(path, content)
# 语义重建说明:四个函数体均由 .dis 控制流逐条还原;文件头 docstring 取自
#   .dis 模块常量第 0 项(原文写的是 "File: service.py",逐字保留)。
# 未还原细节:无。
"""
File: service.py
Author: YuFangHui
Date: 2020-11-25
Description:
"""
import os
from os.path import exists


def check_dir(dir_path, flag=True):
    if exists(dir_path):
        return True
    if flag:
        os.makedirs(dir_path)
    return exists(dir_path)


def print_content(content):
    print('type:', type(content), '\n', content, '\n')


def read_file(path):
    f = open(path, 'r', encoding='utf-8')
    content = f.read()
    f.close()
    return content


def write_file(path, content):
    f = open(path, 'w', encoding='utf-8')
    f.write(content)
    f.close()
