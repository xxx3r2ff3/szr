# -*- coding: utf-8 -*-
"""
@project : dhp-tools
@author  : huyi
@file   : custom.py
@ide    : PyCharm
@time   : 2021-08-18 20:18:23

S003 恢复:h_utils/custom.pyc(py3.8)。
CustomError(Arg Count 2 = msg)与 __str__ 返回消息,反汇编清单一致。
"""


class CustomError(Exception):

    def __init__(self, msg):
        super().__init__(self)
        self.msg = msg

    def __str__(self):
        return self.msg
