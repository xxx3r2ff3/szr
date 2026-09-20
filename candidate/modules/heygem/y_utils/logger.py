# -*- coding: utf-8 -*-
# 出处:candidate/modules/heygem/y_utils/logger.py ← y_utils/logger.pyc
# 反汇编:evidence/modules/_heygem_orphans/disasm/y_utils_logger.pyc.dis(py3.8)
# 签名/常量按 .dis 对齐:create_logger(log_path='/home/y_demo/log/',
#   log_name='y_demo.log', level=logging.INFO)(MAKE_FUNCTION flags=1,defaults 3 元)
#   模块尾:config = config.get_config();logger = create_logger(...)
# 语义重建说明:函数体按 .dis 控制流忠实还原(basicConfig(level/format/datefmt)
#   → makedirs → WatchedFileHandler + Formatter → getLogger + addHandler → return)。
# 未还原细节:无。
"""
File: logger.py
Author: YuFangHui
Date: 2019/11/19
Description:
"""
import logging.handlers
import os
from y_utils import config


def create_logger(log_path='/home/y_demo/log/', log_name='y_demo.log', level=logging.INFO):
    logging_msg_format = '[%(asctime)s] [%(filename)s[line:%(lineno)d]] [%(levelname)s] [%(message)s]'
    logging_date_format = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=level, format=logging_msg_format, datefmt=logging_date_format)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_path = os.path.join(log_path, log_name)
    file_handler = logging.handlers.WatchedFileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(logging_msg_format))
    logger_h = logging.getLogger()
    logger_h.addHandler(file_handler)
    return logger_h


config = config.get_config()
logger = create_logger(config.get('log', 'log_dir'), config.get('log', 'log_file'), logging.INFO)
