# -*- coding: utf-8 -*-
# 出处:candidate/modules/heygem/h_utils/sweep_bot.py ← h_utils/sweep_bot.pyc
# 反汇编:evidence/modules/_heygem_orphans/disasm/h_utils_sweep_bot.pyc.dis(py3.8)
# 签名/常量按 .dis 对齐:def sweep(rubbish: list, flag=True)
#   (MAKE_FUNCTION flags=5 = DEFAULTS|ANNOTATIONS;defaults=(True,);annotations={'rubbish': list})
# 语义重建说明:函数体按 .dis 控制流忠实还原(flag 为假 → 打印关闭提示;
#   为真 → 遍历 rubbish,按 exists/isfile/isdir 分派清理,整体 except Exception 记日志)。
# 未还原细节:无。
"""
@project : ai_detection_server
@author  : huyi
@file   : sweep_bot.py
@ide    : PyCharm
@time   : 2021-12-08 12:02:29
"""
import os
import shutil

from y_utils.logger import logger


def sweep(rubbish: list, flag=True):
    if flag:
        try:
            for x in rubbish:
                if os.path.exists(x) is False:
                    logger.info('扫地机器人无法找到目标:[{}]'.format(x))
                    continue
                if os.path.isfile(x):
                    logger.info('扫地机器人清理文件:[{}]'.format(x))
                    os.remove(x)
                    continue
                if os.path.isdir(x):
                    logger.info('扫地机器人清理目录:[{}]'.format(x))
                    for filename in os.listdir(x):
                        file_path = os.path.join(x, filename)
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                            continue
                        if os.path.isdir(file_path):
                            shutil.rmtree(file_path)
        except Exception as e:
            logger.error('扫地机器人工作异常，异常信息:[{}]'.format(e.__str__()))
    else:
        logger.info('扫地机器人关闭，无法工作')
