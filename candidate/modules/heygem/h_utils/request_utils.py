# -*- coding: utf-8 -*-
# 出处:candidate/modules/heygem/h_utils/request_utils.py ← h_utils/request_utils.pyc
# 反汇编:evidence/modules/_heygem_orphans/disasm/h_utils_request_utils.pyc.dis(py3.8)
# 结构/签名/常量均按 .dis 逐字对齐:
#   request_post(url, param, timeout=10)  —— .dis MAKE_FUNCTION flags=1, defaults=(10,)
#   download(url, file_path)
#   download_file(file_url, local_path)
# 语义重建说明(按字节码控制流忠实还原,无臆造):
#   request_post 为 while True + try/except BaseException 的重试循环(fails 上限 3);
#   download 为分段 Range 续传(count 上限 10,chunk_size=65536,进度条 50 格);
#   download_file 为 with 双层上下文(chunk_size=4096,5M 打印一次进度)。
# 未还原细节:无(函数体均由 .dis 控制流完整还原)。
"""
@project : dhp-tools
@author  : huyi
@file   : request_utils.py
@ide    : PyCharm
@time   : 2021-09-03 16:00:32
"""
import json
import os
import sys
import time

import requests

from y_utils.logger import logger


def request_post(url, param, timeout=10):
    result = 0
    fails = 0
    while True:
        try:
            if fails >= 3:
                break
            headers = {'content-type': 'application/json'}
            ret = requests.post(url, json=param, headers=headers, timeout=timeout)
            if ret.status_code == 200:
                text = json.loads(ret.text)
                if text['code'] != 0:
                    fails += 1
                    logger.info('第[{}]次失败回调结果为:{}'.format(fails, text))
                    time.sleep(3)
                    continue
                else:
                    logger.info('成功回调结果为:{}'.format(text))
                    result = 1
                    break
            else:
                fails += 1
        except BaseException as ex:
            fails += 1
            logger.error('第[{}]次回调异常报错:{}'.format(fails, ex.__str__()))
            logger.info('网络连接出现问题, 正在尝试再次请求: ', fails)
    return result


def download(url, file_path):
    count = 0
    r1 = requests.get(url, stream=True, verify=False, timeout=(20, 20))
    total_size = int(r1.headers['Content-Length'])
    if os.path.exists(file_path):
        temp_size = os.path.getsize(file_path)
    else:
        temp_size = 0
    print(temp_size)
    print(total_size)
    r1.close()
    while count < 10:
        if count != 0:
            temp_size = os.path.getsize(file_path)
        if temp_size >= total_size:
            break
        count += 1
        logger.info('第[{}]次下载文件,已经下载数据大小:[{}],未下载数据大小:[{}]'.format(
            count, temp_size, total_size))
        headers = {'Range': f'bytes={temp_size}-{total_size}'}
        r = requests.get(url, stream=True, verify=False, headers=headers)
        with open(file_path, 'ab') as f:
            for chunk in r.iter_content(chunk_size=65536):
                if count != 1:
                    f.seek(temp_size)
                if chunk:
                    temp_size += len(chunk)
                    f.write(chunk)
                    f.flush()
                    done = int(50 * temp_size / total_size)
                    sys.stdout.write('\r[%s%s] %d%%' % (
                        '█' * done, ' ' * (50 - done), 100 * temp_size / total_size))
                    sys.stdout.flush()
        print('\n')
        r.close()
    return file_path


def download_file(file_url, local_path):
    chunk_size = 4096
    lst_size = 0
    total_size = 0
    with requests.get(file_url, stream=True, verify=False, timeout=1200) as r:
        with open(local_path, 'wb') as ff:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if total_size - lst_size > 5242880:
                    logger.info('downloading file %f M' % (total_size / 1048576))
                    lst_size = total_size
                total_size += len(chunk)
                ff.write(chunk)
    return local_path
