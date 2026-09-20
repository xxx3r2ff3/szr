# -*- coding: utf-8 -*-
"""update —— R005 静态重建(源 update.cp310-win_amd64.pyd,T4R,私有集)。

证据标签(见 szr2026_out/diff_harness/REBUILD_WORKFLOW.md §0):
  [pyd 0x…]   Ghidra 反编译函数入口(evidence/modules/update/static/decomp/out/)
  [pyd py=N]  反编译中的 Cython 错误路径 py 行号(按行切片见 slices_*.txt)
  [槽 0x…]    StringTab 槽位(constants/stringtab_dump.json)
  [E1 eN]     VM 内 oracle 真实行为探针(runtime/probe_update_eN*.py + _out.json)

要点(全部有二进制/实测出处):
* 模块级 `Config` 类与 modules.core.config 的同名类同形:__init__(self, file) 做
  存在性门控后 configparser.read(file, encoding='utf-8-sig');get_config(section,
  option, defaultVal=None) 未命中且无默认值时原样抛 [pyd 0x180001280/0x180001cf0]。与
  candidate/modules/core/config.py 的 E1 结论一致(R002 冻结基线复核)。
* 4 个模块级函数名与 varnames 均取自字符串表 [槽 abs_path1/abs_path2/decoded_bytes/
  decoded_str/block_size/…];`block_size` 是编译期 int 常量 0x2000=8192
  [pyd 0x180012240: DAT_18001f348 = PyLong_FromLong(0x2000)],故**不是**模块全局名
  (oracle dir() 无 block_size,E1 e7)。
* decode_base64 用 `base64.b64decode(s, validate=True)` + `.decode('utf-8')`,
  失败(ValueError/binascii.Error/UnicodeDecodeError 族)返回原对象 [E1 e7:
  'a-b_c=' 与 'hello' 原样返回,'aGVsbG8=' → 'hello';b'…' 也可]。
* download_file:requests.get(url, stream=True) → int(headers.get('content-length',0))
  → with open(dist,'wb') + with tqdm(total=…, unit='iB', unit_scale=True,
  unit_divisor=1024) → for data in response.iter_content(8192): bar.update(len(data));
  f.write(data);**返回 dist**(不做 status_code 判定)[E1 e7]。
* get_update_md5_files:print('开始检测') → GET baseurl+'/md5_files.txt';200 时按 '\n'
  切行、strip 后按 ',' 拆 (file_name, file_md5),本地文件存在且 md5 相同则跳过,否则
  收集 `(原始文件名, baseurl + 文件名.replace('\\','/').replace('./','/'))`;非 200 打印
  '请求失败，状态码：'+str(status) 后返回空表 [E1 e7 十一种响应矩阵]。
* update():见函数内注释(py 170-250 全流程),[E1 e5/e7/e8] 给出校验分支矩阵。
"""
import argparse
import base64
import configparser
import hashlib
import os
import sys
from datetime import datetime

import requests
from tqdm import tqdm


class Config:
    """[pyd 0x180001280 = update.Config.__init__,0x180001cf0 = Config.get_config]"""

    def __init__(self, file):
        # strings 顺序:os.path.exists → configparser.ConfigParser → read(encoding=
        # 'utf-8-sig');门控失败即 `Exception(file + ' not found')`(PyUnicode_Concat)。
        if not os.path.exists(file):
            raise Exception(file + ' not found')
        self.config = configparser.ConfigParser()
        self.config.read(file, encoding='utf-8-sig')

    def get_config(self, section, option, defaultVal=None):
        # [pyd py≈27..28] self.config.get(section, option) 两参;异常时
        # `defaultVal != None`(RichCompare op 3)为真则返回默认,否则原样重抛。
        try:
            return self.config.get(section, option)
        except Exception:
            if defaultVal is None:
                raise
            return defaultVal


def print_yellow(message):
    # [pyd 0x1800045d0] f-string 三段拼接 + print();
    # 槽 0x18001eed0='\x1b[93m'、0x18001ee18='\x1b[0m'。
    print(f'\033[93m{message}\033[0m')


def print_yellow_tip(message, title='温馨提示'):
    # [pyd 0x180004ac0] 三次 print_yellow:f'--- {title} ---' / message / '---------------'
    # 默认值取自缓存常量 ('温馨提示',) [pyd 0x1800114e0]。
    print_yellow(f'--- {title} ---')
    print_yellow(message)
    print_yellow('---------------')


def decode_base64(s):
    # [pyd 0x180005140] b64decode(validate=True) → decode('utf-8');
    # except (base64.binascii.Error, UnicodeDecodeError): return s
    # (尾部 `*param_2 += 1; return param_2` 即"返回原参数")。
    try:
        decoded_bytes = base64.b64decode(s, validate=True)
        decoded_str = decoded_bytes.decode('utf-8')
        return decoded_str
    except (base64.binascii.Error, UnicodeDecodeError):
        return s


def get_md5(file_path):
    # [pyd 0x180002350] with open(file_path,'rb') as f: md5=hashlib.md5();
    # 循环 f.read(block_size=8192) → update;末尾 close 后 f.hexdigest()。
    with open(file_path, 'rb') as f:
        md5_val = hashlib.md5()
        data = f.read(8192)
        while data:
            md5_val.update(data)
            data = f.read(8192)
    return md5_val.hexdigest()


def _update_source_url(config):
    """[COM-D002] 更新源根地址:显式配置 + fail-closed(不内置任何主机)。

    取值顺序:环境变量 ``SZR_UPDATE_SOURCE_URL`` → ``config.ini [Update]
    source_url`` → 空。三者皆空即**硬失败**(打印明确原因后 sys.exit(2)),
    绝不回落到任何原厂/第三方更新主机。返回值去掉尾部 '/' 以保持原实现的
    字符串拼接形态(baseurl + file_name 无尾斜杠)。
    """
    url = (os.environ.get('SZR_UPDATE_SOURCE_URL') or '').strip()
    if not url:
        try:
            url = (config.get_config('Update', 'source_url', '') or '').strip()
        except Exception:                      # noqa: BLE001 - 配置缺失按未配置处理
            url = ''
    url = url.rstrip('/')
    if not url:
        print_yellow_tip(
            '更新源未配置：请在 config.ini 写入 [Update] source_url（或设置环境变量 '
            'SZR_UPDATE_SOURCE_URL）。本客户端不内置任何原厂/第三方更新主机。')
        sys.exit(2)
    return url


def are_paths_same(path1, path2):
    # [pyd 0x180005c60] normcase(abspath(p)).lower() 各一次,RichCompare op 2 (==)。
    abs_path1 = os.path.normcase(os.path.abspath(path1)).lower()
    abs_path2 = os.path.normcase(os.path.abspath(path2)).lower()
    return abs_path1 == abs_path2


def download_file(url, dist):
    # [pyd 0x180003100;缓存常量 ('content-length',0) = 0x1800114e0 DAT_18001f508]
    # [I001 E2E 定谳 i001-fix-3] 整机证据(第六轮 U5):下载中断残件恰为响应体
    # 前 1024 字节(sha 定谳)→ pyd 以 **1024 字节分块**写盘(R005 按 decompile
    # 推定的 8192 属 get_md5 的 block_size,此处被证伪)。
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(dist, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
            for data in response.iter_content(1024):
                bar.update(len(data))
                f.write(data)
    return dist


def get_update_md5_files(baseurl, testurl=None, devurl=None):
    # [pyd 0x180006ae0;切片 slices_get_update_md5_files.txt py=91..164]
    # 来源解析:先打印 '开始检测'(缓存常量 DAT_18001f528),再 GET <url>+'/md5_files.txt'。
    # [I001 E2E 定谳 i001-fix-2] 整机证据(i001_bootstrap.update_chain 首轮+次轮,
    # evidence/integration/i001_obs/):pyd 实际只采 baseurl 与 testurl 两根清单
    # (无 devurl 第三请求;devurl 仅在 --dev 时作为 baseurl 进入采集),且同名的
    # 待下载项按"后采根覆盖先采根"合并 —— 观测到唯一下载请求 URL 绑定第二根
    # ('.../human-infer2-test' + 相对路径,原清单同名项来自第一根也未发出)。
    print('开始检测')
    merged = {}

    def collect(root):
        response = requests.get(root + '/md5_files.txt')
        if response.status_code != 200:
            print('请求失败，状态码：' + str(response.status_code))
            return
        lines = response.text.split('\n')
        for line in tqdm(lines):
            line = line.strip()
            if not line:
                continue
            file_name, file_md5 = line.split(',')
            file_url = root + file_name.replace('\\', '/').replace('./', '/')
            if os.path.exists(file_name) and get_md5(file_name) == file_md5:
                continue
            merged[file_name] = file_url                       # i001-fix-2:后根覆盖

    collect(baseurl)
    if testurl:
        collect(testurl)
    return list(merged.items())


def update():
    # [pyd 0x18000b630;切片 slices_update_body.txt py=170..255]
    config = Config('config.ini')                                    # py170
    api_host = config.get_config('Server', 'host')                   # py170(缓存常量 ('Server','host'))
    api_host = decode_base64(api_host)                               # py171
    api_host = api_host.replace('https://', '')                      # py172
    # [COM-D002 语义替换] 原实现的更新源根地址是硬编码的原厂 AList 主机
    # ('http://<原厂更新主机>:5244/d/www/update')。原厂主机字面量已全部删除:
    # 更新源改由唯一的显式配置给出(config.ini [Update] source_url,或环境变量
    # SZR_UPDATE_SOURCE_URL),默认空串 = fail-closed;未配置时**硬失败**并给出
    # 明确错误,绝不回落到任何内置/原厂/第三方主机(见 _update_source_url)。
    update_source = _update_source_url(config)                       # COM-D002
    checkurl = (update_source + '/human-update/'
                + api_host + '.txt')                                 # py174
    try:
        response = requests.get(checkurl)
        texts = response.text.split(',')                             # py181
    except Exception:
        print_yellow_tip('网络错误，无法更新')                        # py177/178
        exit()
    validtime = texts[0]                                             # py182
    # py184/185:售后期判定;py187/188:可更新性判定(E1 e8 矩阵:
    # '20200101' → 两条提示;'abc'/'' → 仅"无法更新";未来日期 → 均不打印)。
    try:
        expired = int(datetime.now().strftime('%Y%m%d')) > int(validtime)
    except ValueError:
        expired = None                                               # 非法日期串:不可判定
    if expired is True:
        print_yellow_tip('当前系统售后已到期，无法更新')
    if expired is not False:
        print_yellow_tip('当前系统无法更新')
        exit()
    parser = argparse.ArgumentParser(description='Train gui')         # py191/192/193
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--dev', action='store_true', default=False)
    args, _unknown = parser.parse_known_args()                       # py194
    baseurl = update_source + '/human-infer2'
    # [I001 E2E 定谳 i001-fix-2] 整机证据:第二清单根 = '.../human-infer2-test'
    # (无尾斜杠,URL 拼接出 'human-infer2-testsub/...' 即其原始形态),
    # 并非 R005 推定的 '.../human-update/test/'(该根在整机下从未被请求)。
    testurl = update_source + '/human-infer2-test'
    devurl = update_source + '/human-infer2-dev'
    if args.dev:
        baseurl = devurl                                             # py218
    elif args.test:
        baseurl = update_source + '/human-infer2-test'               # py213
    # [I001 E2E 定谳 i001-fix-2b] 整机证据(第四轮):通过售后门后、清单采集前,
    # pyd 还会请求一次 'human-update/test/<host>.txt'(U2/U3 过期/非法日期变体
    # 从不发出该请求;stub 对两路同文,其响应的用途不可分辨,按原样补齐请求面)。
    try:
        requests.get(update_source + '/human-update/test/'
                     + api_host + '.txt')
        downloads = get_update_md5_files(baseurl, testurl, devurl)   # py223
    except Exception:
        print_yellow_tip('网络错误，无法更新')                        # py224
        exit()
    if downloads:
        print('开始下载')                                             # py227(缓存常量 DAT_18001f6e0)
        # [I001 E2E 定谳 i001-fix-2c] 整机证据(第二/六轮):凡到达下载段,
        # update_tmp 即存在(pip 门触发)且**在 pip 之后被清除(无论成败)**
        # (第六轮 U4/U5 终态 dirs 均只剩原路径)——它是下载段的标记目录,
        # 而非下载中转(直写证据见下)。
        os.makedirs('update_tmp', exist_ok=True)
        for index, file in enumerate(downloads):                     # py228
            # [I001 E2E 定谳 i001-fix-2/2c] 整机证据(i001_bootstrap.update_chain 六轮):
            #   * 逐文件先打印 '[i/n] <相对路径>';下载异常 → 'update fail <相对路径>'
            #     后继续(不外溢);
            #   * 下载**直写最终相对路径**(第六轮 U5:中断残件=前 1024 字节,
            #     就在最终路径上;对仅 Python open 持有的目标也能覆盖,第四轮 U6);
            #   * 目标被写锁(典型 = 运行中的 pyd,Windows 对运行中 DLL 禁写)→
            #     落 'new_<名>' 交 loader(load) 下次启动换件(R001);oracle 根现存
            #     new_update.cp310-win_amd64.pyd 即该机制的现实痕迹。此分支在本
            #     E2E 不可触达(独占锁夹具会提前干扰清单期 md5 读取,属夹具伪像),
            #     按 R005 切片 py235..243 + 现实痕迹保留。
            print('[%d/%d] %s' % (index + 1, len(downloads), file[0]))
            try:
                os.makedirs(os.path.dirname(file[0]) or '.', exist_ok=True)
                download_file(file[1], file[0])
            except PermissionError:
                if are_paths_same(file[0], os.path.dirname(__file__)):   # py237/238
                    continue
                download_file(file[1], 'new_' + str(os.path.basename(file[0])))
            except Exception:
                print('update fail ' + file[0])
                continue
        print('下载完成')                                             # py231 后(缓存常量 DAT_18001f530)
    if os.path.exists('update_tmp'):
        os.system('runtime\\Scripts\\pip.exe config set global.index-url '
                  'https://mirrors.nju.edu.cn/pypi/web/simple/')     # py246
        os.system('runtime\\Scripts\\pip.exe install -r requirements.txt')  # py249
        # [I001 E2E 定谳 i001-fix-2c] pip 环节之后 update_tmp 一律清除
        # (第六轮 U4/U5 终态均无 update_tmp,含存在失败残件的 U5)。
        import shutil
        shutil.rmtree('update_tmp', ignore_errors=True)


# [pyd 0x180012240 模块级收尾] 反编译比较链:__name__ 先与 '__main__'、再与 'update'、
# 再与 'update_tmp' 比较,命中后调用 update();实测以 'update' 名导入即触发主流程
# (E1 e4/e6:import update → 打印提示 + SystemExit(None))。
if __name__ == '__main__' or __name__ in ('update', 'update_tmp'):
    update()
