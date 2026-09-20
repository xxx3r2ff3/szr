# -*- coding: utf-8 -*-
"""util —— R009 全量重建(自 modules/core/util.cp310-win_amd64.pyd,冻结基线 v6.18.0,
`source_sha256=31b6c77b23248fc405c7bde4316b8028c6768a0a2e2ddd08d16f1eee29422910`)。

证据体系(详见 reports/modules/core__util-impl.md):
  * E2 静态:Ghidra 12.1.3 无头反编译(evidence/modules/core__util/static/pseudocode/,
    逐函数 `core__util__FUN_<va>__<va>.c`;`__Pyx_StringTabEntry` 槽位解析见
    static/stringtab.json,行内 `STR("…")` 即槽位字面量);
  * E2 数值常量:FUN_18002a030(0/1/2/5/10/16/32/128/200/201/256/1023/1024/65535/65536);
  * E3 调用方:app_mqtt.py:50/56 从本模块 import;
  * E1 动态:VM 内 oracle 解释器三支探针
    (runtime/probe1_out.json / probe2_out.json / probe3_out.json),
    逐点输入/输出/异常/stdout 均为实测。

模块级结构(E2 `__pyx_pymod_exec_util`,VA 0x18002a540 补建反编译):
base64/ctypes(wintypes)/json/os/platform/random/shutil/socket/hashlib(md5)/
tempfile/threading/time/sys/uuid(uuid4)/psutil/requests/tqdm/urllib/
modules.core.config(api_host, api_key, cache_dir)/winreg/win32pipe/win32file;
`system = platform.system()`(py=23);全部函数/类定义 py 行号见报告对照表。
导入期副作用:经 modules.core.config 打印 5 行横幅(E1 实测,本模块无额外输出)。
"""
import base64
import ctypes
from ctypes import wintypes
from hashlib import md5
import json
import os
import platform
import random
import shutil
import socket
import sys
import tempfile
import threading
import time
from uuid import uuid4
import urllib

import psutil
import requests
from tqdm import tqdm

from modules.core.config import api_host, api_key, cache_dir

import win32file
import win32pipe
import winreg

system = platform.system()


# ---------------------------------------------------------------- 设备标识
def get_windows_computer_id():
    """机器唯一标识。[pyd FUN_180001090 @0x180001090,pyd py=37..59]

    Windows:注册表 HKCU\\Environment\\SZR_ID(REG_SZ),缺失时 uuid4 生成并写回;
    非 Windows:HOME/.szr_id 文本文件。E1:连调三次恒等;注册表实测
    {value: 8c4d6cbe-9261-46c5-b16f-c766e9ca6def, kind: REG_SZ}(probe3)。
    """
    if system == 'Windows':
        try:
            # E2(py=40):读路径 OpenKey 只用默认访问权(KEY_READ);E1 反证:若这里
            # 传 KEY_WRITE,QueryValueEx 会因缺 KEY_QUERY_VALUE 失败 →
            # 每次调用都新生成 uuid 并写回注册表(2026-09-10 首轮捕获实测)。
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment')
            deviceUid, _ = winreg.QueryValueEx(key, 'SZR_ID')
        except Exception:
            deviceUid = str(uuid4())
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment', 0,
                                 winreg.KEY_WRITE)
            winreg.SetValueEx(key, 'SZR_ID', 0, winreg.REG_SZ, deviceUid)
        winreg.CloseKey(key)
        return deviceUid
    szr_file = os.path.join(os.environ.get('HOME'), '.szr_id')
    if os.path.exists(szr_file):
        with open(szr_file, 'r', encoding='utf-8') as f:
            deviceUid = f.read()
    else:
        deviceUid = str(uuid4())
        with open(szr_file, 'w', encoding='utf-8') as f:
            f.write(deviceUid)
    return deviceUid


def verify(version):
    """启动授权校验。[pyd FUN_180003e40 @0x180003e40,pyd py=64..87]

    E1(probe1/2/3):
      * 先 print('local device id ' + deviceUid);
      * POST {api_host}/api/verifylocal,headers={'Content-Type':'application/json',
        'api_key':api_key},data=json.dumps({'version':…,'deviceUid':…}),verify=False;
      * status_code != 200 → print(response.text) + time.sleep(10) + exit()(py=76..79,
        probe3 verify_403:traceback util.py:79 SystemExit,stdout 为响应原文);
      * 200 → 本地摘要 h1=md5(('hq,'+version+','+api_host+','+deviceUid).encode()).hexdigest(),
        h2=md5(h1.encode('utf-8')).hexdigest();h2 != response.text 则
        print('无权限启动') + exit()(py=81..86)。h2 公式由 probe4 以"比较间谍"定谳:
        verification(v1.2.3)=0034fc4690e0edaac30de155bd511031。
    """
    deviceUid = get_windows_computer_id()
    print('local device id ' + deviceUid)
    headers = {'Content-Type': 'application/json'}
    headers['api_key'] = api_key
    data = {'version': version, 'deviceUid': deviceUid}
    response = requests.post(
        api_host + '/api/verifylocal',
        data=json.dumps(data),
        headers=headers,
        verify=False,
    )
    if response.status_code != 200:
        print(response.text)
        time.sleep(10)
        sys.exit()  # [I001 E2E 定谳 i001-D1] oracle 为 sys.exit()(裸 exit() 会多出 SystemExit(None) 参数,auth_client 实测)
    local_hash = md5(('hq,' + version + ',' + api_host + ',' + deviceUid).encode()).hexdigest()
    local_hash = md5(local_hash.encode('utf-8')).hexdigest()
    if local_hash != response.text:
        print('无权限启动')
        sys.exit()  # [I001 E2E 定谳 i001-D1] 同上
    return deviceUid


# ---------------------------------------------------------------- 编解码工具
class Base64Util:
    """URL 友好的无填充 base64。[pyd py=91..99]"""

    @staticmethod
    def encode(text):
        # E1:encode('hello')='aGVsbG8'(无 '=' 填充);'中文'='5Lit5paH'。
        return base64.b64encode(text.encode()).decode().rstrip('=')

    @staticmethod
    def decode(text):
        # E1:decode('aGVsbG8=')='hello';decode('!!!not-base64!!!') 原样返回输入。
        try:
            return base64.b64decode(text + '=' * (-len(text) % 4)).decode()
        except Exception:
            return text


class NamedPipe:
    """跨平台命名管道封装。[pyd py=106..158;E1 probe1 __init__ 实测属性]"""

    def __init__(self, name):
        self.name = name
        self.sys = platform.system()
        if self.sys == 'Windows':
            self.pipe_path = '\\\\.\\pipe\\' + name
            self.pipe = win32pipe.CreateNamedPipe(
                self.pipe_path,
                win32pipe.PIPE_ACCESS_DUPLEX,
                win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
                1, 65536, 65536, 0, None)
        else:
            print('Currently not supported for ' + self.sys)
            self.pipe_path = '/tmp/' + name

    def open(self):
        if self.sys == 'Windows':
            win32pipe.ConnectNamedPipe(self.pipe, None)
        else:
            self.pipe = open(self.pipe_path, os.O_WRONLY)

    def write(self, data):
        if self.sys == 'Windows':
            win32file.WriteFile(self.pipe, data)
        else:
            self.pipe.write(data)

    def close(self):
        if self.sys == 'Windows':
            win32file.CloseHandle(self.pipe)
        else:
            self.pipe.close()
            if os.path.exists(self.pipe_path):
                os.unlink(self.pipe_path)


class StoppableThread(threading.Thread):
    """可停止线程基类。[pyd py=161..174;E1 __init__ 不收参数]"""

    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()
        self._stop_event.clear()

    def run(self):
        while not self._stop_event.is_set():
            time.sleep(0.1)

    def stop(self):
        self._stop_event.set()


class RepeatedTimer:
    """周期定时器。[pyd py=176..196;E1:start 后 0.3s 间隔触发,stop 后不再触发]"""

    def __init__(self, interval, function, *args, **kwargs):
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self.function(*self.args, **self.kwargs)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()


# ---------------------------------------------------------------- 定时器
def setTimeout(func, sec):
    """sec 秒后执行一次 func,返回已启动的 Timer。[pyd FUN_18000c960,pyd py=198..202]

    E1(probe1/2/5):返回对象 type=Timer;到期触发一次后线程回收;daemon 继承创建
    线程(主线程创建 → False)。
    """
    t = threading.Timer(sec, func)
    t.start()
    return t


def setInterval(func, sec):
    """每 sec 秒执行 func,返回首个 Timer。[pyd FUN_18000d2c0,pyd py=204..213]

    E1(probe1/2/5):返回 type=Timer;0.1s 间隔 0.35s 内触发 3 次;对返回对象
    cancel() **不能**停止后续触发(每次触发在闭包 func_wrapper 里新建下一个
    Timer),结束时活动线程数=2(主线程 + 待触发 Timer)。daemon 取自创建线程
    (主线程创建 → False;probe1 的 True 是探针自身在守护线程里调用的假象)。
    """
    def func_wrapper():
        func()
        setInterval(func, sec)

    t = threading.Timer(sec, func_wrapper)
    t.start()
    return t


# ---------------------------------------------------------------- 文件工具
def remove_dir(filepath):
    """清空并重建目录(不是删除!)。[pyd FUN_18000d880,pyd py=215..221]

    E1(probe1/2):已存在目录 → rmtree + mkdir(调用后目录存在且为空);
    不存在 → 直接 mkdir;路径是文件 → shutil.rmtree 抛 NotADirectoryError
    (traceback util.py:219)。
    """
    if os.path.exists(filepath):
        shutil.rmtree(filepath)
        os.mkdir(filepath)
    else:
        os.mkdir(filepath)


def get_random_port():
    """返回一个当前可绑定的随机端口。[pyd FUN_18000e0f0,pyd py=223..233]

    E1(probe1,200 次采样):int,min=1281/max=65301;E2 常量 1023/65535 与
    randint 调用点 → randint(1023, 65535);绑定失败打印 'Port %s is in use'。
    """
    port = random.randint(1023, 65535)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(('0.0.0.0', port))
            return port
        except OSError:
            print('Port %s is in use' % port)
    return get_random_port()


def kill_port(port):
    """结束监听指定端口的进程。[pyd FUN_18000f710,pyd py=236..242]

    E1(probe1):空闲端口与自起监听进程均返回 None 且无 stdout;进程被终止。
    """
    for conn in psutil.net_connections():
        if conn.laddr.port == port and conn.status == 'LISTEN':
            proc = psutil.Process(conn.pid)
            proc.terminate()
            proc.wait(timeout=5)


def download_zip(url):
    """下载 zip 到临时文件,返回 (url, tmp_file)。[pyd FUN_180010150,pyd py=244..258]

    E1(probe1/2,本地 stub):GET url(stream=True),写入 tempfile.TemporaryFile();
    返回二元组 (url, _TemporaryFileWrapper);进度条走 stderr。
    """
    if not url.startswith('http'):
        url = api_host + url
    tmp_file = tempfile.TemporaryFile()
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    pbar = tqdm(total=total_size, unit='iB', unit_scale=True, unit_divisor=1024)
    for chunk in response.iter_content(chunk_size=1024):
        tmp_file.write(chunk)
        pbar.set_description('正在下载中......')
        pbar.update(len(chunk))
    pbar.close()
    return url, tmp_file


def download_file(url, dist=None):
    """下载文件到 dist(默认 cache_dir/<basename>),返回落地路径。
    [pyd FUN_180011770,pyd py=260..292]

    E1(probe1/2):相对 URL 前拼 api_host;stdout '下载文件 <dist>';
    实际写盘字节与响应一致(4096/5000 字节);进度条走 stderr。
    """
    if not url.startswith('http'):
        url = api_host + url
    if dist is None:
        dist = os.path.join(cache_dir, os.path.basename(url))
    print('下载文件', dist)
    if not os.path.exists(os.path.dirname(dist)):
        os.makedirs(os.path.dirname(dist))
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(dist, 'wb') as f:
        pbar = tqdm(total=total_size, unit='iB', unit_scale=True, unit_divisor=1024)
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
            pbar.update(len(chunk))
        pbar.close()
    return dist


def support_gbk(zip_file):
    """把 zip 内以 cp437 误读的中文名还原为 gbk。[pyd FUN_180013bb0,pyd py=294..302]

    E1(probe2):入参必须是 ZipFile(传 str → AttributeError 'str' object has no
    attribute 'NameToInfo',util.py:295);GBK zip 名还原成功且 NameToInfo 键顺序
    变为 [ascii.txt, 中文文件.txt, 子目录/另一个.txt](仅重写发生变化的键);
    UTF-8 标志的 zip 名 encode('cp437') 抛 UnicodeEncodeError(不吞异常)。
    """
    for name in zip_file.NameToInfo.copy():
        new_name = name.encode('cp437').decode('gbk')
        if new_name != name:
            info = zip_file.NameToInfo[name]
            info.filename = new_name
            del zip_file.NameToInfo[name]
            zip_file.NameToInfo[new_name] = info
    return zip_file


def append_log(file_path, log_message):
    """追加一行日志(utf-8、平台换行)。[pyd FUN_180014600,pyd py=305..312]

    E1(probe1):目录缺失自动 makedirs(dirname, exist_ok=True);写入
    log_message + '\\n';Windows 文本模式落地为 CRLF。
    """
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(log_message + '\n')


# ---------------------------------------------------------------- 控制台输出
def print_yellow(message):
    """黄色(ANSI 93)整行输出。[pyd FUN_180015870,pyd py=315..317]
    E1:'\\x1b[93m' + str(message) + '\\x1b[0m' + 换行。"""
    print('\033[93m' + str(message) + '\033[0m')


def print_red(message):
    """红色(ANSI 91)整行输出。[pyd FUN_180015b70,pyd py=321..323]
    E1:'\\x1b[91m' + str(message) + '\\x1b[0m' + 换行。"""
    print('\033[91m' + str(message) + '\033[0m')


def print_yellow_tip(message, title='温馨提示'):
    """黄框提示三行。[pyd FUN_180016060,pyd py=327..331]
    E1:'\\x1b[93m--- {title} ---\\x1b[0m' / '\\x1b[93m{message}\\x1b[0m' /
    '\\x1b[93m---------------\\x1b[0m'。"""
    print('\033[93m--- ' + str(title) + ' ---\033[0m')
    print('\033[93m' + str(message) + '\033[0m')
    print('\033[93m---------------\033[0m')


def clear_line():
    """清行并把光标移回行首(不换行)。[pyd FUN_1800166d0,pyd py=333..335]
    E1:print('\\x1b[2K', end='\\r')。"""
    print('\033[2K', end='\r')


# ---------------------------------------------------------------- 性能优化
class WindowsPerformanceOptimizer:
    """Windows 进程性能优化。[pyd FUN_1800167d0..FUN_18001f490,pyd py=337..551]

    E1(probe1)实测输出:
      optimize_for_high_performance() → True,打印 '--- 正在优化Windows进程性能 ---' /
      '--- 优化结果 ---' / '✓ 进程优先级已提升至: HIGH' / '✓ 系统定时器精度已提升至: 1ms' /
      '✓ 已防止系统休眠和显示器关闭' / 'Windows性能优化配置完成。'+空行;
      revert_optimization() → None,打印 '\\n--- 正在恢复 Windows 性能设置 ---' /
      '✓ 进程优先级已恢复到原始值: 0x20' / '✓ 系统定时器精度已恢复。' /
      '✓ 防止休眠和显示器关闭的设置已恢复。' / 'Windows 性能设置恢复完成。'。
    """

    def __init__(self):
        self.kernel32 = ctypes.windll.kernel32
        self.user32 = ctypes.windll.user32
        self.winmm = ctypes.windll.winmm
        self.powrprof = ctypes.windll.powrprof
        self.process_handle = self.kernel32.GetCurrentProcess()
        self.original_priority = None
        self.original_execution_state = None
        self.timer_resolution_set = False
        self.target_timer_resolution = 1

    def _get_original_priority(self):
        self.original_priority = self.kernel32.GetPriorityClass(self.process_handle)
        return self.original_priority

    def _set_process_priority(self, realtime=False):
        try:
            self._get_original_priority()
            if realtime:
                priority_class = 0x100  # REALTIME_PRIORITY_CLASS
            else:
                priority_class = 0x80   # HIGH_PRIORITY_CLASS
            if not self.kernel32.SetPriorityClass(self.process_handle, priority_class):
                print_red('SetPriorityClass 失败, 错误代码: ' + str(ctypes.get_last_error()))
                return False
            return True
        except Exception as exc:
            print_red('提高进程优先级失败: ' + str(exc))
            return False

    def _increase_timer_resolution(self, target_timer_resolution=1):
        try:
            self.target_timer_resolution = target_timer_resolution
            if self.winmm.timeBeginPeriod(self.target_timer_resolution) != 0:
                print_red('设置定时器分辨率失败，错误代码: ' + str(ctypes.get_last_error()))
                return False
            self.timer_resolution_set = True
            return True
        except Exception as exc:
            print_red('提高定时器分辨率失败: ' + str(exc))
            return False

    def _prevent_sleep_and_display_off(self):
        try:
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_DISPLAY_REQUIRED = 0x00000002
            self.original_execution_state = self.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
            if not self.original_execution_state:
                print_red('SetThreadExecutionState 失败, 错误代码: '
                          + str(ctypes.get_last_error()))
                return False
            return True
        except Exception as exc:
            print_red('防止休眠设置失败: ' + str(exc))
            return False

    def optimize_for_high_performance(self, realtime_priority=False):
        print('--- 正在优化Windows进程性能 ---')
        try:
            priority_ok = self._set_process_priority(realtime_priority)
            timer_ok = self._increase_timer_resolution(1)
            sleep_ok = self._prevent_sleep_and_display_off()
            print('--- 优化结果 ---')
            if priority_ok:
                print('✓ 进程优先级已提升至: HIGH')
            if timer_ok:
                print('✓ 系统定时器精度已提升至: ' + str(self.target_timer_resolution) + 'ms')
            if sleep_ok:
                print('✓ 已防止系统休眠和显示器关闭')
            print('Windows性能优化配置完成。\n')
            return True
        except Exception as exc:
            print_red('优化过程中出现错误: ' + str(exc))
            return False

    def _revert_process_priority(self):
        try:
            if self.kernel32.SetPriorityClass(self.process_handle,
                                              self.original_priority):
                print('✓ 进程优先级已恢复到原始值: ' + hex(self.original_priority))
            else:
                print_red('恢复进程优先级失败，错误代码: ' + str(ctypes.get_last_error()))
        except Exception as exc:
            print_red('恢复进程优先级时发生错误: ' + str(exc))

    def _revert_timer_resolution(self):
        try:
            if self.timer_resolution_set:
                if self.winmm.timeEndPeriod(self.target_timer_resolution) == 0:
                    print('✓ 系统定时器精度已恢复。')
                else:
                    print_red('恢复定时器分辨率失败，错误代码: '
                              + str(ctypes.get_last_error()))
        except Exception as exc:
            print_red('恢复定时器分辨率时发生错误: ' + str(exc))

    def _revert_sleep_and_display_off(self):
        try:
            self.kernel32.SetThreadExecutionState(0x80000000)  # ES_CONTINUOUS
            print('✓ 防止休眠和显示器关闭的设置已恢复。')
        except Exception as exc:
            print_red('恢复休眠设置时发生错误: ' + str(exc))

    def revert_optimization(self):
        print('\n--- 正在恢复 Windows 性能设置 ---')
        self._revert_process_priority()
        self._revert_timer_resolution()
        self._revert_sleep_and_display_off()
        print('Windows 性能设置恢复完成。')


# ---------------------------------------------------------------- 管理员检查
def check_admin_privileges():
    """管理员权限检查(返回 bool)。[pyd FUN_18001fe10,pyd py=553..565]

    E1(probe1):VM 内以管理员运行 → 返回 True 并打印
    '>>> 权限检查：当前脚本**已**在管理员权限下运行。';
    另两条文案(未/无法确定)见 E2 串表 0x180040d78 / 0x180040a70。
    """
    try:
        if ctypes.windll.shell32.IsUserAnAdmin():
            print('>>> 权限检查：当前脚本**已**在管理员权限下运行。')
            return True
        print('>>> 权限检查：当前脚本**未**在管理员权限下运行。')
        return False
    except Exception:
        print('>>> 权限检查：无法确定权限状态。')
        return False


# ---------------------------------------------------------------- NTFS ADS
def delete_ads_for_file(file_path):
    """删除单文件的 Zone.Identifier 备用数据流。[pyd FUN_180020410,pyd py=567..586]

    E1(probe1/2):命中 → DeleteFileW 返回真 → print('已删除: ' + path + ' 的备用数据流')
    且返回 True;无该流(错误码 2)静默返回 False;其它错误码(只读文件实测 5)→
    print('删除 <path> 的备用数据流失败。错误代码: <code>') 且返回 False。
    """
    ads_path = file_path + ':Zone.Identifier'
    try:
        result = ctypes.windll.kernel32.DeleteFileW(ads_path)
        if result:
            print('已删除: ' + file_path + ' 的备用数据流')
            return True
        error_code = ctypes.windll.kernel32.GetLastError()
        if error_code != 2:
            print('删除 ' + file_path + ' 的备用数据流失败。错误代码: ' + str(error_code))
        return False
    except Exception as exc:
        print_red('删除 ' + file_path + ' 的备用数据流失败: ' + str(exc))
        return False


def delete_ads_recursive(directory):
    """递归清理目录树里的 ADS 并打印汇总。[pyd FUN_180020f80,pyd py=588..609]

    E1(probe1/2):返回 None;stdout 为 '\\n处理完成！' + '扫描文件总数: N' +
    '删除备用数据流总数: M'(N=os.walk 命中的文件总数,含无 ADS 的文件;
    目录不存在 → 0/0;传文件 → 0/0)。汇总由本函数打印,调用方不得重复打印。
    """
    file_count = 0
    ads_count = 0
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_count += 1
                try:
                    if delete_ads_for_file(os.path.join(root, file)):
                        ads_count += 1
                except Exception:
                    pass
    except Exception as exc:
        print_red('扫描目录时出错: ' + str(exc))
    print('\n处理完成！')
    print('扫描文件总数: ' + str(file_count))
    print('删除备用数据流总数: ' + str(ads_count))


# ---------------------------------------------------------------- 窗口图标
class WindowIconManager:
    """任务栏/窗口图标设置。[pyd FUN_180022120..FUN_180025b80,pyd py=610..723]

    E1(probe1):get_window_handle 失败 → print('获取窗口句柄失败: ' + str(exc)) 并返回
    None;extract_icon_from_exe 文件不存在 → print('EXE文件不存在: ' + exe) 返回 None,
    正常 exe → 返回 HICON 整数句柄;set_icon_from_exe/file 取不到句柄 →
    print('无法获取窗口句柄') 并返回 False。
    """

    def __init__(self):
        self.user32 = ctypes.windll.user32
        self.shell32 = ctypes.windll.shell32
        self.kernel32 = ctypes.windll.kernel32

    def get_window_handle(self, title):
        try:
            hwnd = self.user32.FindWindowW(None, title)
            return hwnd
        except Exception as exc:
            print('获取窗口句柄失败: ' + str(exc))
            return None

    def extract_icon_from_exe(self, exe_path, icon_index=0):
        try:
            if not os.path.exists(exe_path):
                print('EXE文件不存在: ' + exe_path)
                return None
            icon_handle = self.shell32.ExtractIconW(None, exe_path, icon_index)
            if not icon_handle:
                print('从EXE文件提取图标失败: ' + exe_path)
                return None
            return icon_handle
        except Exception as exc:
            print('提取图标失败: ' + str(exc))
            return None

    def set_icon_from_exe(self, exe_path, window_title, icon_index=0):
        try:
            hwnd = self.get_window_handle(window_title)
            if not hwnd:
                print('无法获取窗口句柄')
                return False
            icon_handle = self.extract_icon_from_exe(exe_path, icon_index)
            if not icon_handle:
                return False
            self.user32.SendMessageW(hwnd, 0x0080, 1, icon_handle)  # WM_SETICON ICON_BIG
            print('成功设置EXE图标, 句柄: ' + str(icon_handle))
            return True
        except Exception as exc:
            print('设置EXE图标失败: ' + str(exc))
            return False

    def set_icon_from_file(self, icon_path, window_title):
        try:
            hwnd = self.get_window_handle(window_title)
            if not hwnd:
                print('无法获取窗口句柄')
                return False
            icon_handle = self.user32.LoadImageW(None, icon_path, 1, 0, 0, 0x10)
            if not icon_handle:
                print('加载图标文件失败: ' + icon_path)
                return False
            self.user32.SendMessageW(hwnd, 0x0080, 1, icon_handle)
            print('成功设置图标: ' + str(icon_handle))
            return True
        except Exception as exc:
            print('设置图标失败: ' + str(exc))
            return False


# ---------------------------------------------------------------- 事件上报
def send_event_to_umami():
    """[COM-D002 语义替换] 原实现向第三方 umami 分析端点上报 client-status。

    原行为(E1 probe1/2,假 requests 实录):POST <原厂遥测主机>/api/send,
    json 载荷为 {'type':'event','payload':{'website':<站点标识>,
    'name':'client-status','data':{'hostname':urlparse(api_host).netloc}}},
    headers 含 Content-Type/User-Agent,timeout=10;status 200/201 → True,
    其它 → False;异常 → print('Error: ' + str(exc)) 并返回 False。
    即原实现把**用户机器标识(hostname)**发给第三方分析服务。

    按 D002 要求本函数改为**惰性空壳**:不发出任何网络请求,不引用任何第三方
    分析端点/站点标识/UA 字符串,除返回值外无副作用,恒返回 False(未启用)。
    函数名与签名按符号保留规则**原样保留**,以便 app_mqtt 的调度点、事件名与
    契约符号集不变;它永远不会发送任何东西。
    """
    return False
