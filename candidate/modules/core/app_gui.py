# -*- coding: utf-8 -*-
"""modules.core.app_gui — R028 T4R 语义重建(源 app_gui.cp310-win_amd64.pyd,契约切片:8 个模块函数 + 模块体)。

证据标注约定(每处均可回溯二进制):
- [exec L###]  = evidence/modules/core__app_gui/static/pseudocode/FUN_1800757d0__1800757d0.c(pymod_exec)
                 注解伪码行号;模块级导入/赋值顺序全部来自它(_pymod_exec_sequence.json 为提取物)。
- [FUN_x py=N] = 该函数的 Ghidra 反编译文件(FUN_<addr>__<addr>.c)与 __PYX_ERR 携带的 **源码行号 N**
                 (伪码里 `iVar = 0xNN;` 标记),行号与 E1 回溯(probe7:app_gui.py line 110/123)一致。
- [ST 0x...]   = evidence/modules/core__app_gui/static/stringtab.json 槽位(注解器把 DAT_ 槽替换为 STR("…"))。
- [cached]     = InitCachedConstants(FUN_18006fc50)里的缓存元组/整数(_codeobjects.json 的 tuples)。
- [E1 probeN]  = ns_w6gpu 共享目录内 oracle 实测(probe1_app_gui.json / probe6_gui.json / probe7_gui_out.json)。

结构性修正(相对 P007 试点切片):app_gui 的 pyd **自身没有** 横幅与 GPU 门控(串表无 'gpu trt'/
'缺少 nvidia' 字面量);横幅来自 `modules.core.config` 导入 [exec 1472],门控来自
`from modules.core.app_mqtt import start as start_live, vc_run` [exec 2256] 触发的 app_mqtt 模块体
(`缺少 nvidia显卡，当前电脑不可用` + time.sleep(10) + sys.exit())。E1 观测序(pydub 告警 → 横幅 →
门控消息 → SystemExit)与 pymod_exec 导入序(pydub 900 → config 1472 → app_mqtt 2256)完全自洽。

重建范围(契约 R028 namespace 声明的 8 个模块函数;其余符号见文末"未重建符号清单"):
  remove_banned_words / play_wav_on_device / find_live_studio_shortcut / get_shortcut_target /
  start_live_studio / callapi / open_browser / start_gui
"""

# ---- 标准库 / 三方库导入(顺序 = pymod_exec 模块字典写入序 [exec 227-976]) ----
import base64            # [exec 227]
import ctypes            # [exec 245]
import datetime          # [exec 263]
import json              # [exec 281]
import multiprocessing   # [exec 295]
import os                # [exec 309]
import pathlib           # [exec 323]
import platform          # [exec 337]
from queue import Queue  # [exec 370]
import queue             # [exec 385]
import shutil            # [exec 400]
import subprocess        # [exec 415]
import sys               # [exec 429]
import re                # [exec 443]
import threading         # [exec 457]
import time              # [exec 471]
from urllib import parse  # [exec 487-519]
from uuid import uuid4    # [exec 550]
import wave               # [exec 565]
import zipfile            # [exec 579]
import cv2                # [exec 593]
import psutil             # [exec 607]
import pyaudio            # [exec 621]
import webview            # [exec 635]
from playsound import playsound   # [exec 649-665]
import bottle             # [exec 681]
import pygetwindow        # [exec 695]
import tkinter as tk      # [exec 702-709]
import pyautogui          # [exec 723]
import numpy as np        # [exec 737]
from tkinter import Canvas          # [exec 753-767]
from webview.platforms.winforms import is_chromium   # [exec 784-798]
import win32com.client    # [exec 807-814]  (绑定名 win32com)
import screeninfo         # [exec 828]
import win32gui           # [exec 842]
import win32con           # [exec 856]
import win32api           # [exec 870]
import requests           # [exec 884]
from pydub import AudioSegment            # [exec 900-914]  (E1:导入期 pydub RuntimeWarning 首次出现点)
from pathvalidate import sanitize_filename  # [exec 931-945]
from easydict import EasyDict as edict     # [exec 962-976][FUN_1800757d0 py=41 'EasyDict'→'edict']

# ---- 本工程模块导入(顺序 [exec 1253-2446]) ----
from modules.core.cloud_ssh import run_proxy                                   # [exec 1253-1273]
from modules.core.app_util import is_port_in_use, request, runAsync, text2voice  # [exec 1296-1356]
from modules.core.gui_danmu import DanmuApi, closeLiveWindow                   # [exec 1376-1393]
from modules.core.config import (                                               # [exec 1472-1715](横幅在此打印)
    api_host, home_dir, temp_dir, models_dir, bottle_port, agent, vc_port, robot_wjc,
    srs_host, rtmp_port, static_port, cache_dir, index_port,
    thumb_port, strtobool, client_version,
)
from modules.core.gui_old import (                                              # [exec 1827-2103]
    doVideoDiy, downloadVideo, get_video_duration, playlist_close, replylist, support_gbk,
    infer_list, live, products, waititem, playlist, get_playing, playlist_play, playlist_stop,
)
from modules.core.util import (                                                 # [exec 2103-2160]
    WindowIconManager, download_file, download_zip, get_windows_computer_id, kill_port,
    print_red, print_yellow,
)
from modules.core.app_mqtt import start as start_live, vc_run                   # [exec 2256](GPU 门控在 app_mqtt 模块体)
from modules.core.train_util import TrainApi, fix_permission, set_train_window   # [exec 2316-2335](导入期加载 checkpoints/yolov8-face.pt)
from playwright.sync_api import sync_playwright                                 # [exec 2389]
from playwright_stealth import Stealth                                          # [exec 2427-2446]


def remove_banned_words(text, banned_words):                                    # [FUN_180001350 py=108][varnames text,banned_words,cleaned_text,wjc]
    cleaned_text = text                                                         # py=109(无 C 调用:cleaned_text 直接别名 text)
    for wjc in banned_words.split("|"):                                         # py=110 GetAttr(banned_words,'split')+call('|')[ST 0x18009b288];py=111 GetIter
        cleaned_text = cleaned_text.replace(wjc, "")                            # py=112 replace(wjc, '')[ST '' 槽]
    cleaned_text = cleaned_text.replace("[|", "[")                              # py=114 [cached DAT_18009dad8=('[|','[')]
    cleaned_text = cleaned_text.replace("||", "|")                              # py=115 [cached DAT_18009ca58=('||','|')]
    cleaned_text = cleaned_text.replace("|]", "]")                              # py=116 [cached DAT_18009e000=('|]',']')]
    cleaned_text = cleaned_text.replace("[]", "")                               # py=117 [cached DAT_18009d838=('[]','')]
    return cleaned_text                                                         # py=118
    # E1 probe7:('abcXYZdef','XYZ')→'abcdef';banned_words 传 list → AttributeError('list' object has no attribute 'split') @py=110


def play_wav_on_device(wav_file, device_index=7):                               # [FUN_180001d10 py=122][E1 probe6 签名 device_index=7][varnames wav_file,device_index,wf,sample_width,frame_rate,channels,p,stream,data,e]
    wf = wave.open(wav_file, "rb")                                              # py=123 wave.open(wav_file,'rb')[ST 'rb'];E1 probe7:缺文件 FileNotFoundError 由此上抛(try 在其后)
    sample_width = wf.getsampwidth()                                            # py=126
    frame_rate = wf.getframerate()                                              # py=127
    channels = wf.getnchannels()                                                # py=128
    p = pyaudio.PyAudio()                                                       # py=131
    stream = p.open(                                                            # py=134 GetAttr(p,'open')
        format=p.get_format_from_width(sample_width),                           # py=135 kw format/channels/rate/output/output_device_index(PyDict_SetItem 序)
        channels=channels,
        rate=frame_rate,
        output=True,
        output_device_index=device_index,
    )
    try:
        data = wf.readframes(wf.getnframes())                                   # py=144 GetAttr readframes + getnframes() 无参调用 → readframes(n)(无循环)
        stream.write(data)                                                      # py=145
        print("音频播放完成")                                                     # py=146 [cached DAT_18009ce20=('音频播放完成',)]
    except Exception as e:                                                      # py=147 PyErr_ExceptionMatches(Exception)
        print(f"播放音频时出错: {e}")                                             # py=148 PyObject_Format(e)+Concat('播放音频时出错: ')
    finally:
        stream.stop_stream()                                                    # py=151(正常/异常两条路径各出现一次 = finally)
        stream.close()                                                          # py=152
        p.terminate()                                                           # py=153


def find_live_studio_shortcut():                                                # [FUN_180003760 py=159]
    start_menu_path = (                                                         # [varnames 含 start_menu_path(InitCachedConstants 串)]
        pathlib.Path(os.environ["ProgramData"])                                 # py=160 pathlib.Path + os.environ['ProgramData'](PyObject_GetItem)
        / "Microsoft"                                                           # py=161 PyNumber_TrueDivide [ST 'Microsoft']
        / "Windows"                                                             # py=162
        / "Start Menu"                                                          # py=163
        / "Programs"                                                            # py=164
    )
    for shortcut in start_menu_path.glob("**/*.lnk"):                           # py=168 GetAttr glob + '**/*.lnk';list/tuple 快速迭代
        if "直播伴侣" in shortcut.name or "LiveStudio" in shortcut.name:          # py=170 PySequence_Contains(name,'直播伴侣') / ('LiveStudio')
            return shortcut                                                     # code_r0x000180003e90:返回循环项
    return None                                                                 # LAB_180003f27:_Py_NoneStruct;E1 probe7:VM 无直播伴侣 → None


def get_shortcut_target(shortcut_path):                                         # [FUN_180004010 py=178]
    try:
        shell = win32com.client.Dispatch("WScript.Shell")                       # py=179 win32com→client→Dispatch('WScript.Shell')[ST 0x180099ef8]
        shortcut = shell.CreateShortCut(str(shortcut_path))                     # py=180 CreateShortCut(PyUnicode_Type(shortcut_path))
        return shortcut.TargetPath                                              # py=181 GetAttr TargetPath;E1 probe7:不存在的 .lnk → ''(WScript 新建空快捷方式对象)
    except Exception as e:                                                      # py=182 PyErr_GivenExceptionMatches(Exception)
        print(f"解析快捷方式失败: {e}")                                           # py=183 Concat('解析快捷方式失败: ', format(e)) → print
        return None                                                             # pcVar15 = _Py_NoneStruct


def start_live_studio():                                                        # [FUN_180004930 py=189]
    shortcut = find_live_studio_shortcut()                                      # py=190 GetModuleGlobalName('find_live_studio_shortcut') 无参调用
    if not shortcut:                                                            # py=190-191 PyObject_IsTrue
        print("❌ 未找到抖音直播伴侣的快捷方式")                                     # py=191 [cached DAT_18009d078=('❌ 未找到抖音直播伴侣的快捷方式',)];E1 probe7 逐字一致
        return False                                                            # _Py_FalseStruct
    try:                                                                        # py=192
        subprocess.Popen(["start", "", str(shortcut)], shell=True)              # py=197 Popen(PyList_New(3)=['start','',str(shortcut)], shell=True)
        print(f"✅ 已启动抖音直播伴侣: {shortcut}")                                # py=198 Concat('✅ 已启动抖音直播伴侣: ', format(shortcut))
        return True                                                             # _Py_TrueStruct
    except Exception as e:                                                      # py=200
        print(f"❌ 启动失败: {e}")                                                # py=201 Concat('❌ 启动失败: ', format(e))
        return False


# ---- 模块级状态(顺序/初值 = [exec] py=205-225) ----
window = None                                                                   # py=205 _Py_NoneStruct
window_player = None                                                            # py=206
window_thumb = None                                                             # py=207
window_cloud = None                                                             # py=208
window_cloud_process = None                                                     # py=209
window_queue = Queue()                                                          # py=210 Queue() 无参调用
window_store = {}                                                               # py=211 PyDict_New()
window_dy = None                                                                # py=212
window_sph = None                                                               # py=213
ocr_running = False                                                             # py=215 _Py_FalseStruct
ocr_keyword = ""                                                                # py=216 [ST '']
ocr_region = None                                                               # py=217
dy_close_pattern = re.compile(r"\d+:\d{2}:\d{2}\s*关播")                         # py=220 re.compile [ST '\\d+:\\d{2}:\\d{2}\\s*关播']
shutdown_checked = True                                                         # py=221
shutdown_delay = 10                                                             # py=222 [cached DAT_18009ca50=PyLong(10)]
shutdown_doing = True                                                           # py=223
audio_checked = True                                                            # py=224
audio_index = 0                                                                 # py=225 [cached DAT_18009c3a0=PyLong(0)]


def callapi(data):                                                              # [FUN_1800053f0 py=229]
    headers = {"Content-Type": "application/json"}                              # py=230 PyDict_New + SetItem('Content-Type','application/json')
    requests.post(                                                              # py=232 GetModuleGlobalName('requests') → GetAttr 'post'
        "http://127.0.0.1:3060/local_danmu",                                    # [cached DAT_18009d258=('http://127.0.0.1:3060/local_danmu',)][ST 0x18009a010];E1 probe7:SPY requests.post 该 URL
        data=json.dumps(data),                                                  # py=233 kw data=json.dumps(data)
        headers=headers,                                                        # py=233 kw headers
    )
    return None                                                                 # pcVar8 = _Py_NoneStruct(无返回值)


browser_running = False                                                         # py=237 [exec] _Py_FalseStruct
browser_queue = Queue()                                                         # py=238 [exec] Queue()


def open_browser(url, with_stealth=True, headless=False):                       # [FUN_180006ab0 py=245][E1 probe6 签名][闭包 handle_response=FUN_180005bb0]
    global browser_running

    def handle_response(response):                                              # [FUN_180005bb0 py=245-247][qualname 'open_browser.<locals>.handle_response']
        if "/micro/live/cgi-bin/mmfinderassistant-bin/live/msg" in response.url:   # py=246 PySequence_Contains(response.url, ST '/micro/live/cgi-bin/…/live/msg')
            try:                                                                # py=247(ExceptionReset 路径;吞异常)
                data = response.json()                                          # GetAttr(response) 无参调用(Ghidra 丢失了属性名参数,按 playwright Response.json() 还原)
                for msg in data.get("data").get("msgList"):                     # 单参调用 ('data') / ('msgList'):.get 语义;list/tuple 快速迭代
                    print(f"✅ 弹幕: {msg.get('nickname')}: {msg.get('content')}")   # PyTuple_New(4) StrJoin:'✅ 弹幕: '+format(nickname)+': '+format(content)
                    callapi({                                                   # GetModuleGlobalName('callapi') + PyDict_New
                        "type": "chat",                                         # 首个 SetItem 的键/值被反编译器丢弃(register 参数);E1 G2 首轮 golden 校准:{"type":"chat","name":…}
                        "name": msg.get("nickname"),                            # E1 校准:键为 'name',值取 msg.get('nickname')
                        "content": msg.get("content"),
                        "clientMsgId": msg.get("clientMsgId"),
                    })
            except Exception:
                pass

    browser_running = True                                                      # FUN_180006ab0 L140 PyDict_SetItem(moddict,'browser_running',True)(函数入口)
    # py=267 PyObject_IsTrue(with_stealth) 与 py=269 两处 sync_playwright()(同标 0x10d)= 单行条件表达式选择上下文管理器
    with (Stealth().use_sync(sync_playwright()) if with_stealth else sync_playwright()) as p:   # py=269 Stealth().use_sync(sync_playwright()) / sync_playwright()
        browser = p.chromium.launch(headless=headless, channel="msedge")        # py=272-274 GetAttr chromium→launch;kw headless=param_4, channel='msedge'
        auth_state = os.path.join(home_dir, "sph_auth_state.json")             # py=278 os.path.join(home_dir,'sph_auth_state.json')
        if os.path.exists(auth_state):                                          # py=279 os.path.exists + IsTrue
            context = browser.new_context(storage_state=auth_state)             # py=280 kw storage_state
        else:
            context = browser.new_context()                                     # py=282 无参
        page = context.new_page()                                               # py=283
        page.on("response", handle_response)                                    # py=284 on('response', 闭包)
        page.goto(url, wait_until="domcontentloaded", timeout=0)                # py=285 kw wait_until='domcontentloaded', timeout=[cached DAT_18009c3a0=0]
        while browser_running:                                                  # py=287 GetModuleGlobalName('browser_running') IsTrue
            try:                                                                # py=288
                msg = browser_queue.get_nowait()                                # py=290
                msg = edict(json.loads(msg))                                    # py=291
                if msg.type == "danmu":                                         # py=292 PyUnicode_Equals(type,'danmu')
                    app = page.locator("wujie-app")                             # py=293 locator('wujie-app') 只调用一次(E1 G2 首轮校准:oracle 调用序无第二次 locator)
                    app.get_by_placeholder("主播发言").fill(msg.content)         # py=294-295 get_by_placeholder('主播发言')→fill(msg.content)
                    app.get_by_role("button", name="发送").click()              # py=296-297 get_by_role(('button',)[cached DAT_18009d750], name='发送')→click()
            except:                                                             # py=298 bare except(无 GivenExceptionMatches):队列空等
                pass
            try:
                page.wait_for_timeout(1000)                                     # py=302 [cached DAT_18009b800=PyLong(1000)]
            except Exception:                                                   # py=303
                print("浏览器已关闭", url)                                        # py=304 PyTuple_New(2):('浏览器已关闭', 第二元=局部 url;E1 G2 首轮校准:oracle 打印 url 而非异常)
                browser_running = False                                         # py=305 PyDict_SetItem(moddict,'browser_running',False)


# ---- 未重建符号清单(不在 R028 契约 namespace 声明内;pymod_exec 定义序见 _pymod_exec_sequence.json) ----
# exit_transparent_window / create_transparent_window / get_dy_region / get_sph_region /
# rapidocr_to_paddle_format / capture_and_ocr / find_match_img_postion / is_regex / get_cookies /
# find_match_text_postion / start_shutdown / get_windows_scaling / force_activate_window /
# activate_windows_by_title / ocr_log / class Api(DanmuApi, TrainApi) 其余 77 方法 / on_closed
# —— 留待 R028 全量重开;start_gui 对 Api/on_closed 的引用在用例里以双侧同形的记录型桩注入(函数级行为差分)。
# (start_bottle / start_static 已由 U008-a 按反编译重建;Api 类 84 方法中的 7 个 URL/JS 承载方法已由
#  U008-b 按反编译重建(见下方 class Api 定义):start_live/start_app/start_player/refresh_live/
#  start_cloud_page/start_cloud_proxy/get_rtmp_url;其余 77 方法与 on_closed 保持未重建口径。
#  模块定义序 Api→on_closed→start_bottle→start_static→start_gui,on_closed 在本文件以本注释块占位。)
#
# [U008-b R2/R3 裁定注记]
# - R2 死常量:http://127.0.0.1:3062/ 见于串表/constants.txt:1926(stringtab 槽 0x18009a2c8,
#   InitCachedConstants DAT_18009d308=PyTuple_Pack(2,'http://127.0.0.1:3062/','')),唯一活跃引用点为
#   未重建方法 Api.open_video_dir(FUN_180050570 py=1790)的 str.replace('http://127.0.0.1:3062/','')
#   ——旧版 URL 前缀清理残留,无任何 HTTP 行为;按 survey §6-R2 处置:保留死串(串表对齐),不接行为,
#   不在候选代码中出现(open_video_dir 未重建)。
# - R3 thumb_port(3061)归属:已定谳为"宿主不承载 3061 服务端"。证据链:①app_gui 全部伪代码中
#   bottle/Bottle/run 仅见于 start_bottle/start_static 两实例(3060/3063)+模块导入;②thumb_port 仅
#   出现于 config 导入(FUN_1800757d0 py=1522-1530)与 Api.start_player 的 URL 拼接(FUN_18002eda0);
#   ③'/frame' 串全库仅 1 槽,即 start_player 的 URL 第三段;④window_thumb 全部 9 个写点
#   (start_player/close_player/send_palyer/close_app/_close_app/switch_thumb_full/move_monitor_full/
#   on_closed/pymod_exec)均为 webview 窗口管理,无起服;⑤create_transparent_window(FUN_180009d60/
#   FUN_180009f60)为纯 Tk 透明 OCR 窗(ocr_tk/overrideredirect/geometry/Canvas),无 HTTP;⑥其他模块
#   证据 grep 3061 仅 config 默认值(core__config D#35),'其他模块 '/frame' 0 命中;⑦U002 实测启动态
#   3061 拒绝连接(evidence/ui/U002/routes/routes_probe.json row18)。结论:播放窗 iframe 指向的
#   http://127.0.0.1:{thumb_port}/frame 为旧版残留端点,原版宿主进程内无服务端,运行期该 iframe
#   不可达;不重建任何 3061 服务端。若 U008-f 真窗复测发现 3061 有监听(与上述全部证据矛盾),再翻案。


class Api(DanmuApi, TrainApi):                                                  # [ST 'Api'(_maps slot2str 0x18009a180)][U008-b:仅重建 7 个 URL/JS 承载方法,其余 77 方法留 R028 全量]

    def start_live(self, id, title):                                            # [FUN_18002a180 py=932-945][错误标注 :601 'app_gui.Api.start_live']
        global window_player                                                    # 原版对模块级 window_player 赋值 = PyDict_SetItem(moddict,'window_player',…)(:459);Python 函数内需 global 声明才等价
        self.close_app()                                                        # py=932 GetAttr('close_app') 无参调用
        url = f"{api_host}/live?id={id}"                                        # py=934 PyTuple(3) StrJoin [ST '/live?id='](api_host 全局)
        webview._settings["user_agent"] = None                                  # py=935 GetAttr '_settings'+PyObject_SetItem [ST 'user_agent'][None]
        window_player = webview.create_window(                                  # py=936-939 GetAttr 'create_window';PyTuple(2)=(title,url) 位置参 + kwargs
            title,                                                              # param_4(外部传入的窗口标题)
            url,
            width=1400,                                                         # py=939 [cached DAT_18009d2c0=PyLong(0x578)=1400]
            height=800,                                                         # [cached DAT_18009df10=PyLong(800)]
            resizable=False,                                                    # [ST 'resizable'][False]
            js_api=self,                                                        # [ST 'js_api'](param_2=self)
            frameless=False,                                                    # [ST 'frameless'][False]
        )
        window_player.events.closed += self.close_app                           # py=945 GetAttr 'events'/'closed'+PyNumber_InPlaceAdd(self.close_app)+SetAttr 写回
        # 无 return 语句:隐式 None(pcVar15=_Py_NoneStruct)

    def start_app(self, id):                                                    # [FUN_18002b700 py=963-977+][错误标注 :45 'app_gui.Api.start_app']
        global window_player                                                    # 同 start_live:模块级写回(:686 PyDict_SetItem 'window_player')
        self.close_app()                                                        # py=963 GetAttr('close_app') 无参调用
        device_uid = get_windows_computer_id()                                  # py=964-965 GetModuleGlobal('get_windows_computer_id') 无参调用(util 导入)
        url = f"{api_host}/live2?id={id}&deviceUid={device_uid}"                # py=966 PyTuple(5) StrJoin [ST '/live2?id='/'&deviceUid='](FUN_18007e070(…,5,…))
        webview._settings["user_agent"] = None                                  # py=967 PyObject_SetItem [ST 'user_agent'][None]
        window_player = webview.create_window(                                  # py=968-969 kwargs-only(PyDict_New+SetItem 序;FUN_18007d730(create_window,空元组,kwargs))
            title="开播控制台 " + client_version,                                # py=969 PyUnicode_Concat [ST '开播控制台 '][全局 client_version]
            url=url,                                                            # [ST 'url']
            width=1200,                                                         # [cached DAT_18009d688=PyLong(0x4b0)=1200]
            height=860,                                                         # [cached DAT_18009c670=PyLong(0x35c)=860]
            resizable=False,                                                    # [False]
            js_api=self,                                                        # [ST 'js_api']
            frameless=False,                                                    # [False]
        )
        window_player.events.closed += self._close_app                          # py=977 [ST '_close_app'](注意下划线前缀,与 start_live 的 close_app 不同)
        return "ok"                                                             # [ST 'ok'](:808-809)

    def start_player(self, id):                                                 # [FUN_18002eda0 py=1047-1066][错误标注 :981/:1050 'app_gui.Api.start_player']
        global window_thumb                                                     # 模块级写回(:637 PyDict_SetItem 'window_thumb';destroy 路径 :946 同)
        if window_thumb:                                                        # py=1047 IsTrue(全局 window_thumb)
            try:                                                                # except Exception 展开见 :977-1012(PyExc_Exception 匹配)
                window_thumb.destroy()                                          # py=1049 GetAttr 'destroy' 无参调用
                window_thumb = None                                             # py=1050 PyDict_SetItem(moddict,'window_thumb',None)
            except Exception:                                                   # py=1051 destroy 成功才置 None;异常吞掉后不置 None 直接续(url 拼接)
                pass
        url = f"{api_host}/player?id={id}&url=http://127.0.0.1:{thumb_port}/frame"    # py=1054 PyTuple(6) StrJoin [ST '/player?id='/'&url=http://127.0.0.1:'/'/frame'](thumb_port 全局;三段拼接顺序=槽位序)
        webview._settings["user_agent"] = None                                  # py=1055 PyObject_SetItem [ST 'user_agent'][None]
        window_thumb = webview.create_window(                                   # py=1056-1059 PyTuple(2) 位置参 + kwargs
            "直播窗口（窗口采集方式选择win10）",                                  # [ST 全串](:551-552)
            url,
            width=720,                                                          # py=1059 [cached DAT_18009bde0=PyLong(0x2d0)=720]
            height=1280,                                                        # [cached DAT_18009c408=PyLong(0x500)=1280]
            resizable=True,                                                     # [True]
            js_api=self,                                                        # [ST 'js_api']
            frameless=False,                                                    # [False]
            on_top=True,                                                        # [ST 'on_top'][True]
        )
        window_thumb.events.closed += self.close_player                         # py=1066 (0x42a) GetAttr self 'close_player'+InPlaceAdd;全局写回 'window_thumb'
        return "ok"                                                             # [ST 'ok'](:795-796)

    def refresh_live(self):                                                     # [FUN_180030b80 py=1098][错误标注 'app_gui.Api.refresh_live']
        if window:                                                              # IsTrue(全局 window=主窗)
            window.evaluate_js(                                                 # py=1098 GetAttr 属性名被反编译器丢弃(单参方法;按窗口对象+串表语义定 evaluate_js,同 local_danmu 先例)
                "\nconsole.log(\"load js\")\nasync function main() {\n    window.vuethis.fetchLive()\n}\nmain()\n"    # [ST 串](:143/:151;JS 桥注入 window.vuethis.fetchLive(),无 URL 拼接)
            )                                                                   # 无 try/except:evaluate_js 异常上抛;返回 None

    def get_rtmp_url(self):                                                     # [FUN_1800607c0 py≈2029][错误标注 :247 'app_gui.Api.get_rtmp_url';Ghidra 原型(void) 未用 self,Cython 绑定方法仍按实例方法落]
        return f"rtmp://{srs_host}:{rtmp_port}/live"                            # PyTuple(5) StrJoin [ST 'rtmp://'/':'/'/live'](srs_host/rtmp_port 全局;FUN_18007e070(…,5,…))

    def start_cloud_page(self):                                                 # [FUN_180063080 py=2072-2090][错误标注 :743/:814 'app_gui.Api.start_cloud_page']
        global window_cloud                                                     # 模块级写回(:555 区域 PyDict_SetItem 'window_cloud';destroy 路径 :722 同)
        if window_cloud:                                                        # py=2072 IsTrue(全局 window_cloud)
            try:                                                                # except Exception 展开见 :736-760
                window_cloud.destroy()                                          # py=2074 (0x81a)
                window_cloud = None                                             # py=2075 (0x81b) PyDict_SetItem(moddict,'window_cloud',None)
            except Exception:                                                   # py=2076 (0x81c) 吞掉后直接续
                pass
        url = f"{api_host}/cloud"                                               # py=2079 (0x81f) PyUnicode_Concat(format(api_host),'/cloud') 两段拼接 [ST '/cloud']
        webview._settings["user_agent"] = None                                  # py=2080 (0x820)
        window_cloud = webview.create_window(                                   # py=2081-2084 PyTuple(2)=('云算力',url) 位置参 + kwargs(:327-330)
            "云算力",                                                            # [ST '云算力']
            url,
            width=500,                                                          # py=2084 [cached DAT_18009d648=PyLong(500)]
            height=900,                                                         # [cached DAT_18009c9a0=PyLong(900)]
            resizable=True,                                                     # [True](:366)
            js_api=self,                                                        # [ST 'js_api'](:374)
            frameless=False,                                                    # [False](:382-383)
        )
        window_cloud.events.closed += self.close_cloud_page                     # py=2090 (0x82a) GetAttr self 'close_cloud_page'+InPlaceAdd;全局写回 'window_cloud'
        return "ok"                                                             # [ST 'ok'](:561)

    def start_cloud_proxy(self, data):                                          # [FUN_1800641f0 py=2095-2136][错误标注 :258/:1873/:2231 'app_gui.Api.start_cloud_proxy';param_3=dict 载荷]
        global window_cloud_process                                             # 模块级写回(:790 PyDict_SetItem 'window_cloud_process';terminate 路径 :221 同)
        if window_cloud_process:                                                # py=2095 (0x82f)
            window_cloud_process.terminate()                                    # py=2097 (0x831) GetAttr 'terminate' 无参调用
            window_cloud_process = None                                         # py=2098 PyDict_SetItem(moddict,'window_cloud_process',None)
        voice_type = data.get("voice_type", "")                                 # py=2102 (0x836) PyDict_GetItemWithError+缺省 [ST 'voice_type'][ST '']
        ssh_cmd = data.get("ssh_cmd", "")                                       # py=2103 (0x837) [ST 'ssh_cmd'][ST '']
        ssh_password = data["ssh_password"]                                     # py=2104 (0x838) PyDict_GetItemWithError 无缺省串槽(__Pyx_PyDict_GetItem 宏:缺键 KeyError 上抛)
        print("start_cloud_proxy")                                              # py=2106 (0x83a) FUN_18007d730(print 槽 DAT_18009db68, [cached DAT_18009b960=('start_cloud_proxy',)])
        if voice_type == "v7":                                                  # py=2107 (0x83b) FUN_18007eff0(voice_type,'v7',Py_EQ)
            kill_port(index_port)                                               # py=2108 (0x83c) [全局 kill_port/index_port]
            window_cloud_process = multiprocessing.Process(                     # py=2109-2112 (0x83d-0x83f) kwargs PyDict_SetItem 序
                target=run_proxy,                                               # py=2110 [ST 'target'][全局 run_proxy(cloud_ssh 导入)]
                args=(ssh_cmd, ssh_password, index_port),                       # PyTuple(3) [ST 'args'](槽序 local_c8=ssh_cmd,pcVar11=ssh_password,pcVar27=index_port)
                daemon=True,                                                    # [ST 'daemon'][True]
            )
            window_cloud_process.start()                                        # py=2114 (0x842) GetAttr 'start' 无参调用
        time.sleep(3)                                                           # py=2116 (0x844) [cached DAT_18009cfa0=PyLong(3)]
        attempt = 1                                                             # [cached DAT_18009d7a0=PyLong(1)]
        while attempt <= 5:                                                     # [cached DAT_18009d568=PyLong(5)](循环尾部 PyObject_RichCompare op 被反编译器丢弃;按失败串'重试5次后仍失败'锚定 <==5 次尝试;do-while 形态)
            try:
                res = requests.get(f"http://127.0.0.1:{index_port}/health")     # py=2121 (0x849) PyTuple(3) StrJoin [ST 'http://127.0.0.1:'/'/health'](index_port 全局;requests.get 单参)
                res.raise_for_status()                                          # py=2122 (0x84a) GetAttr 'raise_for_status' 无参调用
                if res.text.strip() == "ok":                                    # py=2123 (0x84b) GetAttr 'text'/'strip'+FUN_18007eff0(…,'ok',Py_EQ)
                    print("✅ 目标服务器连通")                                   # [cached DAT_18009cf78=('✅ 目标服务器连通',)]
                    pathlib.Path(os.path.join(home_dir, "is_cloud.lock")).touch()     # [ST 'is_cloud.lock'][全局 pathlib/os/home_dir](行号槽 0x855-0x857=2133-2135)
                    return "ok"                                                 # [ST 'ok'](LAB_1800669d6)
                raise AssertionError                                            # :1808-1809 PyErr_SetNone(PyExc_AssertionError)(strip != 'ok' 路径)
            except Exception:                                                   # py=2126 (0x84e) FUN_18007dda0 匹配 PyExc_Exception
                print("目标服务器无法连通，等待10秒后重试")                       # py=2127 (0x84f) [cached DAT_18009ba30]
                time.sleep(10)                                                  # py=2128 (0x850) [cached DAT_18009ca50=PyLong(10)]
            attempt += 1                                                        # py=2130 (0x852) __Pyx_PyInt_AddObjC inplace=1
        print("❌目标服务器无法连通，重试5次后仍失败")                            # py=2132 (0x854) [cached DAT_18009b7e8]
        return "目标服务器无法连通"                                              # [ST 串](:2137-2141 plVar12=STR 后直返)


# ---- 未重建符号清单(续) ----
# Api 其余 77 方法(qualname 槽见 _maps.json 'Api.*';U008-b 未触及,含 start_playlist/start_cloud_proxy 之外
# 的 URL 无关方法):get_version/start_playmsg/play_text/play_audio/select_videos/stop_live/close_live/
# live_push_event/start_playlist/change_title/store_app/_close_app/close_app/close_player/send_palyer/
# mqtt_msg/sph_request/open_sph/close_dy/open_eos/open_qy/open_clue/open_buyin/open_dy_url/open_new_dy_url/
# select_region/get_audio_devices/try_audio/ocr_status/start_douyin/get_screenshot/cancel_shutdown/
# change_ocr/start_openlive/switch_thumb_full/get_monitors/move_monitor_full/install_cable/open_dir/
# open_video_dir/remove_wjc/get_camera_devices/get_local_models/select_player_dir/slice_audio_file/
# asr_file/select_ref_wav/vc_audio_file/select_site_video/close_cloud_page/stop_cloud_proxy/build_launcher
# 等,留待 R028 全量重开。
# [U008-b 注记]start_playlist(FUN_180025200,py=853-858 URL 段):request("/api/public/get-live-playlist",
# {"id": …}) 为 app_util.request 相对路径调用,完整 URL = api_host + '/api/public/get-live-playlist' 由
# app_util.request(app_host or api_host 拼接,已重建+golden 承载)完成;方法本体(playlist 窗口逻辑,4530 行
# 伪码)留 R028 全量。'{"id": …}' 的值寄存器被反编译器丢弃,未定谳。


def on_closed():                                                                # [R028 前置补建 2026-09-19]
    """窗口关闭后的进程级清场(原为未重建占位,此处按证据补建)。

    pyd 证据:qualname 'app_gui.on_closed';串表 'Signal received, terminating
    processes...'/'terminate'/'TerminateProcess'/b'_exit'/'is_cloud.lock'。
    行为面:srs、TTS、云代理等子进程并非 daemon,主进程退出后会残留,按监听端口
    收割;进程内的 bottle/thumb/static 服务端口不能 kill_port(会误杀自身进程)。
    """
    print('Signal received, terminating processes...')
    from modules.core.config import (
        srs_port, mqtt_port, tts_port, voxcpm_port, luxtts_port, omnivoice_port,
        srs_rtc_port, srs_proxy_port, srs_rtc_proxy_port,
    )
    for port in (srs_port, mqtt_port, tts_port, voxcpm_port, luxtts_port,
                 omnivoice_port, vc_port, index_port, rtmp_port,
                 srs_rtc_port, srs_proxy_port, srs_rtc_proxy_port):
        try:
            kill_port(port)
        except Exception:
            pass
    try:
        pathlib.Path(os.path.join(home_dir, "is_cloud.lock")).unlink()
    except Exception:
        pass
    os._exit(0)


def start_bottle():                                                             # [FUN_18006b7e0 py≈2200][U008-a 重建]
    app = bottle.Bottle()                                                       # py=2201 GetModuleGlobalName('bottle')→GetAttr 'Bottle' 无参调用
    @app.route("/local_video_thumb")                                            # py=2203 GetAttr(app,'route') 1 位置参 [ST '/local_video_thumb'](bottle 默认 GET)
    def local_video_thumb():                                                    # py=2204 [PyCode_New firstlineno=0x89c][varnames('headers','video_path','video','ret','frame','body')=DAT_18009c860][qualname 'start_bottle.<locals>.local_video_thumb']
        headers = {"Content-Type": "image/png"}                                 # py=2205-2206 PyDict_New+SetItem [ST 'Content-Type'/'image/png']
        video_path = bottle.request.query.thumb                                 # py=2207 GetAttr 链 bottle→request→query→thumb(属性访问;FormsDict 缺参 → '')
        video_path = parse.unquote(video_path)                                  # py=2208 GetModuleGlobalName('parse')→GetAttr 'unquote' 1 参
        video = cv2.VideoCapture(video_path)                                    # py=2210 GetAttr 'VideoCapture' 1 参(py=2209 空行/注释不可考)
        ret, frame = video.read()                                               # py=2211 GetAttr 'read' 无参调用→UNPACK(2)(tuple/list 快速路径)
        if ret:                                                                 # py=2212 PyObject_IsTrue(ret 为假→函数尾隐式返回 None→bottle 200 空体 text/html)
            video.release()                                                     # py=2214 GetAttr 'release' 无参调用(py=2213 空行/注释不可考)
            frame = cv2.resize(frame, (360, 640))                               # py=2215 GetAttr 'resize' 2 位置参 [cached DAT_18009d380=PyTuple_Pack(2,PyLong(0x168),PyLong(0x280))=(360,640)]
            body = cv2.imencode(".png", frame)[1]                               # py=2216 GetAttr 'imencode' 2 位置参 [ST '.png']→GetItemInt(1)
            return bottle.HTTPResponse(body.tobytes(), **headers)               # py=2217 GetAttr 'HTTPResponse' 1 位置参 + PyDict_Copy(headers) 作 **kwargs(落 more_headers → Content-Type: image/png)
    @app.post("/local_danmu")                                                   # py=2219 GetAttr(app,'post') 1 位置参 [ST '/local_danmu'](仅 POST;GET → bottle 405)
    def local_danmu():                                                          # py=2220 [PyCode_New firstlineno=0x8ac][varnames('danmu',)=DAT_18009c4e0][qualname 'start_bottle.<locals>.local_danmu']
        danmu = bottle.request.json                                             # py=2222 GetAttr 链(唯一带 __PYX_ERR 的语句 py=2222;不在 try 内,异常直接上抛)
        try:                                                                    # py=2223(推断)ExceptionSave/Reset 对出现在 window_player 段之前→try 从此开始
            window_player.evaluate_js(                                          # py=2224(推断)GetModuleGlobalName('window_player')→GetAttr 'evaluate_js' 1 参
                "\n    console.log(\"load js\")\n    async function main() {\n        window.vuethis.pushMsg(`"     # [ST 0x18009aff8 串表#329 89B]
                + json.dumps(danmu)                                             # GetModuleGlobalName('json')→GetAttr 'dumps' 1 参;两处 PyNumber_Add = 运行期 '+' 串联(非 f-string)
                + "`)\n    }\n    main()\n    "                                 # [ST 0x18009a890 串表#594 25B]
            )
        except:                                                                 # 无 PyErr_GivenExceptionMatches → bare except(window_player 未建/evaluate_js 抛错均吞掉)
            pass
        return "ok"                                                             # [ST 'ok'](成功/被吞两条路径都到达)
    # ---- [自建前端 2026-09-19] 本地控制台页面与引擎接口(全离线,零云端) ----
    from modules.core import local_engine
    from modules.core import train_service            # [训练接入 2026-09-20] 一键训练(模特/声音资产制备)
    from modules.core import works_service            # [短视频接入 2026-09-20] 本地短视频生成

    # [训练接入 2026-09-20] 上传走 multipart,bottle 默认 MEMFILE_MAX 100KB
    # 不够训练视频;提到 512MB(各端点自身的体积上限校验不受影响)。
    bottle.BaseRequest.MEMFILE_MAX = 512 * 1024 * 1024

    _ui_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "..", "..", "local_ui"))

    @app.get("/local_ui")
    def local_ui_page():
        resp = bottle.static_file("index.html", root=_ui_dir,
                                  mimetype="text/html; charset=utf-8")
        resp.set_header("Cache-Control", "no-cache, no-store, must-revalidate")
        return resp

    @app.get("/local_ui/<fname:path>")
    def local_ui_asset(fname):
        return bottle.static_file(fname, root=_ui_dir)

    # ---- [自建前端 2026-09-19 第二批] 资源管理/直播持久化/本地 LLM 代理 ----
    import urllib.request as _urlreq
    import urllib.error as _urlerr

    # [R-2-001/T-064] Windows 大小写不敏感保留名黑名单(含带扩展形态: NUL.txt 同为设备名)
    _WIN_RESERVED = frozenset(
        ("CON", "PRN", "AUX", "NUL")
        + tuple("COM%d" % i for i in range(1, 10))
        + tuple("LPT%d" % i for i in range(1, 10))
    )

    def _safe_name(name):
        """资源名白名单校验:拒绝路径分隔符/..、尾点尾空格与 Windows 保留名,防目录逃逸与资产破坏。"""
        if not isinstance(name, str):
            return None
        if name != name.rstrip(". "):           # [R-1A-001/T-003][R-2-001/T-064] 拒尾点/尾空格(对原始输入判,须先于 strip,否则 "abc " 被洗白)
            return None
        name = name.strip()                     # 首尾空白收敛(前导空白维持旧语义),后续校验均针对有效名(白名单只收紧不放松,契约 §1.3)
        if not name:
            return None
        if "/" in name or "\\" in name or ".." in name or ":" in name or name.startswith("."):
            return None
        if len(name) > 255:                     # [R-2-001/T-064] 限长(NTFS 单组件上限 255)
            return None
        if name.split(".", 1)[0].upper() in _WIN_RESERVED:   # [R-2-001/T-064] 保留名,含 CON.txt 带扩展形态
            return None
        return name

    def _json(obj):
        bottle.response.content_type = "application/json; charset=utf-8"
        return json.dumps(obj, ensure_ascii=False)

    def _json_err(status, msg):
        """[R-1D-008/T-031][R-1D-009/T-032][T-061][R-2-007/T-067] 错误响应统一
        {"ok":false,"error":<中文>}+指定 HTTP 状态(契约 C2,替代裸 500/HTML)。"""
        bottle.response.status = status
        return _json({"ok": False, "error": msg})

    _BODY_LIMIT = 10 * 1024 * 1024              # [R-2-007/T-067] 请求体统一上限 10MB

    def _drain_request_body():
        """[R-2-007/T-067] 排干超限请求体:循环 read 丢弃,循环回环上限按 Content-Length。
        不排干就提前结束响应会令 WSGIRef 提前关闭连接,客户端 send 阶段收 RST、读不到 413(续3 验收实测)。"""
        remain = int(bottle.request.content_length or 0)
        try:
            stream = bottle.request.body
            while remain > 0:
                chunk = stream.read(min(remain, 65536))
                if not chunk:
                    break                       # 客户端提前断流:能排多少排多少,413 照常回
                remain -= len(chunk)
        except Exception:
            pass                                # 排干失败不改 413 语义(响应仍要回)

    def _body_limit_exceeded():
        """[R-2-007/T-067] 提前校验 Content-Length;超限先排干请求体(见上),
        再 status=413 + Content-Type application/json + {"ok":false,"error":...},稳定可诊断。"""
        if (bottle.request.content_length or 0) > _BODY_LIMIT:
            _drain_request_body()               # [T-067/续3] 先排干再回,防 RST 吞响应
            return _json_err(413, "请求体过大,上限10MB")
        return None

    def _norm_res_name(n):
        """资源名规范化(NTFS 语义):大小写归一 + 路径规整 + 剥尾点尾空格。"""
        if not isinstance(n, str):
            return None
        return os.path.normcase(os.path.normpath(n)).rstrip(". ")

    def _resource_in_use(kind, name):
        """在用判定:仅直播中(running)且名字规范化后相等才算在用。"""
        # [R-1D-002/T-003](含 R-2-003 波次2 实测) running 双条件:下播后 status 残留的
        # model/voice 不再永久锁死 delete/rename;在用拒绝文案保留(契约 §1.1)。
        st = local_engine.status()
        if not st.get("running"):
            return False
        if kind == "models":
            cur = st.get("model")
        elif kind == "voices":
            cur = st.get("voice")
        else:
            return False
        # [R-1A-001/T-003] 两侧规范化比较,堵大小写/尾点/尾空格变体绕过
        return _norm_res_name(cur) == _norm_res_name(name)

    _res_op_lock = threading.Lock()             # [R-1A-001/T-003] 资源操作互斥锁(WSGIRef 按线程服务,防 TOCTOU)

    def _safe_resource_op(kind, name, op, new=None):
        """删除/改名公共路径:在用拒绝 + 异常兜底,返回 (ok, error)。"""
        # [R-1A-001/T-003] 全程持锁:在用/存在/冲突校验与 rmtree/rename 之间不再有并发窗口
        with _res_op_lock:
            if _resource_in_use(kind, name):
                return False, "资源正在直播中使用,请先下播"
            base = models_dir if kind == "models" else os.path.join(home_dir, "voices")
            path = os.path.join(base, name)
            try:
                if op == "delete":
                    if not os.path.isdir(path):
                        return False, "not found"
                    shutil.rmtree(path)
                elif op == "rename":
                    tgt = os.path.join(base, new)
                    if not os.path.isdir(path):
                        return False, "not found"
                    if os.path.exists(tgt):
                        return False, "conflict"
                    os.rename(path, tgt)
                    # [R-2-001/T-064] 回读校验:目标必须以请求名原样出现在目录里,
                    # 被系统规整(如尾点被剥)即回滚,杜绝列表名与真实目录名分叉
                    if new not in os.listdir(base):
                        try:
                            os.rename(tgt, path)
                            return False, "rename verify failed, rolled back"
                        except OSError:
                            return False, "rename verify failed, rollback failed"
                return True, None
            except OSError as e:
                return False, "文件系统错误: %s" % e

    @app.post("/api/resources/models/delete")
    def api_models_delete():
        name = _safe_name((bottle.request.json or {}).get("name"))
        if not name:
            return _json({"ok": False, "error": "invalid name"})
        ok, err = _safe_resource_op("models", name, "delete")
        return _json({"ok": ok, "error": err})

    @app.post("/api/resources/models/rename")
    def api_models_rename():
        body = bottle.request.json or {}
        old, new = _safe_name(body.get("old")), _safe_name(body.get("new"))
        if not old or not new:
            return _json({"ok": False, "error": "invalid name"})
        ok, err = _safe_resource_op("models", old, "rename", new)
        return _json({"ok": ok, "error": err})

    @app.post("/api/resources/voices/delete")
    def api_voices_delete():
        name = _safe_name((bottle.request.json or {}).get("name"))
        if not name:
            return _json({"ok": False, "error": "invalid name"})
        ok, err = _safe_resource_op("voices", name, "delete")
        return _json({"ok": ok, "error": err})

    @app.post("/api/resources/voices/rename")
    def api_voices_rename():
        body = bottle.request.json or {}
        old, new = _safe_name(body.get("old")), _safe_name(body.get("new"))
        if not old or not new:
            return _json({"ok": False, "error": "invalid name"})
        ok, err = _safe_resource_op("voices", old, "rename", new)
        return _json({"ok": ok, "error": err})

    _lives_path = os.path.join(home_dir, "local_lives.json")

    _LIVE_FIELDS = ("name", "platform", "type", "status", "date")
    _LIVE_LIMIT = {"name": 60, "platform": 30, "type": 30, "status": 30, "date": 30}

    def _sanitize_live(item):
        """直播项白名单字段化,阻断经 local_lives.json 的持久化注入。"""
        if not isinstance(item, dict):
            return None
        out = {}
        for k in _LIVE_FIELDS:
            v = item.get(k)
            if v is None:
                v = ""
            v = str(v).replace("<", "&lt;").replace(">", "&gt;")[:_LIVE_LIMIT[k]]
            out[k] = v
        return out

    @app.get("/api/lives")
    def api_lives_list():
        if os.path.isfile(_lives_path):
            try:
                with open(_lives_path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("items"), list):
                    # [R-2-004/T-066] 契约 C7:读路径按写入端 500 上限截断最近 500 条(列表尾部);
                    # ≤500 条时切片原样返回,缺省行为不变
                    return _json({"items": [_sanitize_live(x) for x in data["items"][-500:]]})
            except (ValueError, OSError):
                pass
        return _json({"items": []})

    @app.post("/api/lives/save")
    def api_lives_save():
        early = _body_limit_exceeded()             # [R-2-007/T-067] 超限稳定 413+JSON 错误体
        if early:
            return early
        items = (bottle.request.json or {}).get("items")
        if not isinstance(items, list) or len(items) > 500:
            return _json({"ok": False, "error": "items required"})
        clean = [x for x in (_sanitize_live(x) for x in items) if x]
        tmp = _lives_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"items": clean}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, _lives_path)
        return _json({"ok": True})

    @app.get("/api/engine/config")
    def api_engine_config_get():
        path = os.path.join(home_dir, "local_config.json")
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return _json(json.load(f))
            except (ValueError, OSError):
                pass
        return _json({"banned_words": []})

    @app.post("/api/engine/config")
    def api_engine_config_set():
        body = bottle.request.json or {}
        words = body.get("banned_words")
        if not isinstance(words, list) or not all(isinstance(w, str) for w in words):
            return _json({"ok": False, "error": "banned_words list required"})
        words = [w.strip() for w in words if w.strip()][:200]
        path = os.path.join(home_dir, "local_config.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"banned_words": words}, f, ensure_ascii=False)
        os.replace(tmp, path)
        return _json({"ok": True, "count": len(words)})

    _LLM_BASE = {
        "DeepSeek": "https://api.deepseek.com/chat/completions",
        "智谱 AI": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "MiniMax": "https://api.minimax.chat/v1/text/chatcompletion_v2",
        "火山方舟": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        "千问": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "百度千帆": "https://qianfan.baidubce.com/v2/chat/completions",
        "讯飞 MaaS": "https://spark-api-open.xf-yun.com/v1/chat/completions",
        "OpenAI": "https://api.openai.com/v1/chat/completions",
        "硅基流动": "https://api.siliconflow.cn/v1/chat/completions",
    }

    @app.post("/api/llm/chat")
    def api_llm_chat():
        early = _body_limit_exceeded()             # [R-2-007/T-067] 超限稳定 413+JSON 错误体
        if early:
            return early
        body = bottle.request.json or {}
        prompt = (body.get("prompt") or "").strip()
        system = (body.get("system") or "").strip()
        provider = body.get("provider") or "DeepSeek"
        model = (body.get("model") or "").strip()
        key = (body.get("key") or "").strip()
        if not prompt:
            return _json({"ok": False, "error": "prompt required"})
        if not key:
            return _json({"ok": False, "error": "未配置 APIKey,请在大模型页保存"})
        if not model:
            return _json({"ok": False, "error": "model required"})
        base = _LLM_BASE.get(provider)
        if not base:
            return _json({"ok": False, "error": "未知提供商: " + provider})
        messages = ([{"role": "system", "content": system}] if system else []) \
            + [{"role": "user", "content": prompt}]
        payload = json.dumps({"model": model, "messages": messages, "stream": False}).encode()
        req = _urlreq.Request(
            base, data=payload, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + key})
        try:
            with _urlreq.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
            text = data["choices"][0]["message"]["content"]
            return _json({"ok": True, "text": text})
        except _urlerr.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            return _json({"ok": False, "error": "上游 %s: %s" % (e.code, detail)})
        except Exception as e:  # 网关层统一兜底,错误原文回传前端
            return _json({"ok": False, "error": str(e)})

    @app.get("/api/engine/models")
    def api_engine_models():
        items = []
        if os.path.isdir(models_dir):
            # [资源完整性 2026-09-20] 增补 live.mp4 在场过滤(与 /api/engine/start
            # 及 works 生成门的消费面一致,只收紧):库内确有仅 model.ini 的残缺
            # 模特(oracle 带入),曾致下拉默认选中即"模特资源不存在或不完整"。
            items = sorted(d for d in os.listdir(models_dir)
                           if os.path.isfile(os.path.join(models_dir, d, "model.ini"))
                           and os.path.isfile(os.path.join(models_dir, d, "live.mp4")))
        return _json({"items": items})              # [R-1D-017/T-061] 裸 json.dumps 改 _json,Content-Type=application/json(契约 C1);成功体形态不变

    @app.get("/api/engine/voices")
    def api_engine_voices():
        items = []
        vdir = os.path.join(home_dir, "voices")
        if os.path.isdir(vdir):
            items = sorted(d for d in os.listdir(vdir)
                           if os.path.isdir(os.path.join(vdir, d))
                           and any(f.lower().endswith(".wav")
                                   for f in os.listdir(os.path.join(vdir, d))))
        return _json({"items": items})              # [R-1D-017/T-061] 同上(契约 C1);成功体形态不变

    @app.get("/api/engine/status")
    def api_engine_status():
        return _json(local_engine.status())         # [R-1D-017/T-061] 手工拼 JSON 改统一 _json(契约 C1);字段签名不变(§1.1 冻结)

    @app.post("/api/engine/start")
    def api_engine_start():
        # [R-1D-008/T-031](含 R-1A-006) 开播前资源校验:不满足即回 400+ok:false 中文错误,
        # 不进入 local_engine.start_live(不 spawn 渲染子进程);校验只收紧入口,不改引擎。
        req = bottle.request.json or {}
        model, voice = req.get("model"), req.get("voice")
        if not model or not voice:
            return _json_err(400, "资源不存在")
        model, voice = _safe_name(model), _safe_name(voice)   # [T-031] model/voice 过 _safe_name 同款白名单(契约 §1.3 只收紧不放松)
        if not model or not voice:
            return _json_err(400, "资源名称非法")
        model_dir = os.path.join(models_dir, model)
        if not os.path.isdir(model_dir):
            return _json_err(400, "模特资源不存在")
        # face 即 local_engine._render_child 所需 <models_dir>/<model>/live.mp4
        if not os.path.isfile(os.path.join(model_dir, "live.mp4")):
            return _json_err(400, "模特资源不完整")
        voice_dir = os.path.join(home_dir, "voices", voice)   # 与 local_engine 的 voice_dir 同源
        if not os.path.isdir(voice_dir):
            return _json_err(400, "音色资源不存在")
        if not any(f.lower().endswith(".wav") for f in os.listdir(voice_dir)):
            return _json_err(400, "音色资源不完整(缺少 wav)")
        return _json(local_engine.start_live(model, voice))   # [R-1D-017/T-061] 契约 C1

    @app.post("/api/engine/speak")
    def api_engine_speak():
        req = bottle.request.json or {}
        text = req.get("text", "")
        _cfgp = os.path.join(home_dir, "local_config.json")
        try:
            with open(_cfgp, encoding="utf-8") as _f:
                _bw = json.load(_f).get("banned_words") or []
            if _bw:
                text = remove_banned_words(text, "|".join(_bw))  # 原版函数签名:竖线拼接字符串,非列表
        except (ValueError, OSError):
            pass
        return _json(local_engine.speak(text))      # [R-1D-017/T-061] 契约 C1;成功体形态不变

    @app.post("/api/engine/stop")
    def api_engine_stop():
        # [R-1D-017/T-061](含 R-2-006) 统一 _json(契约 C1);按"未在播即失败"收口(批2验收断言):
        # 空闲(running=false)下播回 {"ok":false,"error":"当前未在播"},不再恒 ok:true;
        # 在播时行为不变(引擎结果原样透传);下播异常回 ok:false+中文 error(契约 C2),不再以 500 HTML 冒出
        try:
            if not local_engine.status().get("running"):
                return _json({"ok": False, "error": "当前未在播"})
            return _json(local_engine.stop_live())
        except Exception as e:
            return _json({"ok": False, "error": "下播失败: %s" % e})

    @app.post("/api/engine/record")
    def api_engine_record():
        req = bottle.request.json or {}
        # [R-1D-009/T-032](含 R-2-005) seconds 非法("abc"/null 等)统一 400+JSON 错误体,不再 500 HTML(契约 C2);
        # 缺省 10 与 clamp 3..60 语义保持不变(契约 §1.1)
        try:
            sec = min(max(int(req.get("seconds", 10)), 3), 60)
        except (TypeError, ValueError):
            return _json_err(400, "seconds invalid")
        name = local_engine._srs_stream_name()
        src = "http://127.0.0.1:%d/%s.flv" % (local_engine.SRS_HTTP, name)
        out = os.path.join(temp_dir, "live_rec.mp4")
        ffmpeg = os.path.join("bin", "ffmpeg.exe")
        try:
            subprocess.run([ffmpeg, "-y", "-v", "error", "-rw_timeout", "8000000",
                            "-i", src, "-t", str(sec), "-c", "copy",
                            "-movflags", "+faststart", out],
                           timeout=sec * 4)
            ok = os.path.isfile(out) and os.path.getsize(out) > 1000
        except subprocess.TimeoutExpired:
            ok = False
        return _json({"ok": ok, "url": "/media/live_rec.mp4",
                           "error": None if ok else "录制失败(未开播?)"})

    # ---- [训练接入 2026-09-20] 一键训练端点(模型见 modules/core/train_service.py) ----
    @app.post("/api/train/upload")
    def api_train_upload():
        up = bottle.request.files.get("file")
        if up is None:
            return _json_err(400, "缺少文件字段 file")
        kind = bottle.request.forms.get("kind") or ""
        res = train_service.save_upload(kind, up.filename, up.file)
        return _json(res)

    @app.post("/api/train/start")
    def api_train_start():
        req = bottle.request.json or {}
        res = train_service.start(req.get("type") or "", req.get("path") or "",
                                  req.get("refer_text") or "",
                                  req.get("name") or "")   # [命名训练 2026-09-20] 可选资源名(即目录名,中文合法)
        return _json(res)

    @app.get("/api/train/status")
    def api_train_status():
        return _json(train_service.status())

    # ---- [短视频接入 2026-09-20] 本地生成端点(模型见 modules/core/works_service.py) ----
    @app.post("/api/works/generate")
    def api_works_generate():
        req = bottle.request.json or {}
        return _json(works_service.generate(req.get("text") or "",
                                            req.get("model") or "",
                                            req.get("voice") or ""))

    @app.get("/api/works/status")
    def api_works_status():
        return _json(works_service.status())

    @app.get("/api/works/list")
    def api_works_list():
        return _json({"items": works_service.list_works()})

    @app.get("/works/<fname>")
    def api_works_file(fname):
        return bottle.static_file(fname, root=works_service.WORKS_DIR,
                                  mimetype="video/mp4")

    @app.get("/media/<fname>")
    def api_media(fname):
        return bottle.static_file(fname, root=temp_dir)

    app.run(host="127.0.0.1", port=bottle_port, quiet=True)  # [自建前端安全整改] 原为0.0.0.0;本地控制台API(删除/LLM代理)仅对本机开放                       # py=2241 GetAttr 'run' + kwargs(PyDict_SetItem 序)host/port/quiet [ST '0.0.0.0'][全局 bottle_port][True];返回 None


def start_static():                                                             # [FUN_18006c660 py≈2244][U008-a 重建]
    app = bottle.Bottle()                                                       # py=2245 GetModuleGlobalName('bottle')→GetAttr 'Bottle' 无参调用
    @app.get("/local_video")                                                    # py=2247 GetAttr(app,'get') 1 位置参 [ST '/local_video']
    def local_video():                                                          # py=2248 [PyCode_New firstlineno=0x8c8][varnames('filename',)=DAT_18009d408][qualname 'start_static.<locals>.local_video']
        filename = bottle.request.query["file"]                                 # py=2249 下标访问(PyDict 快速路径/PyObject_GetItem;缺参 KeyError→bottle 500)[ST 'file']
        return bottle.static_file(filename, root=temp_dir, mimetype="video/mp4")    # py=2250 GetAttr 'static_file' 1 位置参 + kwargs(PyDict_SetItem 序)root/mimetype [ST 'video/mp4'][全局 temp_dir]
    app.run(host="127.0.0.1", port=static_port, quiet=True)  # [自建前端安全整改] 同上                       # py=2252 GetAttr 'run' + kwargs host/port/quiet [全局 static_port];返回 None


def start_gui():                                                                # [FUN_18006d430 py=2258][varnames api,url,debug + cell on_shown]
    global window
    # [自建前端 2026-09-19] 不再启动 app_mqtt.start(其内部走 verify 授权 + MQTT);
    # SRS/TTS/渲染由 local_engine 按需拉起。
    threading.Thread(target=start_bottle, daemon=True).start()                  # py=2262
    threading.Thread(target=start_static, daemon=True).start()                  # py=2263
    api = Api()                                                                 # py=2265 GetModuleGlobalName('Api') 无参调用
    import time as _t
    url = "http://127.0.0.1:%d/local_ui?boot=%d" % (bottle_port, int(_t.time()))  # [自建前端] 本地控制台页面(时间戳击穿 WebView2 缓存)
    window = webview.create_window(                                             # py=2267 webview.create_window(kw…)
        title="哈基米数字人",                                             # [自建前端品牌] 重建版品牌名                                                      # py=2268 [ST '数字人系统']
        url=url,
        min_size=(1600, 1000),                                                  # [cached DAT_18009bda0=(PyLong(0x640), PyLong(1000))]
        js_api=api,
        confirm_close=True,
        localization={"global.quitConfirmation": "确定要退出么?"},              # py=2273 PyDict_New+SetItem → PyDict_Copy
    )
    window.events.closed += on_closed                                           # py=2275 PyNumber_InPlaceAdd(events.closed, on_closed) + SetAttr
    set_train_window(window)                                                    # py=2276
    debug = True if os.path.exists(".debug") else False                         # py=2279 exists('.debug') → IsTrue → C bint(start(debug=) 时转 True/False)

    def on_shown():                                                             # [FUN_18006cd20 py=2282][varnames icon_manager,launched_from_exe]
        icon_manager = WindowIconManager()                                      # py=2283 GetModuleGlobalName('WindowIconManager') 无参调用
        launched_from_exe = os.environ.get("LAUNCHED_FROM_EXE")                 # py=2285 os.environ.get('LAUNCHED_FROM_EXE')
        if launched_from_exe:                                                   # py=2286 IsTrue
            icon_manager.set_icon_from_exe(window, launched_from_exe)           # py=2287 PyTuple_New(2)=(window 全局, launched_from_exe);E1 G2 首轮:oracle 传 2 参

    window.events.closed += on_closed                                           # py=2289 第二次 closed += on_closed(pyd 如此,原样保留)
    window.events.shown += on_shown                                             # py=2290
    webview.start(debug=debug, private_mode=False)                              # py=2291 kw debug=True/False, private_mode=False


if __name__ == "__main__":                                                      # [exec py=2294] PyUnicode_Equals(__name__,'__main__')
    start_gui()                                                                 # [exec py=2295]
