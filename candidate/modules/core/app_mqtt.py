# -*- coding: utf-8 -*-
"""modules.core.app_mqtt — R034 T4R 全量语义重建(源 app_mqtt.cp310-win_amd64.pyd,M1.7 二轮校准版)

证据标注约定(每处修正均可回溯二进制):
- [pyd raw L###] = evidence/modules/core__app_mqtt/static/pseudocode/ 下 Ghidra 反编译行号
  (pymod_exec/py_*/init_cached_constants,复制自 pyd_static/core__app_mqtt/pseudocode/)
- [pyc ###]     = szr2026_out/m1_calibration_truth/app_mqtt_pycdas.txt 行号(pyc 字节码,权威真相)
- [ST slot]     = evidence/modules/core__app_mqtt/static/stringtab.json 槽位
- [py=N]        = 反编译回溯 py 行号(filt2 标记)
- [版本分歧]     = pyd(Apr 2025) 与 pyc(Mar 2026) 编译自不同版本源码;重建保留 pyd 证据版并注记

G2 捕获口径(2026-09-11,R034 补实):用例经 fixture 注入的 szr_mqtt_env 助手在
**双侧同条件**下打桩 torch.cuda.is_available → True 绕过导入门控,对 18 个契约函数
做函数级真差分;MQTT broker/HTTP 全部走本地 stub 端口与记录型 fake,绝不真连。
无打桩的导入用例保持门控行为(横幅/GBK 输出以 capture 期 PYTHONIOENCODING=utf-8 为准)。
"""

import gc  # [pyc 3569-3570 IMPORT_NAME gc][pyd pymod_exec gc 导入块] empty_cache 使用 gc.collect

from hashlib import md5  # noqa: F401  [pyc 3573-3576 from hashlib import md5]
import multiprocessing   # [pyc 3578]
import pathlib           # [pyc 3581]
from queue import Queue  # [pyc 3584-3586]
import socket            # [pyc 3589]
import platform          # [pyc 3592]
import subprocess        # [pyc 3595]
import os                # [pyc 3598]
import json              # [pyc 3601]
import sys               # [pyc 3604]
import time              # [pyc 3607]
from colorama import Fore, Style  # [pyc 3610-3615]
import paho.mqtt.client as mqtt   # [pyd 导入表 'paho.mqtt.client'+名字绑定 mqtt][pyc 同构]
from easydict import EasyDict as edict  # [pyd pymod_exec easydict/EasyDict/edict 三步][pyc 同]
import threading          # [pyc 3625]
import requests           # [pyc 3628]
import torch              # [pyc 3631]

if not torch.cuda.is_available():          # [pyd pymod_exec py=20 IsTrue][pyc 3666-3672]
    print("缺少 nvidia显卡，当前电脑不可用")  # [ST 0x1800413a8][pyc 3675]
    time.sleep(10)                          # [pyd init_cached PyTuple_Pack(1,PyLong(10))][pyc 3678]
    sys.exit()                              # [pyd pymod_exec sys.exit 无参调用][pyc 3681]

from modules.core.app_infer import run_process  # [pyc 3691][pyd PyList_New(1)]

# [版本分歧] config 导入名单: pyd=25 名(PyList_New(0x19), pymod_exec 证据, 含 luxtts_port/
# omnivoice_port/gpu_memory_utilization); pyc=22 名(pycdas 3713 无此三名)。此处按 pyd 证据保留。
from modules.core.config import (      # [pyd py=26-52]
    mqtt_host, mqtt_port, home_dir, gpt_port, vsa_port, tts_port, vc_port,
    api_host, api_key, infer_count, srs_host, srs_port, strtobool,
    allow_accounts, srs_cloud, srs_proxy, srs_proxy_port, srs_rtc_port,
    srs_rtc_proxy_port, allow_host, index_port, voxcpm_port, luxtts_port,
    omnivoice_port, gpu_memory_utilization,
)

# [版本分歧] util 导入名单: pyd=8 名(PyList_New(8), 含 send_event_to_umami); pyc=7 名
# (pycdas 3722 无 send_event_to_umami——但 pyd start py=788 调用 Timer(send_event_to_umami))。
from modules.core.util import (         # [pyd py=53]
    WindowsPerformanceOptimizer, check_admin_privileges,
    get_windows_computer_id, print_red, print_yellow,
    send_event_to_umami, setInterval, setTimeout,
)

from modules.core.util import kill_port, verify  # [pyc 3727-3730][pyd py=63]

import urllib3                # [pyc 3760][pyd py=64]
urllib3.disable_warnings()    # [pyc 3762-3764][pyd py=66]

# [COM-D002] 遥测总开关:**默认 False(disabled by default / fail-closed)**。
# 候选不内置任何第三方分析端点;即使该开关被显式打开,被调度的
# send_event_to_umami 也已是惰性空壳(零出网、无站点标识),不会发送任何数据。
TELEMETRY_ENABLED = False


def srs_run():                                                                       # py 69
    srs_home = os.path.realpath("srs")                                               # [pyc 275-278][pyd raw L128 realpath("srs")]
    command = f"{os.path.join(srs_home, 'objs', 'srs')} -c {os.path.join(srs_home, 'console.conf')}"  # [pyc 281-308][pyd raw L285-541 join(objs,srs)+' -c '+join(console.conf)]
    subprocess.call(command, shell=platform.system() != "Windows",                    # [pyc 296-306 COMPARE_OP 3(!=)][pyd raw L720 RichCompare(op=3)=Py_NE — 一轮误判 Py_EQ,二轮修正]
                    stdout=subprocess.DEVNULL, cwd=srs_home)                          # [pyc 306-312][pyd raw L773-799]


def gptsovits_run():                                                                  # py 80
    # [版本分歧] pyd 有前置存在性检查(sub_180002190 raw: py=82 exists('modules/gptsovits'),
    # py=83 print_red(f'缺少 v3 声音模块 ({module_dir} 目录不存在)'), ST 0x180041d68/0x180041db0;
    # py 行 84 为语句行且 c=2943→2975 无错误标记=无衅语句,判为 return);pyc 无此块。
    # 重建按 pyd 证据保留,评分子集计分时剔除。
    module_dir = "modules/gptsovits"                                                  # [pyd varnames(module_dir,python_path,command)][ST 'modules/gptsovits']
    if not os.path.exists(module_dir):                                                # [pyd filt2 py=82 exists(plVar1)]
        print_red(f"缺少 v3 声音模块 ({module_dir} 目录不存在)")                        # [pyd raw 3-part f-string c=2906-2943][ST 0x180041d68='缺少 v3 声音模块 (']
        return                                                                         # [pyd py行84 空缺语句+c=2943-2975 无错误标记→无衅 return(见上注)]
    python_path = os.path.realpath(module_dir + "/runtime/python.exe")                 # [pyd filt2 PyUnicode_Concat(module_dir,'/runtime/python.exe')][pyc 246-250 单字面量=常量折叠,语义同]
    if not os.path.exists(python_path):                                               # [pyc 252-258][pyd filt2 py=86]
        python = "python"                                                             # [pyc 260][pyd STR('python')]
    else:
        python = python_path
    command = f"{python} start_api.py --port {gpt_port}"                               # [pyc 264-274][pyd filt2 3 段 StrBuild]
    subprocess.call(command, shell=platform.system() != "Windows",                     # [pyc 292-306 COMPARE_OP 3(!=)][pyd raw 同 srs_run op=3]
                    cwd=os.path.join(module_dir))                                      # [pyc 296-308 join 单参=恒等;pyd raw 无 join → cwd=module_dir,行为等价]


def vsa_run():                                                                        # py 97
    # [版本分歧] 同 gptsovits_run:pyd 有 prelude('缺少 v2 模块 (' ST 0x180041380);pyc 无。
    module_dir = "modules/vsa"                                                        # [pyd varnames][ST 'modules/vsa']
    if not os.path.exists(module_dir):                                                # [pyd filt2 py=99]
        print_red(f"缺少 v2 模块 ({module_dir} 目录不存在)")                            # [pyd py=100][ST 0x180041380]
        return                                                                         # [同 gptsovits_run 判据]
    python_path = os.path.realpath(module_dir + "/runtime/python.exe")                 # [pyd filt2 py=102 Concat]
    if not os.path.exists(python_path):                                               # [pyc 322-328][pyd py=103]
        python = "python"
    else:
        python = python_path
    command = f"{python} start_api.py --port {vsa_port}"                               # [pyc 332-342]
    subprocess.call(command, shell=platform.system() != "Windows",                     # [pyc COMPARE_OP 3(!=)]
                    cwd=os.path.join(module_dir))                                      # [pyc join 单参]


def qftts_run():                                                                      # py 114
    # [版本分歧] prelude 同上('缺少 v6 声音模块 (' ST 0x180041760)。
    module_dir = "modules/qftts"                                                      # [pyd varnames(module_dir,python_path,model_name,model_dir,command)]
    if not os.path.exists(module_dir):                                                # [pyd filt2 py=116]
        print_red(f"缺少 v6 声音模块 ({module_dir} 目录不存在)")                        # [pyd raw DAT_180041760]
        return
    python_path = os.path.realpath(module_dir + "/runtime/python.exe")                 # [pyd filt2 py=119]
    if not os.path.exists(python_path):                                               # [pyc 404-410]
        python = "python"
    else:
        python = python_path
    model_name = "zipvoice_distill"                                                   # [pyc 429][pyd raw L452-453]
    model_dir = "pretrained_models"                                                   # [pyc 430][pyd raw L455-456]
    command = f"{python} start_api.py --model-name {model_name} --model-dir {model_dir} --port {tts_port}"  # [pyc 434-460 7 段 BUILD_STRING][pyd raw PyTuple(7)]
    subprocess.call(command, shell=platform.system() != "Windows",                     # [pyc COMPARE_OP 3(!=)]
                    cwd=os.path.join(module_dir))                                      # [pyc join 单参]


def vc_run():                                                                         # py 138
    # [版本分歧] prelude 同上('缺少 v5 声音模块 (' ST 0x180041a48)。
    module_dir = "modules/vc"                                                         # [pyd varnames]
    if not os.path.exists(module_dir):                                                # [pyd filt2 py=140]
        print_red(f"缺少 v5 声音模块 ({module_dir} 目录不存在)")                        # [pyd raw DAT_180041a48]
        return
    python_path = os.path.realpath(module_dir + "/runtime/python.exe")                 # [pyd filt2 py=143]
    if not os.path.exists(python_path):                                               # [pyc 474-480]
        python = "python"
    else:
        python = python_path
    command = f"{python} start_api.py --port {vc_port} --diffusion-steps 20"            # [pyc 484-502][ST ' --diffusion-steps 20']
    subprocess.call(command, shell=platform.system() != "Windows",                     # [pyc COMPARE_OP 3(!=)]
                    cwd=os.path.join(module_dir))


def indextts_run():                                                                   # py 155
    # [版本分歧] prelude 同上('缺少 v7 声音模块 (' ST 0x1800415f8);且 pyd 有 env 注入块
    # (raw: STR 'os'/'environ'/'copy'/'env' + gpu_memory_utilization),pyc 无——按 pyd 证据保留。
    module_dir = "modules/indextts"                                                   # [pyd varnames(module_dir,python_path,command,env)]
    if not os.path.exists(module_dir):                                                # [pyd filt2 py=157]
        print_red(f"缺少 v7 声音模块 ({module_dir} 目录不存在)")                        # [pyd raw DAT_1800415f8]
        return
    python_path = os.path.realpath(module_dir + "/runtime/python.exe")                 # [pyd filt2 py=160]
    if not os.path.exists(python_path):                                               # [pyc 551-557]
        python = "python"
    else:
        python = python_path
    command = f"{python} start_api.py --port {index_port}"                             # [pyc 561-571]
    env = os.environ.copy()                                                            # [pyd STR 'environ'/'copy' + F_Call]
    env["gpu_memory_utilization"] = gpu_memory_utilization                             # [pyd PyDict_SetItem('gpu_memory_utilization')]
    subprocess.call(command, shell=platform.system() != "Windows",                     # [pyc COMPARE_OP 3(!=)]
                    cwd=os.path.join(module_dir), env=env)                             # [pyd kw 'env']


def voxcpm_run():                                                                     # py 176
    # [版本分歧] prelude 同上('缺少 v8 声音模块 (' ST 0x1800418e0)。
    module_dir = "modules/voxcpm"                                                     # [pyd varnames]
    if not os.path.exists(module_dir):                                                # [pyd filt2 py=178]
        print_red(f"缺少 v8 声音模块 ({module_dir} 目录不存在)")                        # [pyd raw DAT_1800418e0]
        return
    python_path = os.path.realpath(module_dir + "/runtime/python.exe")                 # [pyd filt2 py=181]
    if not os.path.exists(python_path):                                               # [pyc 628-634]
        python = "python"
    else:
        python = python_path
    command = f"{python} start_api.py --port {voxcpm_port}"                             # [pyc 638-648]
    subprocess.call(command, shell=platform.system() != "Windows",                     # [pyc COMPARE_OP 3(!=)]
                    cwd=os.path.join(module_dir))


# [版本分歧] luxtts_run / omnivoice_run:pyd 存在(方法表 omnivoice meth=0x18000adf0、
# qualname 'app_mqtt.luxtts_run'/'app_mqtt.omnivoice_run';模块级字典设置 py=193/210),
# pyc 完全无此二函数(0 引用)。按 run_* 家族模板 + ST 消息('缺少 v9 声音模块 (' 0x180041358 /
# '缺少 v10 声音模块 (' 0x180041c28)+ luxtts_port/omnivoice_port 重建,评记 N/A(真相侧缺失)。
def luxtts_run():                                                                     # py 193 (仅 pyd)
    module_dir = "modules/luxtts"
    if not os.path.exists(module_dir):
        print_red(f"缺少 v9 声音模块 ({module_dir} 目录不存在)")
        return
    python_path = os.path.realpath(module_dir + "/runtime/python.exe")
    if not os.path.exists(python_path):
        python = "python"
    else:
        python = python_path
    command = f"{python} start_api.py --port {luxtts_port}"
    subprocess.call(command, shell=platform.system() != "Windows",
                    cwd=os.path.join(module_dir))


def omnivoice_run():                                                                  # py 210 (仅 pyd)
    module_dir = "modules/omnivoice"
    if not os.path.exists(module_dir):
        print_red(f"缺少 v10 声音模块 ({module_dir} 目录不存在)")
        return
    python_path = os.path.realpath(module_dir + "/runtime/python.exe")
    if not os.path.exists(python_path):
        python = "python"
    else:
        python = python_path
    command = f"{python} start_api.py --port {omnivoice_port}"
    subprocess.call(command, shell=platform.system() != "Windows",
                    cwd=os.path.join(module_dir))


def is_port_in_use(port):                                                             # py 227
    # 一轮漏读异常分支;二轮证据:pyd raw sub_18000c160 L479-480 出现 _Py_TrueStruct 装载返回
    # (bare except 无需名字匹配,与串表无 'OSError'/'Exception' 一致);pyc 764-830 权威。
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:                       # [pyc 742-752][pyd raw L81-99]
        try:                                                                            # [pyc 754 SETUP_FINALLY]
            s.bind(("0.0.0.0", port))                                                  # [pyc 757-768][ST 0x180041aa8 '0.0.0.0']
            return False                                                                 # [pyc 770-786 __exit__(None×3)+LOAD False]
        except:                                                                          # [pyc 788-800 POP_EXCEPT+__exit__(None×3)][pyd raw L479-480 True 返回]
            return True                                                                   # [pyc 810-812 LOAD_CONST True]


def clear_screen():                                                                   # py 236
    sys.stdout.write("\x1bc")                                                          # [pyc 843-850][ST 0x1800412d0 '\x1bc']
    sys.stdout.flush()                                                                 # [pyc 855-860]


def empty_cache():                                                                    # py 241
    # 二轮修正:pyc 883-905 SETUP_FINALLY+异常吞噬(POP_EXCEPT+return None);
    # pyd raw sub_18000d1e0 无名字匹配调用(bare except)+双 None 返回路径,一致。
    try:                                                                                # [pyc 885]
        gc.collect()                                                                    # [pyc 886-889]
        torch.cuda.empty_cache()                                                       # [pyc 892-897]
    except:                                                                             # [pyc 905-910 POP_EXCEPT]
        pass


runner_dict = {}                                                                      # py 249 [pyc 3800-3802 BUILD_MAP+STORE][pyd PyDict_New]


def do_message(msg):                                                                  # py 252
    # 二轮全量重建。证据:pycdc 渲染完整(无 incomplete);pyd sub_18000d710 STR 分派序列
    # 与 pyc 常量表逐一对应;get('fast', False) 缺省元组=DAT_180042998(raw L3482/L3858);
    # '请求失败，请稍后再试'=ST 0x180041cd8;顺序 if(无 elif)由 pycdas 平铺跳转证实。
    if msg.type == "live_change":                                                       # [pyd STR 序][pyc 929-938]
        live = msg.live
        device = msg.device
        if device.inferId not in runner_dict:                                            # [pyd STR 'runner_dict'+'inferId']
            print_red("请求失败，请稍后再试")                                             # [ST 0x180041cd8]
            return
        p = runner_dict[device.inferId]                                                  # [pyd BINARY_SUBSCR]
        p.runner_queue.put_nowait(("live", live))                                        # [pyd STR 'put_nowait'+'live']
    if msg.type == "guard_change":
        option = msg.option
        device = msg.device
        if device.inferId not in runner_dict:
            print_red("请求失败，请稍后再试")
            return
        p = runner_dict[device.inferId]
        p.runner_queue.put_nowait(("guard", option))                                     # [pyd STR 'guard']
    if msg.type == "gen_video_timer":
        option = msg.option
        device = msg.device
        if device.inferId not in runner_dict:
            print_red("请求失败，请稍后再试")
            return
        p = runner_dict[device.inferId]
        p.runner_queue.put_nowait(("video_timer", option))                               # [pyd STR 'video_timer']
    if msg.type == "cache_time":
        cache_time = msg.cache_time
        cache_only = msg.cache_only
        device = msg.device
        if device.inferId not in runner_dict:
            print_red("请求失败，请稍后再试")
            return
        p = runner_dict[device.inferId]
        p.runner_queue.put_nowait(("cache_time", cache_time, cache_only))
    if msg.type == "live_changesite":
        device = msg.get("device")                                                       # [pyd STR 'get']
        if device.inferId not in runner_dict:
            print_red("请求失败，请稍后再试")
            return
        site = msg.get("site")
        model = msg.get("model")
        voice = msg.get("voice")
        assistVoice = msg.get("assistVoice")
        p = runner_dict[device.inferId]
        p.runner_queue.put_nowait(("changesite", site, model, voice, assistVoice))        # [pyd STR 'changesite']
    if msg.type == "live_sendmsg":
        device = msg.get("device")
        if device.inferId not in runner_dict:
            print_red("请求失败，请稍后再试")
            return
        content = msg.get("content")
        fast = msg.get("fast", False)                                                    # [pyd DAT_180042998=('fast',False) raw L3482]
        replyTo = msg.get("replyTo")
        p = runner_dict[device.inferId]
        p.runner_queue.put_nowait(("sendmsg", content, fast, replyTo))                    # [pyd STR 'sendmsg']
    if msg.type == "live_sendvoice":
        device = msg.get("device")
        if device.inferId not in runner_dict:
            print_red("请求失败，请稍后再试")
            return
        voice = msg.get("voice")
        name = msg.get("name")
        fast = msg.get("fast", False)                                                    # [pyd DAT_180042998 raw L3858]
        replyTo = msg.get("replyTo")
        p = runner_dict[device.inferId]
        p.runner_queue.put_nowait(("sendvoice", name, voice, fast, replyTo))              # [pyd STR 'sendvoice']
    if msg.type == "live_changeaudio":
        device = msg.device
        if device.inferId not in runner_dict:
            print_red("请求失败，请稍后再试")
            return
        audio_index = msg.audio_index
        p = runner_dict[device.inferId]
        p.runner_queue.put_nowait(("changeaudio", audio_index))                           # [pyd STR 'changeaudio']
    if msg.type == "live_danmu":
        device = msg.device
        if device.inferId not in runner_dict:
            print_red("请求失败，请稍后再试")
            return
        content = msg.content
        nickname = msg.nickname
        method = msg.method
        p = runner_dict[device.inferId]
        p.runner_queue.put_nowait(("danmu", method, nickname, content))                   # [pyd STR 'danmu']
    if msg.type == "live_wenda":
        device = msg.device
        if device.inferId not in runner_dict:
            print_red("请求失败，请稍后再试")
            return
        wendas = msg.wendas
        p = runner_dict[device.inferId]
        p.runner_queue.put_nowait(("wenda", wendas))                                      # [pyd STR 'wenda']
    if msg.type == "live_fenwei":
        device = msg.device
        if device.inferId not in runner_dict:
            print_red("请求失败，请稍后再试")
            return
        fenweis = msg.fenweis
        p = runner_dict[device.inferId]
        p.runner_queue.put_nowait(("fenwei", fenweis))                                    # [pyd STR 'fenwei']
    if msg.type == "vcam_change":
        device = msg.device
        if device.inferId not in runner_dict:
            print_red("请求失败，请稍后再试")
            return
        status = msg.status
        p = runner_dict[device.inferId]
        p.runner_queue.put_nowait(("vcam", status))                                       # [pyd STR 'vcam']
    if msg.type == "camera_change":
        device = msg.device
        if device.inferId not in runner_dict:
            print_red("请求失败，请稍后再试")
            return
        index = msg.index
        status = msg.status
        rotate = msg.rotate
        camera_type = msg.camera_type
        rtmp_url = msg.rtmp_url
        p = runner_dict[device.inferId]
        p.runner_queue.put_nowait(("camera", index, status, rotate, camera_type, rtmp_url))  # [pyd STR 'camera']
    if msg.type == "audio_change":
        device = msg.device
        if device.inferId not in runner_dict:
            print_red("请求失败，请稍后再试")
            return
        status = msg.status
        p = runner_dict[device.inferId]
        p.runner_queue.put_nowait(("audio", status))                                      # [pyd STR 'audio']
    if msg.type == "record_change":
        device = msg.device
        if device.inferId not in runner_dict:
            print_red("请求失败，请稍后再试")
            return
        index = msg.index
        status = msg.status
        record_type = msg.record_type
        p = runner_dict[device.inferId]
        p.runner_queue.put_nowait(("record", index, record_type, status))                 # [pyd STR 'record']
    if msg.type == "live_stop":
        device = msg.device
        if device.inferId not in runner_dict:
            print_red("请求失败，请稍后再试")
            return
        status = msg.status
        p = runner_dict[device.inferId]
        p.runner_queue.put_nowait(("stoping", status))                                    # [pyd STR 'stoping']
        return


def on_connect(deviceId, client, userdata, flags, rc, properties):                     # py 434
    if rc == 0:                                                                        # [pyd raw sub_1800163e0 L62-111 与 0 比较][pyc 1695-1701]
        print("Connected MQTT Broker!")                                                # [pyc 1705][ST 0x180041c88]

        public_ip = requests.get(f"{api_host}/api/public/get-ip")                       # [pyd raw L530-532 + Concat][pyc 1719-1726]
        public_ip = public_ip.text.strip()                                              # [pyd raw L556-560 .text.strip()][pyc 1729-1732]
        print(f"Public Ip {public_ip}")                                                # [pyd raw L396 Concat('Public Ip ')][pyc 1736-1740]

        def func():                                                                    # py 444
            if client.is_connected():                                                  # [pyd filt2 py=445]
                used_devices = []                                                      # [pyd filt2 py=446 PyList_New(0)]
                for key, p in runner_dict.items():                                      # [pyd filt2 py=447 items+GetIter]
                    if not p["runner_process"].is_alive():                              # [pyd filt2 py=448][pyc 1811-1814 POP_JUMP_IF_TRUE→continue]
                        continue
                    used_devices.append(p["device"].id)                                 # [pyd filt2 py=450 GetAttr device→id+LApp][pyc 1822-1830]
                gpu_count = torch.cuda.device_count()                                   # [pyd filt2 py=452]
                srs = f"{srs_host}:{srs_port}"                                          # [pyd filt2 py=453 3 段 f-string]
                if srs_proxy:                                                           # 二轮修正:[pyd raw sub_180014270 L747-756 FUN_180035b90(IsTrue)+iVar5==0 跳分支][pyc 1846-1850]
                    srs = f"{srs_proxy}:{srs_proxy_port}"                               # [pyd filt2 py=455 StrBuild(srs_proxy,':',srs_proxy_port)]
                client.subscribe(f"live/{deviceId}/#")                                  # [pyd filt2 py=456][pyc 1854-1862]
                client.publish("device", json.dumps({                                   # [pyd filt2 py=457 STR 'device'+dumps]
                    "type": "live_device_connected",
                    "deviceId": deviceId,
                    "name": socket.gethostname(),
                    "gpu_count": int(gpu_count),                                        # 二轮修正:[pyd raw L1134 PyNumber_Long][pyc 1873]
                    "infer_count": int(infer_count),                                    # 二轮修正:[pyd raw L1202 PyNumber_Long][pyc 1878]
                    "public_ip": public_ip,
                    "srs": srs,
                    "srs_cloud": srs_cloud,
                    "allow_accounts": allow_accounts,
                    "used_devices": used_devices,
                    "allow_host": allow_host,
                }))
                print("Connected Live online")                                          # [pyd raw L1530 DAT_180042650=('Connected Live online',)][pyc 1885]
                return

        setTimeout(func, 1)                                                             # [pyd raw sub_1800163e0 L488-501 PyTuple(2)+DAT_180042bd8=1][pyc 1893-1896]
    else:
        print("Failed to connect, return code %d\n", rc)                                # [pyd raw L120-124 DAT_180041b98+param_6][pyc 1900-1906]


def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):         # py 483
    print("MQTT Discounted", reason_code)                                               # [pyd raw py_on_disconnect L125-136][pyc 1920-1928]


def on_message(context, mqtt_queue, sys_queue, client, userdata, msg):                  # py 488
    # 二轮全量重建。证据:pycdas 1986-2606 完整字节码;pyd sub_180017860 STR 序列
    # ('/api/checklocallive'/'hq,live,'/'srs_api'/rtc 系列等)与之一一对应。
    topic = msg.topic                                                                    # [pyc 1990-1993]
    pyload = msg.payload.decode("utf-8")                                                  # [pyc 1995-2000]
    print(f"{Fore.YELLOW}{topic}{Style.RESET_ALL}",                                       # 二轮修正:单 print 双 f-string 参
          f"{Fore.YELLOW}{pyload}{Style.RESET_ALL}")                                      # [pyc 2002-2022 BUILD_STRING×3+CALL 2][pyd raw L834 PyTuple_New(2)]
    app_topic = ""                                                                       # [pyc 2024-2026][ST 0x180041888 '']
    app_host = ""
    topics = topic.split("/")                                                            # [pyc 2032-2040]
    if len(topics) == 3 and srs_cloud:                                                    # [pyc 2042-2056 len/CMP(==,2)/srs_cloud][pyd raw L921 PyObject_Size]
        app_host = topics[len(topics) - 1]                                                # [pyc 2058-2072 Size-1+BINARY_SUBSCR]
        app_topic = f'{app_host.replace(":", "_")}_'                                      # [pyc 2074-2090][pyd DAT_1800427c0=(':','_') raw L1001][ST '_']
        app_host = f"https://{app_host}"                                                  # [pyc 2092-2100][ST 'https://']
    msg = edict(json.loads(pyload))                                                        # [pyc 2102-2114]
    if msg.type == "live_start":                                                           # [pyc 2116-2124]
        live = msg.live
        device = msg.device
        room_url = msg.roomUrl
        client_topic = f"live/{device.deviceUid},{device.inferId}/client"                    # [pyc 2146-2166]
        print_yellow(f'live.model: {live.get("model")}')                                    # [pyc 2168-2186]
        print_yellow(f'live.voice: {live.get("voice")}')                                    # [pyc 2188-2206]
        print_yellow(f'live.assistVoice: {live.get("assistVoice")}')                        # [pyc 2208-2226]
        try:
            r = requests.post(f"{app_host or api_host}/api/checklocallive",                  # [pyc 2230-2244 JUMP_IF_TRUE_OR_POP][ST '/api/checklocallive']
                              data=json.dumps({"id": live.id}),
                              headers={"Content-Type": "application/json", "apiKey": api_key},  # [pyc 2250-2266]
                              timeout=180, verify=False)                                     # [pyc 2268-2274 常量 180/False]
            if r.status_code == 200:                                                          # [pyc 2278-2286]
                checktext = r.text
            else:
                raise Exception(r.text)                                                       # [pyc 2296-2304 RAISE_VARARGS]
        except:                                                                               # [pyc 2310-2330]
            print_red("系统验证失败")                                                          # [pyc 2317-2321]
            mqtt_queue.put_nowait((client_topic, json.dumps({"type": "notice", "message": "系统验证失败"})))  # [pyc 2325-2346]
            return
        passtext = md5(f"hq,live,{live.id}".encode("utf-8")).hexdigest()                       # [pyc 2356-2378][ST 'hq,live,']
        passtext = md5(passtext.encode("utf-8")).hexdigest()                                   # [pyc 2382-2398]
        if not checktext or checktext != passtext:                                             # [pyc 2400-2410 CMP(!=,3)]
            print_red("系统验证失败")
            mqtt_queue.put_nowait((client_topic, json.dumps({"type": "notice", "message": "系统验证失败"})))
            return
        try:
            ispc = strtobool(msg.ispc)                                                          # [pyc 2452-2460]
        except:                                                                                 # [pyc 2466-2476 ispc=False]
            ispc = False
        try:
            p = runner_dict[device.inferId]                                                      # [pyc 2480-2488]
            p.runner_process.terminate()
            p.runner_process.join()
        except:                                                                                 # [pyc 2514-2520]
            pass
        runner_dict[device.inferId] = edict(runner_process=None, runner_queue=None, ispc=ispc, device=device)  # [pyc 2522-2534 kwnames 4]
        # [I-C E2E 定谳 2026-09-11] pyd 的 on_message 此处**无** print(run_process)
        # (E2E 双侧实测:oracle 零输出、候选多一行 repr;该行来自 pyc 2546-2552,
        # 属 pyc(2026-03) 新增 —— 按"pyd 证据优先"口径删除,R034 §3 二轮校准的
        # pyc 采信在此点被 E2E 推翻)。
        runner_queue = context.Manager().Queue()                                                # [pyc 2554-2562]
        runner_process = context.Process(target=run_process,                                     # [pyc 2566-2598 kwnames target/args/daemon]
                                          args=(msg, mqtt_queue, sys_queue, runner_queue, live, device, room_url, ispc, app_host, app_topic),
                                          daemon=True)
        runner_dict[device.inferId]["runner_process"] = runner_process                            # [pyc 2602-2614]
        runner_dict[device.inferId]["runner_queue"] = runner_queue                                # [pyc 2616-2628]
        runner_process.start()
    if msg.type == "live_disconnected":                                                           # [pyc 2638-2646]
        try:
            device = msg.device
            p = runner_dict[device.inferId]
            p.runner_process.terminate()
            p.runner_process.join()
            p.device = None                                                                        # [pyc 2724-2728 STORE_ATTR]
        except:
            pass
    if msg.type == "rtc_session":                                                                  # [pyc 2742-2750]
        device = msg.device
        srs_api = f"http://{srs_host}:{srs_port}{msg.rtc_data.api}"                                 # [pyc 2760-2780]
        print("srs_api", srs_api)                                                                   # [pyc 2784-2790]
        headers = {"Content-Type": "application/json"}                                              # [pyc 2794-2800]
        try:
            session = requests.post(srs_api, data=json.dumps(msg.rtc_data), headers=headers)         # [pyc 2808-2824]
            print("srs_session", session.json())                                                    # [pyc 2828-2838]
            session = session.json()
            session["sdp"] = session["sdp"].replace(f" {srs_rtc_port} ", f" {srs_rtc_proxy_port} ")  # [pyc 2850-2878]
            topic = f"live/{device.deviceUid},{device.inferId}/client"                                # [pyc 2886-2906]
            client.publish(topic, json.dumps({"type": "rtc_session_answer", "session": session}))     # [pyc 2908-2928]
        except Exception as e:                                                                        # [pyc 2936-2944]
            print("srs_error", e)                                                                      # [pyc 2952-2958]
    try:                                                                                               # [pyc 2984-2992]
        do_message(msg)
    except Exception as e:                                                                             # [pyc 3000-3020]
        print(e)


def start(window_queue, browser_queue):                                                                 # py 641
    # 二轮按 pycdas 2652-3520 全量字节码重建;[pyd-only] 标注 = pyd 独有(pyc 无)且 pyd 证据充分处保留。
    check_admin_privileges()                                                                             # [pyc 2653-2655]
    optimizer = WindowsPerformanceOptimizer()                                                            # [pyc 2657-2659]
    optimizer.optimize_for_high_performance(realtime_priority=False)                                      # [pyc 2661-2666]
    context = torch.multiprocessing.get_context("spawn")                                                  # [pyc 2668-2674 LOAD_GLOBAL torch→ATTR multiprocessing][pyd raw 同源 GetAttr('multiprocessing')]
    mqtt_queue = context.Manager().Queue()                                                               # [pyc 2676-2682]
    sys_queue = context.Manager().Queue()                                                                # [pyc 2684-2690]
    try:                                                                                                  # [pyc 2692-2728]
        pathlib.Path(os.path.join(home_dir, "is_cloud.lock")).unlink()                                    # [pyc 2694-2704][ST 'is_cloud.lock']
    except:
        pass
    deviceId = verify("live")                                                                             # [pyc 2730-2734][pyd filt2 py=657 F_Call(verify,'live')]
    port = srs_port                                                                                       # [pyc 2736-2738]
    kill_port(port)                                                                                       # [pyc 2740-2744][pyd py=661]
    t = threading.Thread(target=srs_run, daemon=True)                                                      # [pyc 2746-2752][pyd py=662 Thread(target=…,daemon=True)]
    t.start()                                                                                              # [pyc 2764-2768]
    while not is_port_in_use(port):                                                                        # [pyc 2776-2782 POP_JUMP_IF_TRUE→180]
        time.sleep(1)                                                                                      # [pyc 2784-2790]
        print(f"checking {port}")                                                                          # [pyc 2792-2802][ST 'checking ']
    print(f"check {port} ok")                                                                              # [pyc 2808-2820][ST 'check '/' ok'][pyd 3 段 StrBuild 长度证据一致]
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)                                                 # [pyc 2824-2836]
    client.will_set("device", json.dumps({"type": "live_device_disconnected", "deviceId": deviceId}))       # [pyc 2838-2860]
    # [版本分歧] pyd py=670 will_set 首参为闭包变量('t'),pyc 为 'device' 字面量;按 pyc 记。
    def send_mqtt():                                                                                       # py 676
        # [E1 修正 2026-09-11 R034]原 M1.7 误写为 `for topic, msg in mqtt_queue.get():`
        # (对单次 get 结果做 FOR_ITER)。真相:pyc 2872-2918 是 get→UNPACK_SEQUENCE 2→
        # publish→JUMP_ABSOLUTE 回环的 `while True`(打桩双侧差分 probe1:for 版在
        # 队列首元素为 str 时抛 ValueError,oracle 无此行为)。
        while True:                                                                                         # [pyc 2872-2878 回环]
            topic, msg = mqtt_queue.get()                                                                   # [pyc 2880-2892 UNPACK_SEQUENCE 2]
            client.publish(topic, msg)                                                                      # [pyc 2894-2910]
    t = threading.Thread(target=send_mqtt, daemon=True)                                                    # [pyc 2912-2922]
    t.start()                                                                                              # [pyc 2926-2932]
    def sys_message():                                                                                      # py 685
        # [版本分歧] pyd sys_message(sub_180020000 STR 序)含 luxtts/omnivoice 分支;pyc 止于 voxcpm。
        while True:                                                                                         # [pyc 2968 JUMP_ABSOLUTE→2]
            msg = edict(json.loads(sys_queue.get()))                                                        # [pyc 2972-2986]
            if msg.type == "start_voice":                                                                    # [pyc 2988-2996]
                if msg.version == "gptsovits" or msg.version == "gpt-sovits":                                # [pyc 2998-3016 POP_JUMP_IF_TRUE 链]
                    threading.Thread(target=gptsovits_run, daemon=True).start()
                if msg.version == "vsa":                                                                     # [pyc 3036-3044 平铺 if]
                    threading.Thread(target=vsa_run, daemon=True).start()
                if msg.version == "qftts":
                    threading.Thread(target=qftts_run, daemon=True).start()
                if msg.version == "vc":
                    threading.Thread(target=vc_run, daemon=True).start()
                if msg.version == "indextts":
                    threading.Thread(target=indextts_run, daemon=True).start()
                if msg.version == "voxcpm":
                    threading.Thread(target=voxcpm_run, daemon=True).start()
                # [pyd-only] luxtts/omnivoice 分支:pyd STR 序列含 'luxtts'/'luxtts_run'/'omnivoice'/'omnivoice_run'
            if msg.type == "danmu":                                                                          # [pyc 3076-3184]
                browser_queue.put_nowait(json.dumps(msg))                                                    # [pyc 3086-3098]
    t = threading.Thread(target=sys_message, daemon=True)                                                    # [pyc 3112-3122]
    t.start()
    def send_window():                                                                                       # py 734
        while True:                                                                                           # [pyc 3592 JUMP_ABSOLUTE→2]
            try:                                                                                              # [pyc 3594 SETUP_FINALLY→330]
                msg = edict(json.loads(window_queue.get()))                                                    # [pyc 3598-3612]
                print("send_window", msg)                                                                      # [pyc 3614-3622]
                if msg.type == "live_disconnected":                                                             # [pyc 3624-3632]
                    for key, p in runner_dict.items():                                                          # [pyc 3634-3646]
                        if p.ispc:                                                                              # [pyc 3650-3654]
                            p_device = p["device"]                                                              # [pyc 3656-3660 BINARY_SUBSCR]
                            try:
                                p.runner_process.terminate()
                                p.runner_process.join()
                                p.device = None                                                                    # [pyc 3698-3702 STORE_ATTR]
                            except:
                                pass
                            if p_device:                                                                        # [pyc 3704-3708]
                                deviceUid = get_windows_computer_id()                                            # [pyc 3710-3712]
                                client.publish(f"live/{deviceUid}", json.dumps({"type": "live_disconnected", "device": p_device}))  # [pyc 3714-3740]
                if msg.type == "start_voice":                                                                    # [pyc 3746-3754]
                    if msg.version == "gptsovits" or msg.version == "gpt-sovits":                                 # [pyc 3756-3774]
                        threading.Thread(target=gptsovits_run, daemon=True).start()
                    if msg.version == "vsa":
                        threading.Thread(target=vsa_run, daemon=True).start()
                    if msg.version == "qftts":
                        threading.Thread(target=qftts_run, daemon=True).start()
                    if msg.version == "vc":
                        threading.Thread(target=vc_run, daemon=True).start()
                    if msg.version == "indextts":
                        threading.Thread(target=indextts_run, daemon=True).start()
                    do_message(msg)                                                                                # [pyc 3906-3912]
            except Exception as e:                                                                                # [pyc 3918-3926]
                print(e)                                                                                           # [pyc 3932-3938]
    t = threading.Thread(target=send_window, daemon=True)                                                          # [pyc 3186-3196]
    t.start()
    setInterval(empty_cache, 600)                                                                                  # [pyc 3442-3448 常量 600]
    # [pyd-only] Timer(send_event_to_umami):pyd py=788(ATTR Timer+GLOBAL send_event_to_umami),
    # pyc start 无(Names 表无 Timer)。pyd 间隔常量为缓存整型之一,未定案:
    # TODO(ambiguous): Timer 间隔 ∈ {60,180,200,600}(pyd 缓存常量集),反编译未锁定。
    # [COM-D002 语义替换] 原实现无条件调度该定时器。现改为受 TELEMETRY_ENABLED
    # (默认 False)门控:**默认不调度**;而被调度函数本身已是惰性空壳(零出网),
    # 双保险确保遥测永不发出。调用点与符号集保持不变。
    if TELEMETRY_ENABLED:
        threading.Timer(180, send_event_to_umami).start()
    client.on_connect = lambda *args: on_connect(deviceId, *args)                                                    # [pyc 3452-3464 lambda 自由变量 deviceId][pyd lambda1 体 on_connect(t,*args)]
    client.on_message = lambda *args: on_message(context, mqtt_queue, sys_queue, *args)                                # [pyc 3466-3482]
    client.on_disconnect = on_disconnect                                                                              # [pyc 3484-3488]
    client.connect(mqtt_host, int(mqtt_port), 60)                                                                      # [pyc 3490-3504 第三参 keepalive=60 — 一轮漏;pyd raw L916 PyTuple(3) 同三参]
    client.loop_forever(retry_first_connection=False)                                                                  # [pyc 3508-3516]


if __name__ == "__main__":                                                                                             # py 800 [pyc 3772-3780]
    multiprocessing.freeze_support()                                                                                    # [pyc 3782-3788]
    window_queue = Queue()                                                                                              # [pyc 3790-3796]
    browser_queue = Queue()                                                                                             # [pyc 3798-3802]
    start(window_queue, browser_queue)                                                                                  # [pyc 3804-3810]
