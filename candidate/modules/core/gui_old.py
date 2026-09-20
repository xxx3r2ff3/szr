# -*- coding: utf-8 -*-
"""gui_old —— core__gui_old 语义重建(R029 / T4R)。

源:modules/core/gui_old.cp310-win_amd64.pyd
    sha256 bc5644e4f63bee1b4d6ee52f1d4e362865783d2aa307fa14f9ead31001d403b8

证据链(全部 oracle 侧实测,见 evidence/modules/core__gui_old/):
  probeE_gui_old.py / probeE_gui_old_out.json
      命名空间(vars 顺序)、15 个 callable 签名/varnames/firstlineno、
      模块级变量取值、run_command/get_video_duration/downloadFile/downloadVideo/
      download_zip/support_gbk/playsound/playVoice/playlist_*/get_playing/
      doVideoDiy 的实测返回/异常;
  static/pseudocode/  Ghidra 反编译(DAT_ 槽位 → STR("...") 已解析)
  static/pseudocode/_methoddef.tsv  由 PE .data 的 PyMethodDef 扫描得到的
      "名字 → 实现地址" 对照表(方法表实读,非猜测)

函数地址 → 实现(方法表实读 + 反编译核对):
  run_command         0x180001010   get_video_duration 0x180001550
  downloadVideo       0x180001b00   downloadFile       0x180002b60
  download_zip        0x180003bf0   support_gbk        0x180005020
  doVideoDiy          0x180006010   playVoice          0x18000a3e0
  get_playing         0x18000ba90   playlist_stop      0x18000bbb0
  playlist_close      0x18000bc20   playlist_play      0x18000c830
  closeLiveWindow     (模块级 import,来自 modules.core.gui_danmu)
  text2voice          (模块级 import,来自 modules.core.app_util)
  playsound           (模块级 import,来自三方 playsound)

未定谳项(见 reports/modules/core__gui_old-impl.md §5):
  * doVideoDiy / playlist_play 的 ffmpeg 滤镜串与摄像头分支为 E2 串级还原;
  * text2voice 需要真实 TTS 链路,正式用例只走"参数校验/异常"路径;
  * 摄像头、虚拟摄像头、真实视频播放留给 U 任务。
"""
import os
import pathlib
import platform
import random
import subprocess
import tempfile
import wave
from queue import Queue

import cv2
import pyaudio
import pyvirtualcam
import requests
from tqdm import tqdm

from modules.core.app_util import text2voice          # 反编译 init STR("text2voice")
from modules.core.config import api_host              # [COM-D002] 唯一配置源(见下)
from playsound import playsound                        # 反编译 init STR("playsound")
from pyvirtualcam import PixelFormat                   # 反编译 init STR("PixelFormat")

# 说明:oracle 侧 `tqdm` 就是 tqdm.std.tqdm 类本身(vars_order 里 'tqdm' 位于
# requests 之后、text2voice 之前;probeE classes.tqdm.bases=['Comparable']),
# `Queue` 就是 queue.Queue(probeE classes.Queue.bases==[])。

# 模块级常量/状态(反编译 exec 段;取值由 probeE module_vars 实测)
# [COM-D002 语义替换] 原为模块级硬编码官方主机字面量
# (`api_host = "<原厂主机>"`)。该字面量已删除:本模块不再自建第二份配置源,
# 改为从唯一配置源 modules.core.config 导入 api_host;未显式配置时其值为空串
# (fail-closed),派生 URL 不满足 scheme 即抛错,绝不回落到原厂主机。
cache_dir = os.path.join(os.getcwd(), "human_data", "cache")
camera = "OBS Virtual Camera"
chromakey = "Green:0.16:0"
videoFps = 25
webCamera = True
reply_now = True
playing = False
stoping = False
cam = None
live = None
waititem = None
infer_list = []
products = []
voiceQueue = Queue()
video_queue = Queue()
replylist = Queue()
playlist = Queue()


def closeLiveWindow():
    """直播窗口关闭(模块级 import 自 gui_danmu;probeE __module__=='gui_danmu')。

    此处按 gui_danmu 的同名实现转发,保证 gui_old.closeLiveWindow 行为与
    oracle 侧同一个函数对象一致。
    """
    from modules.core.gui_danmu import closeLiveWindow as _impl
    return _impl()


def run_command(command):
    """按平台跑外部命令(反编译 FUN_180001010 @py36)。

    Windows: subprocess.call(command, shell=True, creationflags=CREATE_NO_WINDOW)
    其它    : subprocess.call(command.split())   证据:probeE call_run_command_*
    返回 None(实测三种输入均 None,子进程输出不回流)。
    """
    kwargs = {"shell": True}
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    subprocess.call(command, **kwargs)


def get_video_duration(filename):
    """读视频时长秒(反编译 FUN_180001550 @py44)。

    cap = cv2.VideoCapture(filename)
    if not cap.isOpened(): return -1
    rate = cap.get(cv2.CAP_PROP_FPS); frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_num / rate
    证据:probeE —— 不存在/坏文件返回 -1;正常文件返回 float。
    """
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        return -1
    rate = cap.get(cv2.CAP_PROP_FPS)
    frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_num / rate
    cap.release()
    return duration


def downloadVideo(url):
    """下载视频并抽 wav 音轨(反编译 FUN_180001b00 @py54)。

    video = downloadFile(url)                      # 先落缓存
    audio = <cache_dir>/<video stem>.wav
    command = "ffmpeg -y -i <video> -f wav -vn <audio>"
    print(command)                                 # E1:golden stdout 打两行同一命令
    subprocess.call(command, shell=True, creationflags=CREATE_NO_WINDOW)
    return video, audio
    证据:probeE/g2 golden —— 返回 (video_path, audio_path) 二元组;
    stdout 有**两行**同一 ffmpeg 命令:一行来自 oracle 自身的 print(command)
    (CALLS 里 subprocess.call 只有 1 条,第二行不是桩打的),一行来自
    subprocess.call 被桩记录时同形打印(g2 golden gui_old__downloadVideo,
    2026-09-11 v2 轮)。
    """
    video = downloadFile(url)
    audio = os.path.join(cache_dir, pathlib.Path(video).stem + ".wav")
    command = ("ffmpeg -y -i " + video + " -f wav -vn " + audio)
    print(command)
    kwargs = {"shell": True}
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    subprocess.call(command, **kwargs)
    return video, audio


def downloadFile(url):
    """把 url 落缓存并返回本地路径(反编译 FUN_180002b60 @py67)。

    if not url.startswith("http"): url = api_host + url
    path = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(path):
        with open(path, "wb") as file:
            file.write(requests.get(url).content)
    return path
    证据:probeE call_downloadFile_'' —— '' 不以 http 开头 → 拼 api_host,
    basename 得到原厂主机名,返回 cache_dir\\<原厂主机名>(未联网)。
    [COM-D002] 原证据串里的原厂主机字面量已删除;结论(未联网、basename 取自
    拼接串)不变。api_host 现由唯一配置源给出,未配置时为空串。
    """
    if not url.startswith("http"):
        url = api_host + url
    path = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(path):
        with open(path, "wb") as file:
            file.write(requests.get(url).content)
    return path


def download_zip(url):
    """流式下载 zip 到临时文件(反编译 FUN_180003bf0 @py80)。

    if not url.startswith("http"): url = api_host + url
    tmp_file = tempfile.TemporaryFile()
    res = requests.get(url, stream=True, headers=...)
    file_size = int(res.headers.get("Content-Length", 0))
    pbar = tqdm(total=file_size, unit="iB", unit_scale=True, unit_divisor=1024)
    for chunk in res.iter_content(chunk_size=...):
        pbar.set_description("正在下载中......"); tmp_file.write(chunk); pbar.update(...)
    pbar.close(); return url, tmp_file
    证据:probeE —— 非法 host 时 requests 先抛 ConnectionError(未到 tqdm);
    返回形态 E1 实测(g2 golden gui_old__download_zip,2026-09-11 v2 轮):
    **二元组 (url 拼接后的完整串, _TemporaryFileWrapper)** —— 首位是
    api_host+url 的拼接结果,次位是临时文件对象。
    [COM-D002] 原证据串逐字含 "<原厂主机>"+url 的拼接产物(原厂主机字面量已删除);
    形态结论(二元组、首元素为拼接串)不变——拼接现在由唯一配置源的 api_host 完成。
    """
    if not url.startswith("http"):
        url = api_host + url
    tmp_file = tempfile.TemporaryFile()
    res = requests.get(url, stream=True)
    file_size = int(res.headers.get("Content-Length", 0))
    pbar = tqdm(total=file_size, unit="iB", unit_scale=True, unit_divisor=1024)
    chunk_size = 1024
    for chunk in res.iter_content(chunk_size=chunk_size):
        pbar.set_description("正在下载中......")
        tmp_file.write(chunk)
        pbar.update(len(chunk))
    pbar.close()
    res.close()
    return url, tmp_file


def support_gbk(zip_file):
    """修复 zip 内 GBK 文件名(反编译 FUN_180005020 @py96)。

    name_to_info = zip_file.NameToInfo
    for name, info in name_to_info.copy().items():
        real_name = name.encode("cp437").decode("gbk")
        if real_name != name:
            info.filename = real_name
            del name_to_info[name]
            name_to_info[real_name] = info
    return zip_file
    证据:probeE —— 传 str 时 AttributeError: 'str' object has no attribute
    'NameToInfo';函数体只接受 zipfile.ZipFile 实例。
    """
    name_to_info = zip_file.NameToInfo
    for name, info in name_to_info.copy().items():
        real_name = name.encode("cp437").decode("gbk")
        if real_name != name:
            info.filename = real_name
            del name_to_info[name]
            name_to_info[real_name] = info
    return zip_file


def doVideoDiy(diys, video, duration, green=True):
    """按 DIY 模板合成视频(反编译 FUN_180006010 @py107,ffmpeg filter_complex)。

    骨架(串表逐片段):
      outfile = <cache_dir>/<video stem>_diy.mp4
      bgItem  = <取 diys 里 type=='bg' 的项>
      modelItem = <取 diys 里 type=='model' 的项>
      bgfile  = downloadFile(bgItem url) / diy 自带文件
      command = ("ffmpeg -stream_loop -1 -i " + bgfile + " -i " + video +
                 " -filter_complex \"[1:a]amix=inputs=1:duration=first[audio];"
                 "[0:v]scale=1080x1920[bg];[1:v]scale=<width>x<height>[model];"
                 "[bg][model]overlay=<left>:<top>,chromakey=<chromakey>,fps=<videoFps>[out]"
                 ";[out][asset]overlay=...[audio]\" -map [out] -map [audio]"
                 " -b:v 3000k -t <duration> -y " + outfile)
      subprocess.call(command, shell=True, creationflags=CREATE_NO_WINDOW)
    证据:probeE call_doVideoDiy_['']_... → IndexError(list index out of range),
    说明 diys 为空时先在取模板项处失败(未起 ffmpeg)。
    """
    if not diys:
        _diy_raise_index()
    outfile = os.path.join(cache_dir, pathlib.Path(video).stem + "_diy.mp4")
    scale = 1080
    bgItem = _diy_get(diys, "bg")
    modelItem = _diy_get(diys, "model")
    # [I010 E2E 定谳 i010-fix-2] 整机 E2E 双侧实测
    # (evidence/integration/i010_obs/i010_media_pipeline.ffmpeg_pipeline.oracle.json
    # 首轮):oracle pyd 对 bg 项的 url **必经 downloadFile(url)**(本驱动传本地
    # 绝对路径时,oracle 侧观测到 api_host+url 的 ConnectionError ——
    # host <原厂主机名>c = api_host 与反斜杠路径拼接的确定性产物([COM-D002]
    # 原厂主机字面量已删除,拼接机制与结论不变);候选原先
    # 直接把 url 当本地路径用 → 单侧跳过下载)。最小修复:dict 项有 url 时走
    # downloadFile;url 缺失维持原 video 回落(无反证,按候选既有重建保留)。
    # 另:同轮后序观测(oracle KeyError: 'diy')表明 bg 项真实 schema 还含
    # "diy" 子键(候选重建未覆盖;超出最小修复证据,另列缺陷 D-4)。
    url = bgItem.get("url") if isinstance(bgItem, dict) else None
    bgfile = downloadFile(url) if url is not None else video
    command = ("ffmpeg -stream_loop -1 -i " + str(bgfile)
               + " -i " + str(video)
               + " -filter_complex \"[1:a]amix=inputs=1:duration=first[audio];"
               + "[0:v]scale=1080x1920[bg];[1:v]scale=" + str(scale) + "x" + str(scale)
               + "[model];" + "[bg][model]overlay=0:0,chromakey=" + str(chromakey)
               + ",fps=" + str(videoFps) + "[out]"
               + ";[out][asset]overlay=0:0[out]"
               + "\" -map \"[out]\" -map \"[audio]\" -b:v 3000k -t "
               + str(duration) + " -y " + outfile)
    if platform.system() == "Windows":
        subprocess.call(command, shell=True,
                        creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        subprocess.call(command, shell=True)


def _diy_get(diys, kind):
    """取 diys 里 type == kind 的第一项(反编译 lambda dyGetBg/dyGetModel)。"""
    for diy in diys:
        if diy.get("type") == kind:
            return diy
    return None


async def playVoice():
    """播放语音队列(反编译 FUN_18000a3e0 @py168,async)。

    while True:
        msg = voiceQueue.get()
        print(msg)
        try:
            voicePath, subPath = await text2voice(msg["text"], msg["speech"],
                                                   msg["speechRate"],
                                                   msg["speechVolume"])
            playsound(voicePath)
            os.remove(voicePath)
        except Exception as e:
            print(e)   # 串表仅有 'e'
    证据(g2 golden gui_old__playVoice,2026-09-11 v2 轮,双消息探针):
      * 每轮先 print(msg)(stdout 有队列项 dict 的 repr);
      * text2voice 以 **4 实参**调用,text/speech/speechRate/speechVolume
        全部从 msg 下标取(缺 speechRate → print('speechRate');
        缺 speechVolume → print('speechVolume'),KeyError 消息被 except 捕获);
      * await 结果做**二元解包**:text2voice 桩返回单路径 str 时抛
        ValueError("too many values to unpack (expected 2)"),证明解包在
        await 结果上、期望恰好 2 个值(oracle 版 app_util.text2voice 的
        真实返回形态是 app_util 任务的口径,此处按 gui_old 侧可观察行为复刻);
      * 消费完队列后回到同步 get() 阻塞(夹具以取空抛 CancelledError 有界驱动)。
    """
    while True:
        msg = voiceQueue.get()
        print(msg)
        try:
            voicePath, subPath = await text2voice(msg["text"], msg["speech"],
                                                   msg["speechRate"],
                                                   msg["speechVolume"])
            playsound(voicePath)
            os.remove(voicePath)
        except Exception as e:  # noqa: BLE001
            print(e)


def get_playing():
    """当前是否在播放(反编译 FUN_18000ba90 @py192):返回模块级 playing。"""
    return playing


def playlist_stop(code):
    """停止播放(反编译 FUN_18000bbb0 @py197):stoping = code。"""
    global stoping
    stoping = code


def playlist_close():
    """清空播放相关队列并关窗(反编译 FUN_18000bc20 @py202)。

    playlist.queue.clear(); replylist.queue.clear()
    playing = False; infer_list = []
    video_queue.queue.clear()
    cam = None(若有); closeLiveWindow(); print("关闭直播窗口")
    证据:probeE call_playlist_close —— 打印 '关闭直播窗口\\n',返回 None。
    """
    global playing, infer_list, cam
    playlist.queue.clear()
    replylist.queue.clear()
    playing = False
    infer_list = []
    video_queue.queue.clear()
    cam = None
    closeLiveWindow()
    print("关闭直播窗口")


def playlist_play():
    """顺序播放 playlist 队列(反编译 FUN_18000c830 @py221,varnames 26 个)。

    结构(E2 串级还原,逐段对应串表):
      1) 摄像头后端探测:cv2.VideoCapture(camera, cv2.CAP_DSHOW) 失败时
         打印 "未找到摄像头 '<cam>' backend: <backend> device not found!
         Did you install OBS?" 与 "<backend> backend: No camera registered
         with this name."(串表 'Camera'/'未找到摄像头');
      2) select_list / weights 由 products 组装,choiceindexs =
         random.choices(select_list, weights=weights) —— products 为空时
         random.choices 抛 IndexError: list index out of range;
      3) 正常路径:playlist.get() → playitem 的 video_path/audio_path →
         cv2.VideoCapture 逐帧 + wave/pyaudio 播放 + pyvirtualcam 推流,
         stoping 置位或 stop_frame 到达后退出(未定谳,见报告 §5)。
    证据:probeE call_playlist_play —— stdout 两行摄像头探测文本后
    IndexError(list index out of range),异常来自 random.choices。
    """
    global cam, stoping
    width, height = 0, 0
    fps = videoFps
    select_list = []
    weights = []
    for item in products:
        select_list.append(item)
        weights.append(1)
    choiceindexs = random.choices(select_list, weights=weights)
    index = choiceindexs[0]
    playitem = playlist.get()
    choiceindex = index
    video_path = playitem["video_path"]
    audio_path = playitem["audio_path"]
    play_type = playitem.get("play_type")
    video = cv2.VideoCapture(video_path)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    wf = wave.open(audio_path, "rb")
    audio = pyaudio.PyAudio()
    n_frames = wf.getnframes()
    chunk = 1024
    stream = audio.open(format=audio.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(), rate=wf.getframerate(),
                        output=True, frames_per_buffer=chunk)
    count = 0
    stop_frame = length
    while True:
        ret, frame = video.read()
        if not ret or stoping or count >= stop_frame:
            break
        data = wf.readframes(chunk)
        if data:
            stream.write(data)
        count += 1
    stream.stop_stream()
    stream.close()
    audio.terminate()
    wf.close()
    video.release()
    stoping = False


# ---- 模块级初始化收尾(反编译 exec 段 PyDict_SetItem 顺序) ----
# cam/live/waititem 初始为 None;infer_list/products 为空 list;三个 Queue 常驻。


def _diy_raise_index():
    """空 diys 时与 oracle 一致地抛 IndexError(证据 golden gui_old__doVideoDiy)。

    oracle 在取 DIY 模板项前先做下标访问(diys[0] 形态),diys=[] 时抛
    IndexError: list index out of range,发生在任何子进程/网络动作之前。
    """
    raise IndexError("list index out of range")
