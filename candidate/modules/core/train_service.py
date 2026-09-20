# -*- coding: utf-8 -*-
"""train_service —— 本地一键训练服务(模特/声音资产制备 + 登记)。

红队 R-1B-007/R-1B-016 曾把训练页登记为"待接入"占位;本模块把训练页接线到
本地资产制备链。本产品的"训练"语义 = 素材校验 + 规整 + 登记(与库内既有
30 模特 / 27 音色的磁盘形态一致):

  * 模特:视频(mp4)经 cv2 校验(宽高 ≥ 256、fps>0、帧数>0,阈值取
    train_util._MIN_TRAIN_WIDTH/_MIN_TRAIN_HEIGHT)后,写入
    human_data/models/user_<yyMMddHHmmss>/{live.mp4, model.ini(version=local)}。
    推理端消费面:local_engine._render_child 仅读 <models>/<uid>/live.mp4
    (local_engine.py `face` 拼接);/api/engine/models 按 model.ini 过滤。
  * 声音:音频(mp3/wav)ffmpeg 归一化 16k 单声道(与 app_util.qfttsClone 的
    16k 规整一致)成 refer.wav + 参考文本 refer.txt,写入
    human_data/voices/user_<yyMMddHHmmss>/{refer.wav, refer.txt,
    model.ini(version=qftts)}。推理端消费面:app_util.randomPrompt 取目录内
    .wav + 同名 .txt;qfttsClone 走本地 qftts 服务克隆。

登记原子性:两个分支都**最后写 model.ini**(资源列表按 model.ini 过滤),
半成品目录不会出现在模特/声音列表;训练失败即清理半成品目录。

HTTP 端点(挂在 app_gui.start_bottle,契约 C1 风格 ok/error 中文):
  POST /api/train/upload  multipart 字段 file(+kind=model|voice)→ 暂存缓存目录
  POST /api/train/start   {type: model|voice, path, refer_text?} → 后台线程训练
  GET  /api/train/status  → {running, type, uid, stage, log[], error, done}
"""
import os
import shutil
import subprocess
import sys
import threading
import time

import cv2

from modules.core.config import home_dir

# 上传上限(视频 500MB / 音频 50MB;超出直接拒收,不给半截文件)
MAX_UPLOAD = {"model": 500 * 1024 * 1024, "voice": 50 * 1024 * 1024}
_EXT = {"model": (".mp4",), "voice": (".mp3", ".wav")}
# 缓存与目标目录(config.home_dir 为唯一权威源,不内嵌项目根字面量)
_CACHE_DIR = os.path.join(home_dir, "cache", "train")
_MODEL_ROOT = os.path.join(home_dir, "models")
_VOICE_ROOT = os.path.join(home_dir, "voices")
_FFMPEG = os.path.join("bin", "ffmpeg.exe")

_lock = threading.Lock()
_state = {"running": False, "type": None, "uid": None, "stage": "空闲",
          "log": [], "error": None, "done": True, "started_at": None}


def _say(stage, line):
    """登记一行进度(带时间戳,上限 200 行,防长训练撑爆 status 载荷)。"""
    with _lock:
        _state["stage"] = stage
        _state["log"].append("%s %s" % (time.strftime("%H:%M:%S"), line))
        if len(_state["log"]) > 200:
            del _state["log"][:len(_state["log"]) - 200]


def _new_uid(root):
    """user_<yyMMddHHmmss>;同秒冲突时加序号后缀,保证目录不撞。"""
    uid = "user_" + time.strftime("%y%m%d%H%M%S")
    if not os.path.exists(os.path.join(root, uid)):
        return uid
    i = 1
    while os.path.exists(os.path.join(root, "%s_%d" % (uid, i))):
        i += 1
    return "%s_%d" % (uid, i)


_WIN_RESERVED = {"CON", "PRN", "AUX", "NUL"} | \
    {"COM%d" % i for i in range(1, 10)} | {"LPT%d" % i for i in range(1, 10)}


def _clean_name(name):
    """用户自定义资源名(即目录名):镜像 app_gui._safe_name 口径——拒路径符/
    冒号/首点/尾点尾空格/Windows 保留名/超长;中文合法。空返回 ""(自动编号),
    非法返回 None。"""
    name = (name or "")
    if name != name.rstrip(". "):
        return None
    name = name.strip()
    if not name:
        return ""
    if "/" in name or "\\" in name or ":" in name or ".." in name \
            or name.startswith(".") or len(name) > 255:
        return None
    if name.split(".", 1)[0].upper() in _WIN_RESERVED:
        return None
    return name


def save_upload(kind, filename, stream):
    """暂存上传素材到 human_data/cache/train/;返回 {"ok",...} 契约体。"""
    if kind not in _EXT:
        return {"ok": False, "error": "素材类型非法(仅支持 model/voice)"}
    name = os.path.basename(filename or "")
    if not name.lower().endswith(_EXT[kind]):
        return {"ok": False, "error": "文件格式不支持:%s(要求 %s)"
                % (name or "(空)", "/".join(_EXT[kind]))}
    os.makedirs(_CACHE_DIR, exist_ok=True)
    stored = os.path.join(_CACHE_DIR, "src_%d_%s" % (int(time.time()), name))
    try:
        with open(stored, "wb") as out:
            shutil.copyfileobj(stream, out, 1024 * 1024)
    except OSError as exc:
        return {"ok": False, "error": "保存上传文件失败: %s" % exc}
    if os.path.getsize(stored) > MAX_UPLOAD[kind]:
        os.remove(stored)
        return {"ok": False, "error": "文件过大(上限 %dMB)"
                % (MAX_UPLOAD[kind] // (1024 * 1024))}
    return {"ok": True, "path": stored, "name": name,
            "size": os.path.getsize(stored)}


def _clean_dir(path):
    """清理半成品目录(兼容只读属性,Windows 删树常规兜底)。"""
    def _onerror(func, p, _exc):
        os.chmod(p, 0o666)
        func(p)
    shutil.rmtree(path, onerror=_onerror)


def _train_model(src, uid_dir):
    """模特训练主体:校验 → 拷贝 live.mp4 → 写 ini(最后写,原子登记)。"""
    _say("校验视频", "读取视频参数: " + src)
    cap = cv2.VideoCapture(src)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    _say("校验视频", "宽度=%d 高度=%d FPS=%s 帧数=%d" % (width, height, fps, frames))
    if width < 256 or height < 256 or fps <= 0 or frames <= 0:
        raise ValueError("视频不符合要求。当前参数：宽度=%d，高度=%d，FPS=%s"
                         % (width, height, fps))
    _say("规整素材", "拷贝训练视频 → live.mp4")
    shutil.copyfile(src, os.path.join(uid_dir, "live.mp4"))
    _say("登记资源", "写入 model.ini(version=local)")
    with open(os.path.join(uid_dir, "model.ini"), "wb") as handle:
        handle.write(b"[Model]\r\nversion = local\r\n")


def _auto_transcribe(wav_path):
    """本地 FunASR 自动转写(qftts 模块内建识别器 funasr_asr.only_asr,模型
    缓存于 ~/.cache/modelscope)。子进程隔离:funasr 导入即载 torch,不进主进程。
    返回转写文本;失败返回 ""(调用方回退为要求手填)。"""
    script = ("import sys; sys.path.insert(0, 'modules/qftts'); "
              "from funasr_asr import only_asr; "
              "print(only_asr(sys.argv[1], 'zh'))")
    try:
        out = subprocess.run([sys.executable, "-c", script, wav_path],
                             capture_output=True, text=True, timeout=300)
    except (subprocess.TimeoutExpired, OSError):
        return ""
    for line in reversed((out.stdout or "").strip().splitlines()):
        line = line.strip()
        if line:
            return line
    return ""


def _train_voice(src, uid_dir, refer_text):
    """声音训练主体:ffmpeg 归一化 refer.wav → 参考文本(空则本地 ASR 自动
    转写)→ refer.txt → ini(最后写)。"""
    _say("规整素材", "ffmpeg 归一化音频 → refer.wav(16k 单声道)")
    raw = os.path.join(uid_dir, "raw_refer.wav")
    shutil.copyfile(src, raw)
    refer = os.path.join(uid_dir, "refer.wav")
    rc = subprocess.call([_FFMPEG, "-nostdin", "-loglevel", "quiet", "-y",
                          "-i", raw, "-ac", "1", "-ar", "16000", refer])
    os.remove(raw)
    if rc != 0 or not os.path.isfile(refer) or os.path.getsize(refer) < 1000:
        raise ValueError("音频归一化失败(源文件可能损坏)")
    if not refer_text:
        _say("自动识别", "未填参考文本，本地 FunASR 自动转写录音…")
        refer_text = _auto_transcribe(os.path.abspath(refer))
        if not refer_text.strip():
            raise ValueError("录音自动识别失败（无人声或识别异常），请手动填写参考文本后重试")
        _say("自动识别", "识别结果: " + refer_text[:60])
    _say("登记资源", "写入 refer.txt(参考文本)")
    with open(os.path.join(uid_dir, "refer.txt"), "w", encoding="utf-8") as handle:
        handle.write(refer_text)
    _say("登记资源", "写入 model.ini(version=qftts)")
    with open(os.path.join(uid_dir, "model.ini"), "wb") as handle:
        handle.write(b"[Model]\r\nversion = qftts\r\n")


def _run(train_type, src, refer_text, uid, uid_dir):
    """训练线程主体:统一异常收口 + 完成态登记。"""
    try:
        if train_type == "model":
            _train_model(src, uid_dir)
        else:
            _train_voice(src, uid_dir, refer_text)
        _say("完成", "训练完成，资源已登记: %s" % uid)
        with _lock:
            _state.update(running=False, done=True, error=None)
    except Exception as exc:                      # 线程内兜底,不让工作线程静默死
        _clean_dir(uid_dir)
        _say("失败", "训练失败: %s" % exc)
        with _lock:
            _state.update(running=False, done=True, error=str(exc))
    finally:
        try:
            os.remove(src)                       # 暂存素材用后即清
        except OSError:
            pass


def start(train_type, path, refer_text="", name=""):
    """启动一次训练;进行中重复调用拒绝(对齐 T-027 防重入口径)。
    name:可选资源名(即目录名,中文合法);空则 user_<时间戳> 自动编号。"""
    with _lock:
        if _state["running"]:
            return {"ok": False, "error": "训练进行中，请勿重复操作"}
    if train_type not in ("model", "voice"):
        return {"ok": False, "error": "训练类型非法(model/voice)"}
    if not path or not os.path.isfile(path):
        return {"ok": False, "error": "素材文件不存在，请先上传"}
    if not str(path).lower().endswith(_EXT[train_type]):
        return {"ok": False, "error": "素材格式与训练类型不符"}
    # 声音参考文本可选:留空则训练时本地 FunASR 自动转写(见 _train_voice)
    root = _MODEL_ROOT if train_type == "model" else _VOICE_ROOT
    clean = _clean_name(name)
    if clean is None:
        return {"ok": False, "error": "资源名称非法（不能含 /\\: 等符号或保留名）"}
    if clean:
        if os.path.exists(os.path.join(root, clean)):
            return {"ok": False, "error": "已有同名资源：%s，请换个名字" % clean}
        uid = clean
    else:
        uid = _new_uid(root)
    uid_dir = os.path.join(root, uid)
    os.makedirs(uid_dir, exist_ok=True)
    with _lock:
        _state.update(running=True, type=train_type, uid=uid, stage="启动",
                      log=[], error=None, done=False,
                      started_at=int(time.time()))
    _say("启动", "开始训练 %s → %s" % (os.path.basename(path), uid))
    threading.Thread(target=_run,
                     args=(train_type, path, (refer_text or "").strip(), uid, uid_dir),
                     daemon=True).start()
    return {"ok": True, "uid": uid}


def status():
    """状态快照(锁内浅拷贝,log 列表复制防竞态)。"""
    with _lock:
        snap = dict(_state)
        snap["log"] = list(_state["log"])
        return snap
