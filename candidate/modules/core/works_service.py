# -*- coding: utf-8 -*-
"""works_service —— 本地短视频生成服务(文案 → 数字人口播视频)。

架构(2026-09-20 定稿):
  1. qftts 分段合成:cut_text 切段(与 oracle getQfttsLongClone 同款分段),
     逐段 qfttsClone(每段短,300s 内绰绰有余——曾因整段 101 字一次合成
     正好撞 5 分钟超时失败),pydub 拼接(_merge_pieces)成整段 16k wav;
  2. heygem 离线渲染:modules/heygem/runtime/python.exe run.py
     --audio_path <wav> --video_path <形象源 mp4>,产出 result/*-r.mp4;
     形象源 = 模特 live.mp4(文生视频)或用户上传素材(图生视频);
  3. 成品移入 human_data/works/。

真实进度:state.progress 0-100——合成期按 段 i/N(10~50%),渲染期按
dh.log 帧号/(音频秒×25fps)(50~95%),完成 100。前端据此驱动动画。

要点:
  * heygem 源码平级导入须 PYTHONPATH 提供子目录集(util/ 排除——其 html.py
    遮蔽标准库);
  * run.py 收尾 taskkill 工作子进程,无控制台时 rc 恒 1 但渲染完成——判定以
    产物为准(启动即清空 result/,此后出现的 *-r.mp4 必属本次);
  * 开播中拒绝生成;
  * 端点(app_gui):POST /api/works/generate|upload、GET /api/works/status|list、
    GET /works/<fname>。
"""
import glob
import os
import re
import shutil
import subprocess
import threading
import time

from pydub import AudioSegment

from modules.core import local_engine
from modules.core.app_util import _merge_pieces, cut_text, only_punc, qfttsClone
from modules.core.config import home_dir, models_dir, temp_dir

WORKS_DIR = os.path.join(home_dir, "works")
_RUN_ROOT = os.path.dirname(home_dir)                     # run_root(home_dir 的上级)
_HEYGEM = os.path.join(_RUN_ROOT, "modules", "heygem")
_HE_PY = os.path.join(_HEYGEM, "runtime", "python.exe")
_HE_LOG = os.path.join(_HEYGEM, "data", "log", "dh.log")
# 平级导入所需子目录(实测可跑集);util/ 必须排除(html.py 遮蔽标准库)
_HE_PATH_DIRS = ["face_detect_utils", "face_lib", "h_utils", "y_utils",
                 "service", "model_lib", "pack", "config", "landmark2face_wy",
                 "landmark2face_wy/options", "landmark2face_wy/models",
                 "landmark2face_wy/data"]

_lock = threading.Lock()
_state = {"running": False, "stage": "空闲", "phase": "", "progress": 0,
          "log": [], "error": None, "done": True, "file": None,
          "started_at": None}


def _say(stage, line):
    with _lock:
        _state["stage"] = stage
        _state["log"].append("%s %s" % (time.strftime("%H:%M:%S"), line))
        if len(_state["log"]) > 200:
            del _state["log"][:len(_state["log"]) - 200]


def _set_progress(phase, pct):
    with _lock:
        _state["phase"] = phase
        _state["progress"] = int(pct)


def _synthesize(voice_dir, text):
    """分段合成整段语音;返回 (wav 路径, 秒)。段进度写 state(10~50%)。"""
    segs = [s for s in cut_text(text).split("\n") if s and not only_punc(s)]
    if not segs:
        raise RuntimeError("文案无可合成内容")
    pieces = []
    try:
        for i, seg in enumerate(segs):
            _say("合成语音", "第 %d/%d 段：%s" % (i + 1, len(segs), seg[:24]))
            wav = qfttsClone(voice_dir, seg, 1)
            if not wav or not os.path.isfile(wav):
                raise RuntimeError("语音合成失败（第 %d 段，检查音色 refer 文件）"
                                   % (i + 1))
            pieces.append(wav)
            _set_progress("合成语音", 10 + 40.0 * (i + 1) / len(segs))
        merged = _merge_pieces(pieces, nostdin=True)
        if not merged or not os.path.isfile(merged):
            raise RuntimeError("语音拼接失败")
        seconds = len(AudioSegment.from_wav(merged)) / 1000.0
        for p in pieces:
            try:
                os.remove(p)
            except OSError:
                pass
        return merged, seconds
    except Exception:
        for p in pieces:
            try:
                os.remove(p)
            except OSError:
                pass
        raise


_FRAME_RE = re.compile(r"frame_id:\[(\d+)\]")


def _last_frame_id():
    """读 heygem dh.log 尾部的最新帧号(无则 0)。"""
    try:
        with open(_HE_LOG, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 20000))
            tail = f.read().decode("utf-8", "replace")
        ids = _FRAME_RE.findall(tail)
        return int(ids[-1]) if ids else 0
    except OSError:
        return 0


def _render_offline(wav_path, video_path, audio_seconds):
    """调 heygem run.py 离线渲染;渲染期按帧号报进度(50~95%)。
    判定以产物为准(run.py 无控制台时 rc 恒 1 但渲染完成,见模块 docstring)。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        os.path.join(_HEYGEM, d) for d in _HE_PATH_DIRS)
    err_path = os.path.join(home_dir, "temp", "heygem_render.err")
    t0 = time.time()
    total_frames = max(1, int(audio_seconds * 25) + 50)
    with open(err_path, "wb") as errf:
        proc = subprocess.Popen(
            [_HE_PY, "run.py", "--audio_path", os.path.abspath(wav_path),
             "--video_path", os.path.abspath(video_path)],
            cwd=_HEYGEM, env=env, stdout=errf, stderr=errf)
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            if time.time() - t0 > 1800:
                proc.kill()
                raise RuntimeError("离线渲染超时（30 分钟）")
            _set_progress("离线渲染",
                          50 + 45.0 * min(1.0, _last_frame_id() / total_frames))
            time.sleep(2)
    outs = sorted(glob.glob(os.path.join(_HEYGEM, "result", "*-r.mp4")),
                  key=os.path.getmtime)
    if outs and os.path.getmtime(outs[-1]) >= t0 - 5 \
            and os.path.getsize(outs[-1]) > 10000:
        return outs[-1]
    try:
        with open(err_path, "rb") as f:
            raw = f.read()[-1500:].decode("gbk", "replace")
        tail = " | ".join(raw.strip().splitlines()[-3:])
    except OSError:
        tail = ""
    raise RuntimeError("离线渲染失败(rc=%s): %s" % (rc, tail))


def _run(text, model, voice):
    wav = None
    try:
        _set_progress("合成语音", 10)
        _say("合成语音", "qftts 分段合成整段语音…")
        if not local_engine.ensure_tts():
            raise RuntimeError("TTS 服务启动失败")
        voice_dir = os.path.join(home_dir, "voices", voice)
        wav, seconds = _synthesize(voice_dir, text)
        _say("合成语音", "语音就绪：%.1f 秒" % seconds)

        _set_progress("离线渲染", 50)
        _say("离线渲染", "heygem 逐帧渲染中（约 1-3 分钟）…")
        src_mp4 = _render_offline(
            wav, os.path.join(models_dir, model, "live.mp4"), seconds)

        os.makedirs(WORKS_DIR, exist_ok=True)
        fname = "work_%s.mp4" % time.strftime("%y%m%d%H%M%S")
        shutil.move(src_mp4, os.path.join(WORKS_DIR, fname))
        _set_progress("完成", 100)
        _say("完成", "短视频已生成: %s" % fname)
        with _lock:
            _state.update(running=False, done=True, error=None, file=fname)
    except Exception as exc:
        _say("失败", "生成失败: %s" % exc)
        with _lock:
            _state.update(running=False, done=True, error=str(exc))
    finally:
        if wav:
            try:
                os.remove(wav)
            except OSError:
                pass


def generate(text, model, voice):
    """启动一次生成;进行中/开播中拒绝。"""
    with _lock:
        if _state["running"]:
            return {"ok": False, "error": "已有短视频在生成中，请稍候"}
    text = (text or "").strip()
    if not text:
        return {"ok": False, "error": "请先输入文案"}
    if not model or not voice:
        return {"ok": False, "error": "请选择模特与音色"}
    if not os.path.isfile(os.path.join(models_dir, model, "live.mp4")):
        return {"ok": False, "error": "模特资源不存在或不完整"}
    voice_dir = os.path.join(home_dir, "voices", voice)
    if not os.path.isdir(voice_dir) or not any(
            f.lower().endswith(".wav") for f in os.listdir(voice_dir)):
        return {"ok": False, "error": "音色资源不存在或不完整"}
    if not os.path.isfile(_HE_PY):
        return {"ok": False, "error": "离线渲染模块缺失(modules/heygem)"}
    st = local_engine.status()
    if st.get("running"):
        return {"ok": False, "error": "当前开播中，请先下播再生成短视频"}
    with _lock:
        _state.update(running=True, stage="启动", phase="启动", progress=0,
                      log=[], error=None, done=False, file=None,
                      started_at=int(time.time()))
    _say("启动", "开始生成短视频（%d 字文案）" % len(text))
    threading.Thread(target=_run, args=(text, model, voice),
                     daemon=True).start()
    return {"ok": True}


def status():
    with _lock:
        snap = dict(_state)
        snap["log"] = list(_state["log"])
        return snap


def list_works():
    os.makedirs(WORKS_DIR, exist_ok=True)
    items = []
    for name in os.listdir(WORKS_DIR):
        if name.lower().endswith(".mp4"):
            path = os.path.join(WORKS_DIR, name)
            items.append({"name": name, "size": os.path.getsize(path),
                          "mtime": int(os.path.getmtime(path))})
    items.sort(key=lambda x: -x["mtime"])
    return items
