#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""local_engine —— 本地实时引擎管理器(自建,2026-09-19)。

不依赖 MQTT/云端/令牌,直接编排本地组件:
  * SRS(本地流媒体,RTMP 3935 / HTTP-FLV 3089 / WebRTC 3088);
  * qftts TTS(本地 9885,zipvoice_distill);
  * 实时渲染子进程(原版 chat_infer 编译真身 Runner,本地 ONNX 权重)。

前端(local_ui)通过 start_bottle 注册的 /api/engine/* HTTP 接口驱动本模块。
"""
from __future__ import annotations

import os

# 本机 numba/llvmlite 在 AVX 向量化时误用 SVML(未随包提供)导致 LLVM abort,
# 强制 generic 代码路径绕过(必须在任何 numba/公式库导入前进程环境内生效)。
os.environ.setdefault("NUMBA_CPU_NAME", "generic")

import json
import socket
import subprocess
import sys
import threading
import time

import psutil

from modules.core import config as _cfg

SRS_RTMP = int(_cfg.rtmp_port)
SRS_API = int(_cfg.srs_port)
SRS_HTTP = int(_cfg.srs_http_port)
SRS_RTC = int(_cfg.srs_rtc_port)
import torch.multiprocessing as mp

RUN_ROOT = os.path.abspath(os.getcwd())
RUNTIME_PY = os.path.join(RUN_ROOT, "runtime", "python.exe")

_state = {
    "proc": None,          # 渲染子进程
    "tts_q": None,         # 跨进程 TTS 任务队列(Manager.Queue 代理)
    "model": None,
    "voice": None,
}

# T-027 (R-1D-005): _state 的全部读写统一由这把粗粒度锁保护;原 _state 内的
# speak_lock 移除 —— 全模块仅此一把锁,任何临界区内不得再取锁(禁嵌套防死锁)。
_state_lock = threading.Lock()

# T-027 (R-1D-005): start 的轻量 in-flight 防重入标记(仅在 _state_lock 内读写)。
# 评估:全程持锁会使 status 轮询在 ensure_tts(最长 120s)期间阻塞,不可取;
# 取最小实现 —— 登记/清账持锁,长耗时拉起段无锁,期间并发 start 直接拒绝。
_starting = False


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def wait_port(port: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_in_use(port):
            return True
        time.sleep(1)
    return False


def ensure_srs() -> bool:
    """确保本地 SRS 在线(RTMP 3935)。"""
    if port_in_use(SRS_RTMP):
        return True
    srs_dir = os.path.join(RUN_ROOT, "srs")
    exe = os.path.join(srs_dir, "objs", "srs.exe")
    subprocess.Popen([exe, "-c", "console.conf"], cwd=srs_dir,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wait_port(SRS_RTMP, 20)


def ensure_tts() -> bool:
    """确保本地 qftts TTS 在线(9885)。"""
    if port_in_use(_cfg.tts_port):                       # T-058 (R-1D-014): 字面量改配置引用
        return True
    qdir = os.path.join(RUN_ROOT, "modules", "qftts")
    subprocess.Popen([RUNTIME_PY, "start_api.py",
                      "--model-name", "zipvoice_distill",
                      "--port", str(_cfg.tts_port)],     # T-058 (R-1D-014): 默认 9885 不变
                     cwd=qdir, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
    return wait_port(_cfg.tts_port, 120)                 # T-058 (R-1D-014)


def _render_child(runner_q, mqtt_q, tts_q, model_name: str, voice_dir: str) -> None:
    """渲染子进程主体:复刻已验证的本地驱动流程,循环消费语音队列。"""
    from queue import Queue

    import psutil  # noqa: F401  (保持与主进程依赖一致)
    from modules.core.app_util import qfttsClone
    from modules.core.chat_infer import Runner

    face = os.path.join(os.getcwd(), "human_data", "models", model_name, "live.mp4")
    r = Runner("0", "0,0", Queue(), None)
    r.model_version = model_name
    r.args.face = face
    r.load_video()
    r.start()
    print("[local_engine] 渲染子进程就绪,推流中: /live/0", flush=True)
    while True:
        try:
            text = tts_q.get(timeout=1)
        except Exception:
            continue
        if text is None:
            break
        if not str(text).strip():
            continue
        print("[local_engine] 合成并渲染:", str(text)[:40], flush=True)
        try:
            wav = qfttsClone(voice_dir, text, 1)
            r.run(wav, None)
        except Exception as exc:
            print("[local_engine] 渲染失败:", exc, flush=True)


def start_live(model: str, voice: str) -> dict:
    """开播:拉起 SRS/TTS/渲染子进程。重复调用幂等(先停再起);
    T-027 (R-1D-005): starting 态下并发调用直接拒绝,防半初始化/误杀。"""
    global _starting
    with _state_lock:
        if _starting:                       # T-027: in-flight 防重入(契约允许 start 附 error 键)
            return {"ok": False, "error": "启动进行中,请勿重复操作"}
        _starting = True
        _stop_live_unlocked()               # T-027: 幂等先停复用无锁内部实现(禁嵌套加锁)
    try:
        if not ensure_srs():
            return {"ok": False, "error": "SRS 启动失败"}
        if not ensure_tts():
            return {"ok": False, "error": "TTS 启动失败(%s)" % _cfg.tts_port}  # T-058

        ctx = mp.get_context("spawn")
        runner_q = ctx.Queue()
        mqtt_q = ctx.Queue()
        tts_q = ctx.Queue()
        voice_dir = os.path.join(RUN_ROOT, "human_data", "voices", voice)
        proc = ctx.Process(target=_render_child,
                           args=(runner_q, mqtt_q, tts_q, model, voice_dir),
                           daemon=True)
        proc.start()
        with _state_lock:                   # T-027: 四项登记收进同一临界区
            _state["proc"] = proc
            _state["tts_q"] = tts_q
            _state["model"] = model
            _state["voice"] = voice
        return {"ok": True, "stream": "live/0"}
    finally:
        with _state_lock:                   # T-027: 异常/提前返回也必须清账
            _starting = False


def speak(text: str) -> dict:
    # T-027 (R-1D-005): 检查与 put 收进同一临界区,消除"取到旧队列后 stop_live
    # 置空、put 进死队列丢消息"的竞态;mp.Queue.put 为 feeder 缓冲写,停留极短;
    # is_alive() 为非阻塞轮询,不构成持锁长停。
    with _state_lock:
        q = _state.get("tts_q")
        proc = _state.get("proc")
        if q is None or proc is None or not proc.is_alive():
            return {"ok": False, "error": "引擎未开播"}
        q.put(text)
    return {"ok": True}


def _stop_live_unlocked() -> dict:
    """T-027 (R-1D-005): stop_live 的无锁内部实现,调用方必须已持 _state_lock;
    单独存在只为让持锁的 start_live 复用先停逻辑,避免嵌套加锁。"""
    proc = _state.get("proc")
    if proc is not None:
        try:
            me = os.getpid()
            for p in psutil.process_iter(["pid", "ppid", "name"]):
                try:
                    if p.info["ppid"] == proc.pid or p.info["pid"] == proc.pid:
                        psutil.Process(p.info["pid"]).kill()
                except Exception:
                    pass
            proc.kill()
        except Exception:
            pass
    _state["proc"] = None
    _state["tts_q"] = None
    # R-1D-002 / T-003: 下播后清空 model/voice 登记(置 None 而非删键,
    # status() 字段签名不变),配合 app_gui._resource_in_use 的 running
    # 双条件,使资源下播后可立即删除/改名。清理位于上方进程清理块
    # (整体 try/except 兜底)之后的必经路径,不会被跳过,也不掩盖下播结果。
    _state["model"] = None
    _state["voice"] = None
    return {"ok": True}


def stop_live() -> dict:
    # T-027 (R-1D-005): _state 读写持锁;进程收割体在 _stop_live_unlocked
    with _state_lock:
        return _stop_live_unlocked()


def _srs_stream_name() -> str:
    """向 SRS 查询当前活跃流名(如 live/0,0,0);查不到回落 live/0。"""
    try:
        import urllib.request
        raw = urllib.request.urlopen(
            "http://127.0.0.1:%d/api/v1/streams/" % SRS_API, timeout=3).read()
        data = json.loads(raw)
        streams = data.get("streams") or []
        if streams:
            return streams[0]["url"].lstrip("/")
    except Exception:
        pass
    return "live/0"


def status() -> dict:
    # T-027 (R-1D-005): _state 读取持锁做快照;SRS 流名探测(3s 超时)与端口
    # 探测刻意留在锁外,不阻塞 start/speak 路径;字段集不变(契约 §1.1 恒含
    # running/model/voice/ports/stream_name/stream_url)。
    with _state_lock:
        proc = _state.get("proc")
        model = _state["model"]
        voice = _state["voice"]
    alive = bool(proc and proc.is_alive())
    name = _srs_stream_name()
    return {"running": alive, "model": model, "voice": voice,
            "srs": port_in_use(SRS_RTMP), "tts": port_in_use(_cfg.tts_port),
            "stream_name": name,
            "stream_url": "http://127.0.0.1:%d/%s.flv" % (SRS_HTTP, name),
            "ports": {"rtmp": SRS_RTMP, "api": SRS_API, "http": SRS_HTTP}}
