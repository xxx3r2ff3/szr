# -*- coding: utf-8 -*-
"""chat_infer —— R022 语义重建(源 modules/core/chat_infer.cp310-win_amd64.pyd)。

E1 证据(VM 内 oracle 只读探针,2026-09-10 冻结基线 v6.18.0):
* `evidence/modules/core__chat_infer/runtime/probeA_surface.py` → probeA_out.json:
  导入面/身份/纯函数(`dir()` 57 名、`vars()` 定义序、导入期零网络);
* `runtime/probeC_frames.py` → probeC_out.json:frame/audio 索引族逐点行为 +
  **oracle 真实回溯里的 `chat_infer.py` 行号**(行号级证据);
* `runtime/recon_impl_notes.md`:r2 反汇编(`static/r2/infer_core.txt`)+ Ghidra
  反编译(`static/pseudocode/*.c`,239 份)+ PyMethodDef 方法表逐函数复原笔记。

模块级语句顺序 = `vars(ci)` 插入序(probeA)× `__pyx_pymod_exec` 的
`__Pyx_Import/ImportFrom + PyDict_SetItem` 序(笔记 §1)双路一致。
"""
import base64  # noqa: F401  (py1;DH._decrypt_model 用)
import gc  # noqa: F401  (py2;Runner.clear 用)
import json
import math  # noqa: F401  (py4;UHM.get_face_mask2 用)
from queue import Queue, PriorityQueue  # py5(E1 probeA:Queue 的 __module__)
import subprocess  # noqa: F401  (py6;Runner.start 推流用)
import sys  # noqa: F401  (py7;UHM.__init__ 用)
import threading
import time  # noqa: F401  (py9)
import timeit  # noqa: F401  (py10;wav_run 计时)
import wave  # noqa: F401  (py11;Runner.run)
import librosa  # noqa: F401  (py12)
import numpy as np
import cv2  # noqa: F401  (py14,与 os/argparse 同一物理行)
import os
import argparse  # noqa: F401  (py14;E1:oracle 里是死导入,但 dir() 必须一致)
from tqdm import tqdm  # noqa: F401  (py15)
import torch
import soundfile as sf
import onnxruntime  # noqa: F401  (py18)

from modules.core.app_util import (      # py19(再导出:E1 probeA is 恒等)
    cut_text,                            # py20
    qfttsClone,                          # py21
    only_punc,                           # py22
    text2edgevoice,                      # py23
)                                        # py24

from modules.core.config import (        # py25
    voices_dir,                          # py26
    srs_host,                            # py27
    rtmp_port,                           # py28
    infer_width,                         # py29
    temp_dir,                            # py30
    bitrate,                             # py31
    api_host,                            # py32
)                                        # py33

from modules.core.util import (          # py34
    NamedPipe,                           # py35(E1:本模块不定义,来自 util)
    print_red,                           # py36
    print_yellow_tip,                    # py37
)                                        # py38

from cryptography.fernet import Fernet  # py39
import torch.nn.functional as F  # noqa: F401,E402  (py40;E1 probeA F=torch.nn.functional)


def remove_silent_edges_fast(input_path, output_path, threshold=0.01,
                             chunk_size=2048, keep_silence=0.2):     # py43
    """裁掉音频首尾静音,首尾各保留 keep_silence 秒。[pyd 体 FUN_180001c20,py43-66]

    行号证据(recon §2.12):py47 `sf.read`、py51/59 两个 `range`、py52/60 分块、
    py53/61 两个 genexpr(自由变量字面量证明为 `chunk`/`threshold`,体
    `abs(x) > threshold`,外层 `any(...)`)、py54/62 的 max/min 三元、py66 `sf.write`。
    probeC:文件不存在 → `soundfile.LibsndfileError`,栈顶 py47。
    """
    audio, sr = sf.read(input_path)                                  # py47
    start = 0
    for i in range(0, len(audio), chunk_size):                       # py51
        end = min(i + chunk_size, len(audio))                        # py52
        chunk = audio[i:end]
        if any(abs(x) > threshold for x in chunk):                   # py53
            start = max(0, i - int(sr * keep_silence))               # py54
            break
    end = len(audio)                                                 # py58
    for i in range(len(audio) - 1, start - 1, -chunk_size):          # py59
        chunk = audio[max(i - chunk_size, 0):i]                      # py60
        if any(abs(x) > threshold for x in chunk):                   # py61
            end = min(i + int(sr * keep_silence), len(audio))        # py62
            break
    sf.write(output_path, audio[start:end], sr)                      # py66


def get_frame_index(args):          # py69
    """[pyd METH_O 0x180003810] probeC:单次读 `frame_idx` 原样返回。"""
    return args.frame_idx           # py70


def next_frame_index(args):                       # py73
    """帧号 +1,越界回绕 0。[pyd METH_O 0x180003870,py73-78]

    probeC 行号证据:py74(+1,写在局部量里,异常路径不改写 args)、py75(`>=` 比较)、
    py76 回绕常量 0、py78 写回。
    """
    frame_index = args.frame_idx + 1              # py74
    if frame_index >= args.frame_len:             # py75
        args.frame_idx = 0                        # py76
    else:                                         # py77
        args.frame_idx = frame_index              # py78


def get_wait_frame_index(args):     # py81
    """[pyd METH_O 0x180003ac0] probeC:单次读 `wait_frame_idx`。"""
    return args.wait_frame_idx      # py82


def next_wait_frame_index(args):                  # py85
    """等待帧号 +1,越界回绕 0。[pyd METH_O 0x180003b20,py85-90]"""
    wait_frame_index = args.wait_frame_idx + 1    # py86
    if wait_frame_index >= args.wait_frame_len:   # py87
        args.wait_frame_idx = 0                   # py88
    else:                                         # py89
        args.wait_frame_idx = wait_frame_index    # py90


# E1(py93/94):device 由 torch.cuda.is_available() 三元得到;随后打印。
device = "cuda" if torch.cuda.is_available() else "cpu"               # py93
print("Using {} for inference.".format(device))                       # py94


class UHM(object):
    """人脸/上半身推理封装。[pyd py97-619;方法表见 static/method_table.json]

    E1:导入期不实例化;`__init__` 体使用 print_red,`_get_providers` 使用
    print_yellow_tip(笔记 §4.1)。方法体未逐条展开(登记未定谳)。
    """

    def __init__(self, model_path, device='cuda', gpu_id=0, fernet_key=None):  # py99
        self.model_path = model_path
        self.device = device
        self.gpu_id = gpu_id
        self.fernet_key = fernet_key

    def _decrypt_model(self, encrypted_model_path):          # py136
        with open(encrypted_model_path, 'rb') as f:
            encrypted_data = f.read()
        cipher_suite = Fernet(self.fernet_key)
        return cipher_suite.decrypt(encrypted_data)

    def _get_providers(self, device, gpu_id):                # py143
        providers = []
        if device == 'cuda':
            providers.append(('CUDAExecutionProvider', {'device_id': gpu_id}))
        providers.append('CPUExecutionProvider')
        return providers

    def _get_session_options(self):                          # py168
        sess_options = onnxruntime.SessionOptions()
        return sess_options

    def _create_session_from_bytes(self, model_data, device, gpu_id):   # py177
        return onnxruntime.InferenceSession(
            model_data, sess_options=self._get_session_options(),
            providers=self._get_providers(device, gpu_id))

    def _create_session_from_file(self, model_path, device, gpu_id):    # py184
        return onnxruntime.InferenceSession(
            model_path, sess_options=self._get_session_options(),
            providers=self._get_providers(device, gpu_id))

    def feature_extraction_wenet(self, *args, **kwargs):     # py191
        raise NotImplementedError('UHM.feature_extraction_wenet 未定谳')

    def tensor_norm_no_training(self, *args, **kwargs):      # py216
        raise NotImplementedError('UHM.tensor_norm_no_training 未定谳')

    def get_face_mask(self, *args, **kwargs):                # py222
        raise NotImplementedError('UHM.get_face_mask 未定谳')

    def get_face_mask2(self, *args, **kwargs):               # py262
        raise NotImplementedError('UHM.get_face_mask2 未定谳')

    def gaussian_blur_batch(self, *args, **kwargs):          # py300
        raise NotImplementedError('UHM.gaussian_blur_batch 未定谳')

    def optimized_weight_calculation_gpu_batch(self, *args, **kwargs):   # py328
        raise NotImplementedError('UHM.optimized_weight_calculation_gpu_batch 未定谳')

    def process_xseg_mask(self, *args, **kwargs):            # py362
        raise NotImplementedError('UHM.process_xseg_mask 未定谳')

    def get_complete_imgs(self, *args, **kwargs):            # py377
        raise NotImplementedError('UHM.get_complete_imgs 未定谳')

    def inference(self, frame_list, landmark_list, audio_list, xseg_list):   # py481
        raise NotImplementedError('UHM.inference 未定谳')

    def activate(self):                                      # py604
        raise NotImplementedError('UHM.activate 未定谳')


class DH(object):
    """数字人(DH)推理封装:与 UHM 同构的精简版。[pyd py620-702]"""

    def __init__(self, model_path, device='cuda', gpu_id=0, fernet_key=None):  # py621
        self.model_path = model_path
        self.device = device
        self.gpu_id = gpu_id
        self.fernet_key = fernet_key

    def _ensure_valid_fernet_key(self, input_key):           # py634
        return base64.urlsafe_b64encode(input_key.ljust(32)[:32].encode())

    def _decrypt_model(self, encrypted_model_path):          # py646
        with open(encrypted_model_path, 'rb') as f:
            encrypted_data = f.read()
        cipher_suite = Fernet(self.fernet_key)
        return cipher_suite.decrypt(encrypted_data)

    def _create_session_from_bytes(self, model_data, device, gpu_id):   # py652
        return onnxruntime.InferenceSession(
            model_data, providers=self._get_providers(device, gpu_id))

    def _create_session_from_file(self, model_path, device, gpu_id):    # py659
        return onnxruntime.InferenceSession(
            model_path, providers=self._get_providers(device, gpu_id))

    def _get_session_options(self):                          # py666
        return onnxruntime.SessionOptions()

    def _get_providers(self, device, gpu_id):                # py675
        providers = []
        if device == 'cuda':
            providers.append(('CUDAExecutionProvider', {'device_id': gpu_id}))
        providers.append('CPUExecutionProvider')
        return providers

    def inference(self, img, audio):                         # py687
        raise NotImplementedError('DH.inference 未定谳')

    def activate(self):                                      # py697
        raise NotImplementedError('DH.activate 未定谳')


class Runner(object):
    """帧/音频推进器。[pyd py722-1084]

    E1(py723 `__init__`):建 `Queue()` 与 `PriorityQueue()` 成员;
    `load_video` 用 cv2.VideoCapture + Thread;`start` 用 NamedPipe + ffmpeg Popen
    推 `rtmp://{srs_host}:{rtmp_port}/live/{deviceId}`(网络路径,不执行)。
    """

    def __init__(self, deviceId, inferId, runner_queue, onnx=None):      # py723
        self.deviceId = deviceId
        self.inferId = inferId
        self.runner_queue = runner_queue
        self.onnx = onnx
        self.frame_queue = Queue()
        self.wav_queue = PriorityQueue()
        self.stop = False
        self.interrupting = False
        self.is_waiting = False
        self.full_frames = []
        self.full_landmarks = []
        self.wait_frames = []
        self.frame_idx = 0
        self.frame_len = 0
        self.wait_frame_idx = 0
        self.wait_frame_len = 0
        self.frame_w = 0
        self.frame_h = 0

    def clear(self):                                         # py755
        gc.collect()

    def load_frame(self, *args, **kwargs):                   # py760
        raise NotImplementedError('Runner.load_frame 未定谳')

    def load_video(self, *args, **kwargs):                   # py786
        raise NotImplementedError('Runner.load_video 未定谳')

    def start(self):                                         # py813
        raise NotImplementedError('Runner.start 未定谳')

    def run(self, wav_path, wavhu):                          # py936
        raise NotImplementedError('Runner.run 未定谳')


def get_audio_features(features, index):                       # py703
    """取以 index 为中心、宽 16 帧的特征窗并按需零填充。[pyd 体 0x1800282a0,py703-718]

    probeC 行号证据:py711 `features.shape[0]`(None → TypeError)、py714 对
    `features` 的切片(`mp_subscript` → 'Rec' object is unsliceable);
    笔记 §2.11 证明 704 为 `-8`、705 为 `+8`、716/718 为两次 `torch.cat(dim=0)`。
    """
    start = index - 8                                          # py704
    end = index + 8                                            # py705
    left_pad = 0                                               # (局部量初值;Cython 保证已初始化)
    right_pad = 0
    if start < 0:                                              # py708
        start, left_pad = 0, -start                            # py709
    if end > features.shape[0]:                                # py711
        right_pad = end - features.shape[0]                    # py712
        end = features.shape[0]                                # py713
    audio = torch.from_numpy(features[start:end])              # py714
    if left_pad > 0:                                           # py715
        audio = torch.cat([torch.zeros_like(audio[:left_pad]), audio], dim=0)   # py716
    if right_pad > 0:                                          # py717
        audio = torch.cat([audio, torch.zeros_like(audio[:right_pad])], dim=0)  # py718
    return audio                                               # py719


def run_process(runner_queue, mqtt_queue, tts_queue, deviceId, inferId):   # py1085
    """聊天推理工作进程主体:两条队列线程 + 监督循环。[pyd 体 0x180042b50,py1085-1276]

    行号证据(笔记 §3):py1092 `inferId.split(',')`、py1093
    `torch.cuda.set_device(int(...))`、py1094 `Runner(...)`、py1095 `Queue()`、
    py1096 七标点集合、py1098/1151 两个嵌套函数、py1149/1201 两个 daemon 线程。
    py1203-1276 的监督循环只到“行号 + 被调名字”级 → 尾部登记未定谳。
    """
    # py1092 是 Cython 编译的二元解包(`a, b = <list>`):异常文案走 Cython 自带的
    # __Pyx_RaiseNeedMoreValuesError / __Pyx_RaiseTooManyValuesError(=CPython≤3.9 旧式
    # 文案),不是 3.10 的 "not enough values to unpack"。E1 证据:run_process 用例
    # golden 观测 results.locals.bad_infer_id = "ValueError: need more than 1 value to
    # unpack"(inferId="no-comma" → split 得 1 元)。故按 Cython 语义手工复刻:
    _parts = inferId.split(",")                            # py1092
    if len(_parts) < 2:    # __Pyx_RaiseNeedMoreValuesError(len):1→"1 value",其余→"N values"
        raise ValueError("need more than %d value%s to unpack"
                         % (len(_parts), "" if len(_parts) == 1 else "s"))
    if len(_parts) > 2:    # __Pyx_RaiseTooManyValuesError(2)
        raise ValueError("too many values to unpack (expected 2)")
    deviceId, gpu_id = _parts                              # py1092(解包本体)
    torch.cuda.set_device(int(deviceId))                   # py1093
    runner = Runner(deviceId, inferId, runner_queue, None)  # py1094
    wav_queue = Queue()                                     # py1095
    punds = {';', '?', '!', '。', '？', '！', '…'}           # py1096

    def wav_genera():                                       # py1098
        while True:                                         # (循环头行号未定谳)
            try:
                voice, text, voiceSpeed = tts_queue.get(timeout=1)   # py1101
            except Exception:
                continue
            text = cut_text(text, punds=punds)
            if only_punc(text):
                continue
            wav_path = os.path.join(voices_dir, voice)
            if voice == 'edgetts':
                wav = text2edgevoice(False, text)
            else:
                wav = qfttsClone(voice, text, speed=voiceSpeed)
            remove_silent_edges_fast(wav, wav)
            duration = librosa.get_duration(path=wav, sr=16000)
            wav_queue.put_nowait((voice, text, wav, duration, voiceSpeed, 'normal'))

    threading.Thread(target=wav_genera, daemon=True).start()    # py1149

    def wav_run():                                          # py1151
        # [I009 E2E 定谳 i009-fix-2] oracle 的 wav_run 消费 run_process 内部自建的
        # wav_queue(probe_i009_qtrace:wav_run 阻塞在 get(True, None) 于 py1095 新建
        # 的内部队列;入参 mqtt_queue 无任何 get——它是出站发布队列,chat_start/
        # chat_end 经 put 进 mqtt_queue 由 chat_gui.send_mqtt 发布)。原重建误写为
        # 消费 mqtt_queue,与 oracle 接线相反。
        while True:                                         # (循环头行号未定谳)
            item = wav_queue.get()
            if item is None:
                continue
            mqtt_queue.put_nowait((topic, json.dumps({'type': 'chat_start'})))   # py1164
            runner.run(item, None)
            mqtt_queue.put_nowait((topic, json.dumps({'type': 'chat_end'})))     # py1188

    threading.Thread(target=wav_run, daemon=True).start()       # py1201
    # [I009 E2E 定谳 i009-fix-3] oracle stdout 恒有此行(probe_order/probe_qtrace
    # 双轮:'--- 启动推理进程成功 --- 0,0'),原重建缺失。
    print('--- 启动推理进程成功 --- ' + inferId)
    topic = "chat/" + inferId + "/client"                       # py1203
    while True:                                                 # py1205(尾部未定谳)
        time.sleep(0.1)
