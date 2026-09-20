# -*- coding: utf-8 -*-
"""w2l.load —— R019(语义重建自 modules/w2l/load.cp310-win_amd64.pyd)。

源 pyd:modules/w2l/load.cp310-win_amd64.pyd
        sha256 c0bd838dce56aca711c2203adbce17971a5abb50c70471d591b582e5408a1297
反编译证据:evidence/modules/w2l__load/static/pseudocode/
  * FUN_1800201d0 = __pyx_pymod_exec_load(模块级语句);PyInit_load 只是 PyModuleDef_Init;
  * FUN_18001e6a0 = __Pyx_InitCachedConstants(每函数 PyCode_New 的 varnames + firstlineno);
  * FUN_18001fc00 = StringTab + int/float 常量池;
  * 各函数一个 .c(文件名见行内注释),DAT_ → STR("…") 槽位解析。
静态重建片段(逐语句 py 行号 + 槽位级出处):
  static/recon_module_head.py(import 区/模块级/merge_namespaces/parse_args)
  static/recon_small.py(laplacianSmooth/blendface/adjust_gamma/get_smoothened_boxes)
  static/recon_data.py(get_wav_mel/datagen/face_detect)

E1 证据(VM oracle 侧 probe_e1_load{,2,3,4,5,6,7}.py + _out.json,
归档 evidence/modules/w2l__load/runtime/):
  * 导入期即 `YOLO("checkpoints/yolov8n-face.pt")`(缺文件 → FileNotFoundError)、
    `torch.manual_seed(1234)`、`face_border = cv2.imread("checkpoints/border.png")`;
  * parse_args 12 字段实测与静态推导逐字段一致;
  * adjust_gamma 全 LUT 实测(convertScaleAbs(alpha=1.2) 后按 gamma 建表);
  * blendface 逐像素权重实测;laplacianSmooth 首帧返回副本、后续 max/min 平滑;
  * merge_namespaces 实测 vars()/Namespace;
  * datagen:`frames.get()` 0 参调用、4 元解包、白图占位分支、批次 yield 结构;
  * W2lOnnx 用**自建微型 ONNX 模型**驱动(inference/activate 常量输出可复现);
  * load_onnx:实测 onnx.load 走**硬编码** "modules/w2l/model_fp16.onnx",
    img_size = 该模型 graph.input[1] 的**最后一个** dim(模型维度实验锁定);
  * load_video:视频缺失 → 返回 (frame_len, frame_w, frame_h) = (0,0,0) 且
    runner.full_landmarks = {};视频存在时进入 face_detect → 本机无 CUDA 必抛
    ValueError(见报告"未定谳项");
  * infer:首访问 runner.fps,缺 wav → FileNotFoundError。

覆盖单元:14 个 block(每函数 B0);无分支边、无状态机。
"""
import argparse  # [pyd FUN_1800201d0 py=1]
import os  # [pyd FUN_1800201d0 py=2]
import pathlib  # [pyd FUN_1800201d0 py=3]
import pickle  # [pyd FUN_1800201d0 py=4]
import threading  # [pyd FUN_1800201d0 py=5]
import wave  # [pyd FUN_1800201d0 py=6]
import cv2  # [pyd FUN_1800201d0 py=7]
import librosa  # [pyd FUN_1800201d0 py=8]
import numpy as np  # [pyd FUN_1800201d0 py=9]
import onnx  # [pyd FUN_1800201d0 py=10]
import onnxruntime  # [pyd FUN_1800201d0 py=11]
import torch  # [pyd FUN_1800201d0 py=12]
from tqdm import tqdm  # [pyd FUN_1800201d0 py=13]
from ultralytics import YOLO  # [pyd FUN_1800201d0 py=14]

# [pyd FUN_1800201d0 py=15 空行]
# [pyd FUN_1800201d0 py=16:fromlist=['print_yellow_tip','setInterval'],本模块无同名 def]
from modules.core.util import print_yellow_tip, setInterval
import modules.w2l.audio as audio  # [pyd FUN_1800201d0 py=17]

# [pyd FUN_1800201d0 py=18 空行]
# [pyd FUN_1800201d0 py=19;tuple DAT_180033e68=("checkpoints/yolov8n-face.pt",)]
face_det = YOLO("checkpoints/yolov8n-face.pt")
torch.manual_seed(1234)  # [pyd FUN_1800201d0 py=20;const 0x4d2=1234]
face_border = cv2.imread("checkpoints/border.png")  # [pyd FUN_1800201d0 py=21]

# [pyd FUN_1800201d0 py=22/23 空行注释]


class laplacianSmooth:  # [pyd co_firstlineno=24]
    def __init__(self, smoothAlpha=0.3):  # [pyd FUN_1800010d0 py=25;默认 0.3=cached float]
        self.smoothAlpha = smoothAlpha  # [pyd py=26]
        self.pts_last = None  # [pyd py=27]

    def smooth(self, pts_cur):  # [pyd FUN_180001570 py=29;varnames 见 recon_small.py]
        if self.pts_last is None:  # [pyd py=30]
            self.pts_last = pts_cur.copy()  # [pyd py=31]
            return pts_cur.copy()  # [pyd py=32] 返回副本(原实现怪写法,保留)
        x1 = min(pts_cur[:, 0])  # [pyd py=33]
        x2 = max(pts_cur[:, 0])  # [pyd py=34]
        y1 = min(pts_cur[:, 1])  # [pyd py=35]
        y2 = max(pts_cur[:, 1])  # [pyd py=36]
        width = x2 - x1  # [pyd py=37]
        pts_update = []  # [pyd py=38]
        for i in range(len(pts_cur)):  # [pyd py=39]
            x_new, y_new = pts_cur[i]  # [pyd py=40]
            x_old, y_old = self.pts_last[i]  # [pyd py=41]
            tmp = (x_new - x_old) ** 2 + (y_new - y_old) ** 2  # [pyd py=42]
            w = np.exp(-tmp / (width * self.smoothAlpha))  # [pyd py=43]
            x = x_old * w + x_new * (1 - w)  # [pyd py=44]
            y = y_old * w + y_new * (1 - w)  # [pyd py=45]
            pts_update.append([x, y])  # [pyd py=46]
        pts_update = np.array(pts_update)  # [pyd py=47]
        self.pts_last = pts_update.copy()  # [pyd py=48]
        return pts_update  # [pyd py=49]


def blendface(oldface, newface):  # [pyd FUN_180002f70 py=53]
    mask = np.zeros(newface.shape, dtype=np.uint8)  # [pyd py=54] 立即被下一行覆盖(原缺陷保留)
    mask = cv2.resize(face_border, (newface.shape[1], newface.shape[0]))  # [pyd py=55]
    oldface_mask = np.ones(newface.shape, dtype=np.uint8) * 255  # [pyd py=57]
    oldface_mask = oldface_mask - mask  # [pyd py=58]
    mask = mask.astype("float")  # [pyd py=59]
    mask = mask / 255  # [pyd py=60]
    newface = newface * mask  # [pyd py=61]
    oldface = oldface * (oldface_mask / 255)  # [pyd py=62]
    newface = newface.astype(np.uint8)  # [pyd py=63]
    oldface = oldface.astype(np.uint8)  # [pyd py=64]
    face = cv2.addWeighted(newface, 1, oldface, 1, 0)  # [pyd py=65]
    return face  # [pyd py=66]


def adjust_gamma(image, gamma=1.0):  # [pyd FUN_1800047e0 py=69;默认 1.0=cached float]
    image = cv2.convertScaleAbs(image, alpha=1.2, beta=0)  # [pyd py=70;const 1.2]
    inv_gamma = 1.0 / gamma  # [pyd py=71]
    table = np.array(  # [pyd py=72]
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]  # [pyd py=73]
    ).astype("uint8")  # [pyd py=74]
    return cv2.LUT(image, table)  # [pyd py=75]


def get_smoothened_boxes(boxes, T):  # [pyd FUN_180005790 py=78]
    for i in range(len(boxes)):  # [pyd py=79]
        if i + T > len(boxes):  # [pyd py=80]
            window = boxes[len(boxes) - T:]  # [pyd py=81]
        else:
            window = boxes[i: i + T]  # [pyd py=83]
        boxes[i] = np.mean(window, axis=0)  # [pyd py=84]
    return boxes  # [pyd py=85]


def face_detect(args, images):  # [pyd FUN_180006300 py=88;varnames 18 名见 recon_data.py]
    abox_smoother = laplacianSmooth()  # [pyd py=89]
    predictions = []  # [pyd py=90]
    for frame in images:  # [pyd py=91]
        results = face_det(  # [pyd py=92]
            frame,  # [pyd py=93]
            imgsz=640,  # [pyd py=94]
            conf=0.01,  # [pyd py=95]
            iou=0.5,  # [pyd py=96]
            half=True,  # [pyd py=97]
            augment=False,  # [pyd py=98]
            device='cuda',  # [pyd py=99]
            verbose=False,  # [pyd py=100]
        )
        boxes = results[0].boxes  # [pyd py=101]
        try:  # [pyd py=102]
            f = boxes.xyxy.cpu().numpy()[0]  # [pyd py=104]
            f = abox_smoother.smooth(np.reshape(f, (-1, 2))).reshape(-1)  # [pyd py=105]
            predictions.append(f.astype(int))  # [pyd py=106]
        except:  # [pyd py=107] 裸 except(原样保留)
            predictions.append(np.array([0, 0, 0, 0]))  # [pyd py=108]
    results = []  # [pyd py=110]
    pady1, pady2, padx1, padx2 = args.pads  # [pyd py=111]
    for rect, image in zip(predictions, images):  # [pyd py=112]
        if rect is None:  # [pyd py=113]
            cv2.imwrite('temp/faulty_frame.jpg', image)  # [pyd py=114]
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')  # [pyd py=117]
        y1 = max(rect[1] - pady1, 0)  # [pyd py=121]
        y2 = min(rect[3] + pady2, image.shape[0])  # [pyd py=122]
        x1 = max(rect[0] - padx1, 0)  # [pyd py=123]
        x2 = min(rect[2] + padx2, image.shape[1])  # [pyd py=124]
        results.append([x1, y1, x2, y2])  # [pyd py=126]
    boxes = np.array(results)  # [pyd py=128]
    if not args.nosmooth:  # [pyd py=129]
        boxes = get_smoothened_boxes(boxes, T=5)  # [pyd py=130]
    return boxes  # [pyd py=131]


class W2lOnnx:  # [pyd co_firstlineno=159;varnames 8 名见 FUN_180009870]
    def __init__(self, model_path, device='cuda', gpu_id=0, trt=False):  # [pyd py=160]
        session_options = onnxruntime.SessionOptions()  # [pyd py=167]
        providers = ['CPUExecutionProvider']  # [pyd py=173]
        if trt:  # [pyd py=174]
            providers = [  # [pyd py=177]
                ('CUDAExecutionProvider',
                 {'cudnn_conv_algo_search': 'DEFAULT', 'device_id': gpu_id}),  # [pyd py=178]
                'CPUExecutionProvider',
            ]
            cache_dir = '.trtcache'  # [pyd py=182]
            providers = [  # [pyd py=189]
                ('TensorrtExecutionProvider',
                 {'device_id': gpu_id,
                  'trt_fp16_enable': True,
                  'trt_engine_cache_enable': True,
                  'trt_engine_cache_path': cache_dir,
                  'trt_profile_min_shapes': 'mel_batch:1x1x80x16,img_batch:1x6x256x256,a_alpha:1',
                  'trt_profile_max_shapes': 'mel_batch:8x1x80x16,img_batch:8x6x256x256,a_alpha:8',
                  'trt_profile_opt_shapes': 'mel_batch:4x1x80x16,img_batch:4x6x256x256,a_alpha:4'}),
                'CPUExecutionProvider',
            ]
        self.session = onnxruntime.InferenceSession(  # [pyd py=201]
            model_path, sess_options=session_options, providers=providers)

    def inference(self, mel_batch, img_batch, a_alpha=[1.0]):  # [pyd FUN_18000a5c0 py=204]
        output = self.session.run(  # [pyd py=205]
            ['pred'],  # [pyd py=205] 输出名硬编码(实测 Invalid output name:pred)
            {'mel_batch': mel_batch, 'img_batch': img_batch, 'a_alpha': a_alpha})  # [pyd py=207]
        return output[0]  # [pyd py=208] 返回首个输出张量(E1:np.asarray 得 4 维,非 run 的 list)

    def activate(self, batch_size=1):  # [pyd FUN_18000ab00 py=211]
        mel_shape = self.session.get_inputs()[0].shape  # [pyd py=212]
        img_shape = self.session.get_inputs()[1].shape  # [pyd py=213]
        mel_batch = np.random.random_sample(  # [pyd py=215/218]
            (batch_size, mel_shape[1], mel_shape[2], mel_shape[3])).astype(np.float32)
        img_batch = np.random.random_sample(  # [pyd py=219]
            (batch_size, img_shape[1], img_shape[2], img_shape[3])).astype(np.float32)
        return self.inference(mel_batch=mel_batch, img_batch=img_batch)  # [pyd py=220]


def merge_namespaces(ns1, ns2):  # [pyd FUN_18000bc00 py=223]
    merged_dict = {**vars(ns1), **vars(ns2)}  # [pyd py=224]
    return argparse.Namespace(**merged_dict)  # [pyd py=225]


def load_video(runner, load_frame=True):  # [pyd FUN_18000c1e0 py=228;varnames 20 名]
    args = merge_namespaces(parse_args(), runner.args)  # [pyd py=229/py=231]
    runner.args = args  # [pyd py=231]
    filename = pathlib.Path(args.face).stem  # [pyd py=233]
    dirname = os.path.dirname(args.face)  # [pyd py=234]
    facebox_pklpath = os.path.join(os.path.dirname(args.face), '_facebox.pkl')  # [pyd py=235]
    full_boxes = []  # E1:视频缺失时 runner.full_landmarks 实测为 list
    if os.path.exists(facebox_pklpath):  # [pyd py=236]
        with open(facebox_pklpath, 'rb') as f1:  # [pyd py=237]
            full_boxes = pickle.load(f1)  # [pyd py=238]
    video_stream = cv2.VideoCapture(args.face)  # [pyd py=240]
    frame_len = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))  # [pyd py=241]
    frame_w = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))  # [pyd py=242]
    frame_h = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))  # [pyd py=243]
    print('Reading video boxes...')  # [pyd py=244;cached tuple DAT_180033eb8 + 内建 print]
    if frame_len != len(full_boxes):  # [pyd py=247] RichCompare op=3('!=')
        pbar = tqdm(  # [pyd py=248]
            total=frame_len,  # [pyd py=249]
            bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        face_boxes = []
        while True:
            still_reading, frame = video_stream.read()  # [pyd py=253]
            if not still_reading:  # [pyd py=253]
                video_stream.release()  # [pyd py=254]
                break
            if args.resize_factor > 1:  # [pyd py=257]
                frame = cv2.resize(frame, (frame_w // args.resize_factor,  # [pyd py=258/261]
                                           frame_h // args.resize_factor))  # [pyd py=262]
            if args.rotate:  # [pyd py=266]
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # [pyd py=267]
            y1, y2, x1, x2 = args.crop  # [pyd py=269]
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]
            frame = frame[y1:y2, x1:x2]
            face_boxes.append(frame)  # [pyd py=270]
            pbar.update(1)
        pbar.close()
        full_boxes = face_detect(args, face_boxes).tolist()  # [pyd py=275/276]
        with open(facebox_pklpath, 'wb') as f1:  # [pyd py=282]
            pickle.dump(full_boxes, f1)
        threading.Thread(target=load_frame, daemon=True).start()  # [pyd py=289]
        setInterval(torch.cuda.empty_cache, 10)  # [pyd py=290]
    runner.full_landmarks = full_boxes  # [pyd py=282 setattr runner.full_landmarks]
    return frame_len, frame_w, frame_h  # [pyd py=287/291 三元组]


def load_onnx(runner, onnx_file, device, trt, load_frame=True):  # [pyd FUN_180011720 py=294]
    if trt:  # [pyd py=296]
        print_yellow_tip('开始显卡加速，此过程第一次预计耗时10-30分钟，请耐心等待！')  # [pyd py=297]
    # [pyd py=298:实参 = ("modules/w2l/model_fp16.onnx", device, runner.gpu_id, trt)]
    runner.onnx = W2lOnnx("modules/w2l/model_fp16.onnx", device, runner.gpu_id, trt)
    input_model = onnx.load("modules/w2l/model_fp16.onnx")  # [pyd py=301]
    input_info = input_model.graph.input[1]  # [pyd py=302]
    input_shape = input_info.type.tensor_type.shape  # [pyd py=303]
    runner.args.img_size = [dim.dim_value for dim in input_shape.dim][-1]  # [pyd py=304]


def datagen(args, frames, mels):  # [pyd FUN_1800129b0 py=319;生成器,varnames 19 名]
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []  # [pyd py=320]
    for i, m in enumerate(mels):  # [pyd py=322]
        idx, coord, frame, gap_len = frames.get()  # [pyd py=326] 0 参调用(实测)
        if np.any(np.array(coord) == 0):  # [pyd py=332]
            face = np.ones((args.img_size, args.img_size, 3), dtype=np.uint8) * 255  # [pyd py=333]
        else:
            x1, y1, x2, y2 = coord  # [pyd py=335]
            face = frame[y1:y2, x1:x2]  # [pyd py=336]
            face = cv2.resize(face, (args.img_size, args.img_size))  # [pyd py=337]
        img_batch.append(face)  # [pyd py=342]
        mel_batch.append(m)  # [pyd py=343]
        frame_batch.append(frame)  # [pyd py=344]
        coords_batch.append(coord)  # [pyd py=345]
        if len(img_batch) >= args.wav2lip_batch_size:  # [pyd py=347]
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)  # [pyd py=348]
            img_masked = img_batch.copy()  # [pyd py=350]
            img_masked[:, args.img_size // 2:] = 0  # [pyd py=351]
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.  # [pyd py=353]
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1],  # [pyd py=354]
                                               mel_batch.shape[2], 1])
            yield img_batch, mel_batch, frame_batch, coords_batch  # [pyd py=355]
            img_batch = []  # [pyd py=358]
            mel_batch, frame_batch, coords_batch = [], [], []  # [pyd py=359]
    if len(img_batch) < 1:  # [pyd py=361]
        return
    img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)  # [pyd py=362]
    img_masked = img_batch.copy()  # [pyd py=364]
    img_masked[:, args.img_size // 2:] = 0  # [pyd py=365]
    img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.  # [pyd py=367]
    mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1],  # [pyd py=368]
                                       mel_batch.shape[2], 1])
    yield img_batch, mel_batch, frame_batch, coords_batch  # [pyd py=372]


def get_wav_mel(fps, wav_path):  # [pyd FUN_180016240 py=375;varnames 10 名]
    try:  # [pyd py=376]
        wav = audio.load_wav(wav_path, 16000)  # [pyd py=377]
        mel = audio.melspectrogram(wav)  # [pyd py=378]
        if np.isnan(mel.reshape(-1)).sum() > 0:  # [pyd py=379]
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
    except Exception as e:  # [pyd py=382]
        print(e)  # [pyd py=383]
        return None  # [pyd py=384]
    mel_step_size = 16  # [pyd py=387 局部常量]
    mel_chunks = []  # [pyd py=388]
    mel_idx_multiplier = 80. / fps  # [pyd py=389]
    i = 0  # [pyd py=390]
    while 1:  # [pyd py=391]
        start_idx = int(i * mel_idx_multiplier)  # [pyd py=392]
        if start_idx + mel_step_size > len(mel[0]):  # [pyd py=393]
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])  # [pyd py=394]
            break  # [pyd py=395]
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])  # [pyd py=396]
        i += 1  # [pyd py=397]
    return mel_chunks  # [pyd py=398]


def infer(runner, wav_path, out, tqdm_desc):  # [pyd FUN_180017be0 py=403;varnames 28 名]
    fps = runner.fps  # [pyd py=405]
    mel_chunks = get_wav_mel(fps, wav_path)  # [pyd py=406]
    # [I007 E2E 定谳 i007-fix-4] librosa.load 的 sr 关键字 = **44100**(w2l 模块 init
    # 常量 DAT_180034068=PyLong_FromLong(0xac44),FUN_18001fc00;infer 内
    # PyDict_SetItem(...,'sr',DAT_180034068)。R019 期旧重建误写 16000
    # (该常量 DAT_180033db0 属 get_wav_mel 的 load_wav);E2E 观测
    # (i007_adjud3)audio_data 步长 1764=44100/25 佐证 44100。
    wav_array, sr = librosa.load(wav_path, sr=44100)  # [pyd py=408]
    wav_array = (wav_array * 32767).astype(np.int16)  # [pyd py=409]
    wav_frame_num = int(len(wav_array) / fps)  # [pyd py=410]
    wf = wave.open(wav_path, 'rb')  # [pyd py=414]
    n_frames = wf.getnframes()  # [pyd py=415]
    wav_index = int(n_frames / fps)  # [pyd py=416]
    gen = datagen(runner.args, runner.full_frames, mel_chunks)  # [pyd py=416/418]
    # E1(probe_e1_load9):oracle 在此处读 runner.full_frames(缺该属性即 AttributeError)
    # [I007 E2E 定谳 i007-fix-4] pbar 批阈值读取的是 **runner.batch_size**(runner 直属
    # 属性,非 runner.args.*):整机 E2E 双侧观测 oracle 报 AttributeError:
    # 'types.SimpleNamespace' object has no attribute 'batch_size'(报错类型指向
    # runner 本体);反编译 1053 行 GetAttr(lVar20,STR("batch_size")) 同证。
    pbar = tqdm(total=int(np.ceil(len(mel_chunks) / runner.batch_size)),  # [pyd py=420/422]
                desc=tqdm_desc)
    #   注:oracle co_varnames 28 名无槽计数局部;以 enumerate 计数实现
    #   "audio_data 逐槽步进",登记 1 个额外局部名的呈现偏差(行为字节级对齐)。
    #   [i007-blend 探针订正] _slot 必须跨批次连续(audio_data = wav_array
    #   [_slot*wav_frame_num:...],dualchk 首轮批内重置被 oracle 证伪)。
    _slot = 0
    for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):  # [pyd py=424]
        img_batch = img_batch.transpose((0, 3, 1, 2)).astype(np.float32)  # [pyd py=426]
        mel_batch = mel_batch.transpose((0, 3, 1, 2)).astype(np.float32)  # [pyd py=427]
        # [I007 E2E 定谳 i007-fix-4] 第三实参 = **[runner.args.a_alpha]**(反编译
        # 1690-1728:GetAttr args → GetAttr a_alpha → PyList_New(1) 包裹后按位置
        # 实参传入;a_alpha 为标量,R019 期旧重建的 runner.args and [1.0] 被证伪)
        pred = runner.onnx.inference(mel_batch, img_batch,  # [pyd py=428]
                                     [runner.args.a_alpha])
        # [I007 E2E 定谳 i007-fix-7b] pred 先 astype(np.uint8) 再 resize(blend5
        # 探针 npy + 宿主本地穷举:uint8-first 对 oracle T 的 mask==255 区命中率
        # 1.0,float-first 仅 0.81,框区差 ±1)。所有下游 resize 的源均为 uint8。
        pred = (pred.transpose((0, 2, 3, 1)) * 255.).astype(np.uint8)  # [pyd py=429]
        # [I007 E2E 定谳 i007-fix-4] stoping 分支 = runner.video_queue.queue.clear()
        # (反编译 2095-2170:GetAttr stoping → GetAttr video_queue → GetAttr queue →
        # GetAttr clear 后调用;R019 期旧重建的 raise ValueError('stoping') 被证伪)
        if runner.stoping:  # [pyd py=430]
            runner.video_queue.queue.clear()  # [pyd py=432/433]
        # [I007 E2E 定谳 i007-fix-7] 循环体逐槽重建(oracle 真值探针 i007_adjud3/
        # i007_decode4,证据 = video_queue.put 三元组与 out.write 逐字节 sig):
        #   * 坐标含 0(白图占位槽)→ frame_speech = f 快照(np.array 拷贝,
        #     decode4 A_puts 偶数槽 is_raw=True 逐字节命中);
        #   * 否则 → cv2.resize(p, (x2-x1, y2-y1)) 后 blendface(f_box, ·) 写回
        #     f[y1:y2, x1:x2](f 被原地改写:adjud3 共享 crop 对象的 put1/put2
        #     同 sig 即此效应),frame_speech = f 快照;
        #   * out is None → wf.readframes(int(n_frames/len(mel_chunks))) 记 chunk
        #     (decode4:obs 727=int(16000/22)、fps=30 组 658=int(11200/17) 双点
        #     确认公式),audio_data = wav16 每槽步进 wav_frame_num(=44100/25=
        #     1764,adjud3/decode4 逐元素相等),video_queue.put((帧, 音频, chunk));
        #   * out 非 None → out.write(cv2.resize(frame_speech, (frame_w, frame_h)))
        #     (decode4 B_writes 形状 (240,320,3));put 与 write 二选一
        #     (decode2/decode4:out 非 None 时 0 put/22 write)。
        #   注:oracle co_varnames 28 名无槽计数局部;以 enumerate 计数实现
        #   "audio_data 逐槽步进",登记 1 个额外局部名的呈现偏差(行为字节级对齐)。
        for p, f, c in zip(pred, frames, coords):  # [pyd py=431]
            x1, y1, x2, y2 = c  # [pyd py=432]
            if np.any(np.array(c) == 0):  # [pyd py=437]
                frame_speech = np.array(f)  # [pyd py=438]
            else:
                face = cv2.resize(p, (x2 - x1, y2 - y1))  # [pyd py=440/441](uint8 源)
                f[y1:y2, x1:x2] = blendface(f[y1:y2, x1:x2], face)  # [pyd py=441/442]
                frame_speech = np.array(f)  # [pyd py=443]
            if out is None:  # [pyd py=451]
                chunk = wf.readframes(int(n_frames / len(mel_chunks)))  # [pyd py=453/456]
                audio_data = wav_array[_slot * wav_frame_num:(_slot + 1) * wav_frame_num]
                runner.video_queue.put((frame_speech, audio_data, chunk))  # [pyd py=457]
            else:
                out.write(cv2.resize(frame_speech, (runner.args.frame_w,
                                                    runner.args.frame_h)))  # [pyd py=446/447]
            _slot += 1
        pbar.update(1)
    pbar.close()


def parse_args():  # [pyd FUN_18001cae0 py=460;varnames (parser,args,unknow)]
    parser = argparse.ArgumentParser(  # [pyd py=461]
        description="Inference code to lip-sync videos in the wild using Wav2Lip models")
    parser.add_argument(  # [pyd py=465]
        "--version", type=str, help="checkpoint version", default="local")
    parser.add_argument(  # [pyd py=472]
        "--checkpoint_path", type=str, help="Name of saved checkpoint to load weights from",
        default="modules/w2l/model_fp16.onnx")
    parser.add_argument(  # [pyd py=479]
        "--face", type=str, help="Filepath of video/image that contains faces to use",
        default="test/1.mp4")
    parser.add_argument(  # [pyd py=486]
        "--pads", nargs="+", type=int, default=[0, 15, 0, 0],
        help="Padding (top, bottom, left, right). Please adjust to include chin at least")
    parser.add_argument(  # [pyd py=494]
        "--face_det_batch_size", type=int, help="Batch size for face detection", default=2)
    parser.add_argument(  # [pyd py=500]
        "--wav2lip_batch_size", type=int, help="Batch size for Wav2Lip model(s)", default=1)
    parser.add_argument(  # [pyd py=507]
        "--resize_factor", default=1, type=int,
        help="Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p")
    parser.add_argument(  # [pyd py=514]
        "--crop", nargs="+", type=int, default=[0, -1, 0, -1],
        help="Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width")
    parser.add_argument(  # [pyd py=523]
        "--box", nargs="+", type=int, default=[-1, -1, -1, -1],
        help="Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).")
    parser.add_argument(  # [pyd py=532]
        "--rotate", default=False, action="store_true",
        help="Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.Use if you get a flipped result, despite feeding a normal looking video")
    parser.add_argument(  # [pyd py=540]
        "--nosmooth", default=False, action="store_true",
        help="Prevent smoothing face detections over a short temporal window")
    args, unknow = parser.parse_known_args()  # [pyd py=547]
    args.img_size = 256  # [pyd py=548]
    return args  # [pyd py=549]
