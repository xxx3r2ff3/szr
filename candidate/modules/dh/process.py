# -*- coding: utf-8 -*-
"""modules.dh.process —— 语义重建候选(R014 / T4R)。

源 pyd : modules/dh/process.cp310-win_amd64.pyd
sha256 : 4a3f0e46e4eb9ac2344ec5f78d381468cf96a82a8df464e6abe7731ff9a0f8de

证据来源(行内注释):
  [FUN_xxxxxxxx] Ghidra 反编译 evidence/modules/dh__process/static/pseudocode/
  [py=N]         Cython co_firstlineno / 反编译行号标记
  [E1 xxx]       VM 内 oracle 实测(ns_w2dh 共享目录 *_out.json)
"""
import argparse
import gc
import os
import platform
import subprocess

import cv2
import librosa
import numpy as np
import onnxruntime
import torch
from librosa import istft, stft
from scipy.io.wavfile import write
from tqdm import tqdm
from ultralytics import YOLO

# [FUN_180016c50] 模块级:now_dir=dirname(abspath(__file__));root_dir=abspath(join(now_dir,'../../'))
now_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(now_dir, '../../'))
# [FUN_180016c50] device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ResembleDenoiser(object):
    """[py=21][FUN_1800012d0] denoiser.onnx 的 ONNX 会话包装。"""

    def __init__(self, model_path='denoiser.onnx', device='cpu'):
        # [E1 probe_learn2] 默认相对路径 'denoiser.onnx'(cwd 相对),沙箱内 NoSuchFile
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = \
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        # [I007 E2E 定谳 i007-fix-8] 会话选项按 FUN_1800012d0 反编译修正:
        # inter_op/intra_op_num_threads = **4**(DAT_1800287b8)、log_severity_level =
        # **4**(同常量);R014 期旧重建的 1/1/3 属未验证假设——线程数改变并行
        # 归约顺序,模型输出 ULP 随之漂移,nr 产物 int16 翻位(nr_wav sha 不齐的
        # 最后根因)。providers:cuda → CUDA+cudnn DEFAULT,否则 CPU。
        session_options.inter_op_num_threads = 4
        session_options.intra_op_num_threads = 4
        session_options.log_severity_level = 4
        if device == 'cuda':
            providers = [('CUDAExecutionProvider',
                          {'cudnn_conv_algo_search': 'DEFAULT'})]
        else:
            providers = ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(
            model_path, sess_options=session_options, providers=providers)
        # [I007 E2E 定谳 i007-fix-8] 实例属性三件套(i007_mattrs 探针:vars(dn) 含
        # n_fft=1680 / stft_hop_length=420 / win_length=1680),_istft 读取。
        self.n_fft = 1680
        self.stft_hop_length = 420
        self.win_length = 1680

    def _stft(self, x):                                    # [py=44][FUN_180001ff0]
        # [I007 E2E 定谳 i007-fix-8] 终轮字节级定谳(i007_stftsweep/cossin 探针):
        #   s = librosa.stft(x, n_fft=1680, hop_length=420, win_length=1680,
        #                    window='hann', center=True, pad_mode='reflect')
        #   按 L//420 截帧后 mag=np.abs(s) 逐字节命中(bf607770…);cos/sin =
        #   np.cos/np.sin(np.angle(s))(除法形式 cos=s.real/mag 被证伪)。
        #   上一轮 fix-5 的 torch.stft 形与本形差 1~2 float32 ULP(nr_wav sha
        #   随之偏差),librosa 形与 oracle 完全一致。真 _model_infer 输入维度
        #   mag=[*,841,*](841=1680/2+1);帧数=floor(L/hop)。
        s = stft(x, n_fft=1680, hop_length=420, win_length=1680, window='hann',
                 center=True, pad_mode='reflect')
        s = s[..., :x.shape[-1] // 420]
        mag = np.abs(s)
        phi = np.angle(s)
        return (mag.astype(np.float32), np.cos(phi).astype(np.float32),
                np.sin(phi).astype(np.float32))

    def _istft(self, mag, cos, sin):                       # [py=59][FUN_180003070]
        # [I007 E2E 定谳 i007-fix-8] 反编译 FUN_180003070:real=mag*cos;
        # imag=mag*sin;np.pad(s, ((0,0),(0,0),(0,1)), mode='edge');istft(
        # window='hann', win_length=self.win_length, hop_length=self.stft_hop_length,
        # n_fft=self.n_fft)。批维度 (1,841,F) 贯穿(2-D 输入会撞 pad 广播错,
        # i007_istft 探针实证)。
        real = mag * cos
        imag = mag * sin
        s = real + 1j * imag
        s = np.pad(s, ((0, 0), (0, 0), (0, 1)), mode='edge')
        return istft(s, window='hann', win_length=self.win_length,
                     hop_length=self.stft_hop_length, n_fft=self.n_fft)

    def _model_infer(self, wav):                           # [py=73][FUN_180003a20]
        # [I007 E2E 定谳 i007-fix-8] wav 为 **2-D (1,L)**(1-D 实参触发 np.pad
        # (2,2)-spec 广播 ValueError,i007_mattrs 实证);pad 宽 = 模块常量
        # 441(=0x1b9,DAT_180028ab0=((0,0),(0,441)));np.pad **无 mode 关键字**
        # (FUN_180003a20 无 mode 串槽 → 默认 constant 补零);stft 后 (1,841,F)
        # 直接入模(无 [None] 扩维),sep 三元组原样进 _istft;
        # i007_mattrs:dn.n_fft=1680/dn.stft_hop_length=420/dn.win_length=1680。
        padded_wav = np.pad(wav, ((0, 0), (0, 441)))
        mag, cos, sin = self._stft(padded_wav)
        sep_mag, sep_cos, sep_sin = self.session.run(None, {
            'mag': mag.astype(np.float32), 'cos': cos.astype(np.float32),
            'sin': sin.astype(np.float32)})
        out = self._istft(sep_mag, sep_cos, sep_sin)
        return out

    def denoise(self, wav, sample_rate, batch_process_chunks=False):   # [py=86][FUN_180004ad0]
        chunk_length = int(sample_rate * 10)
        hop_length = int(sample_rate * 4)
        # [I007 E2E 定谳 i007-fix-6] oracle denoise **恒返回二元组 (降噪数组, 44100)**
        # (i007_adjud2/3 探针:batch=False 亦为 tuple,elem0 len=输入长、elem1=int 44100;
        #  R019 期旧重建的两分支单返回值被证伪)。降噪数组长度截回原输入。
        wav_len = len(wav)
        if not batch_process_chunks:
            return self._model_infer(wav[None, :])[0, :wav_len], 44100
        num_chunks = max(1, int(np.ceil((len(wav) - chunk_length) / hop_length)) + 1)
        n_pad = chunk_length + (num_chunks - 1) * hop_length - len(wav)
        if n_pad > 0:
            wav = np.pad(wav, (0, n_pad), mode='reflect')
        chunks = []
        for c in range(num_chunks):
            chunks.append(wav[c * hop_length:c * hop_length + chunk_length])
        abs_max = np.max(np.abs(wav))
        res_chunks = [self._model_infer(chunk[None, :])[0, :len(chunk)]
                      for chunk in chunks]
        res = np.zeros(len(wav), dtype=np.float32)
        for c, chunk in enumerate(res_chunks):
            res[c * hop_length:c * hop_length + chunk_length] += chunk[:len(res[c * hop_length:c * hop_length + chunk_length])]
        return (res / max(abs_max, 1e-8) * abs_max)[:wav_len], 44100


class laplacianSmooth(object):
    """[py=162][FUN_18000aac0] 人脸框平滑。"""

    def __init__(self, smoothAlpha=0.3):
        self.smoothAlpha = smoothAlpha
        self.pts_last = None

    def smooth(self, pts_cur):                             # [py=166][FUN_18000af60]
        # [E1 probe_learn2] pts_last 为 None 时原样返回 pts_cur 并记录
        if self.pts_last is None:
            self.pts_last = np.array(pts_cur, dtype=np.float32)
            return pts_cur
        pts_update = []
        for i in range(len(pts_cur)):
            x_new, y_new = pts_cur[i]
            x_old, y_old = self.pts_last[i]
            tmp = np.exp(-1 / self.smoothAlpha * (np.array([x_new - x_old,
                                                            y_new - y_old]) ** 2))
            w = tmp / tmp.sum()
            x = w[0] * x_new + w[1] * x_old
            y = w[0] * y_new + w[1] * y_old
            pts_update.append([x, y])
        self.pts_last = np.array(pts_update, dtype=np.float32)
        return np.array(pts_update)


# [FUN_180016c50] 模块级:face_det = YOLO(join(root_dir,'checkpoints/yolov8-face.pt'))
#                 (ST 0x180028408 'checkpoints/yolov8-face.pt' + 同处 root_dir 引用;
#                  E1:oracle 侧该权重文件存在,6 247 065 字节)
# 环境门控登记:候选树不携带 6MB 级权重文件(不把权重抄进候选、禁止真联网下载),
# 因此导入期 YOLO 构造在"文件缺失"时退化为同类型空实例 —— 仅保留类型与名字,
# 不改变任何被测函数在无权重环境下的可观测行为(get_landmark/face_detect 均需权重)。
try:
    face_det = YOLO(os.path.join(root_dir, 'checkpoints/yolov8-face.pt'))
except FileNotFoundError:
    # 候选树不携带 6 247 065 字节的 yolov8-face.pt(不抄权重、禁网下载);
    # 用 ultralytics 内置 yaml 建同类型实例,保证"缺 source 文件即 FileNotFoundError"
    # 等与权重无关的行为与冻结版一致(权重相关推理不在候选环境内可达)。
    face_det = YOLO('yolo11n.yaml')
abox_smoother = laplacianSmooth()


def run_command(command):                                  # [py=111][FUN_1800069b0]
    # [FUN_1800069b0] PyObject_RichCompare(platform.system(),'Windows',3) ⇒ '!='
    # [E1 probe_learn] Windows 下 shell=False:run_command('echo x') → FileNotFoundError
    subprocess.check_call(command, shell=(platform.system() != 'Windows'))


def extract_audio(path, out_path, sample_rate=16000):      # [py=115][FUN_180007110]
    # [E1 probe_learn3] stdout 两行;命令 'ffmpeg -y -i %s -f wav -ar %s %s'
    print('[INFO] ===== extract audio from ' + str(path) + ' to ' + str(out_path) + ' =====')
    cmd = 'ffmpeg -y -i ' + str(path) + ' -f wav -ar ' + str(sample_rate) + ' ' + str(out_path)
    os.system(cmd)
    print('[INFO] ===== extracted audio =====')


def extract_images(path, mode):                            # [py=123][FUN_180007b90]
    # [E1 probe_learn3] fps 校验:hubert 必须 25、wenet 必须 20;帧写 frames/<n>.jpg
    full_body_dir = os.path.join(os.path.dirname(path), 'frames')
    if not os.path.exists(full_body_dir):
        os.mkdir(full_body_dir)
    counter = 0
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if mode == 'hubert':
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps != 25:
            raise ValueError('your video fps should be 25!!!')
    elif mode == 'wenet':
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps != 20:
            raise ValueError('your video fps should be 20!!!')
    else:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(total), desc='extract images'):
        ret, frame = cap.read()
        cv2.imwrite(os.path.join(full_body_dir, str(counter) + '.jpg'), frame)
        counter += 1
    cap.release()


def get_audio_feature(wav_path, mode):                     # [py=147][FUN_18000a060]
    # [E1 probe_learn3] hubert→'python <root_dir>/modules/facebook/hubert.py --wav <wav>'
    #                   wenet →'python wenet_infer.py <wav>';其它 mode 不做任何事
    # [I007 E2E 定谳 i007-fix-10] print **无条件**置于 mode 分派之前(终轮 E2E
    #   stdout:3 次 get_audio_feature 调用 → oracle 3 行 'extracting audio
    #   feature...',candidate 仅 1 行;反编译 DAT_180028a78 单缓存打印元组同证)。
    print('extracting audio feature...')
    if mode == 'hubert':
        run_command('python ' + os.path.join(root_dir, 'modules/facebook/hubert.py')
                    + ' --wav ' + str(wav_path))
    elif mode == 'wenet':
        run_command('python wenet_infer.py ' + str(wav_path))


def face_detect(images):                                   # [py=195][FUN_18000c760]
    # [E1 probe_learn3] 空图 → np.array 形状 (N,4) 全 0
    predictions = []
    for frame in images:
        # [I007 E2E 定谳 i007-fix-12c] conf=**0.8**(i007_yargs spy 实证 oracle
        #   face_det 调用实参:imgsz=640/conf=0.8/iou=0.5/half=True/augment=False/
        #   verbose=False/device='cpu';R014 期旧重建的 conf=0.5 属误读,
        #   0.5→0.8 差一级 NMS 幸存框,25 帧地标 y 全线 ±1~2px 漂移即此)。
        boxes = face_det(frame, imgsz=640, conf=0.8, iou=0.5, half=True,
                         augment=False, verbose=False, device=device)
        results = boxes[0].boxes.xyxy.cpu().numpy()
        if len(results) == 0:
            rect = [0, 0, 0, 0]
        else:
            rect = np.reshape(results[0], (4,))  # [I007 E2E 定谳 i007-fix-12b] 保持 float(不 astype(int32)),get_landmark 以浮点框计算(y 中心/方形化),int 截断发生在 get_landmark 终值;R014 期 astype 属未验证假设,终轮 E2E 25 帧地标 ±1 漂移被证伪。
        predictions.append(rect)
    predictions = np.array(predictions)
    return predictions


def get_landmark(path):                                    # [py=234][FUN_18000f860]
    # [E1 probe_learn2] 目录不存在 → 0 帧,返回 None(仅 tqdm 空条)
    full_body_dir = os.path.join(os.path.dirname(path), 'frames')
    images = []
    batch_size = 1
    faceboxs = []
    i = 0
    frame_paths = sorted([os.path.join(full_body_dir, name)
                          for name in os.listdir(full_body_dir)
                          if name.endswith('.jpg')])
    for img_name in tqdm(frame_paths, desc='detect landmarks'):
        img_path = img_name
        frame = cv2.imread(img_path)
        face_detect_results = face_detect([frame])
        facebox = face_detect_results[0]
        # [I007 E2E 定谳 i007-fix-12] y 边界 = **以 y 中心为起点、按 x 宽度取方形**:
        #   ymin = int((y1+y2)/2)、ymax = int(ymin + (x2-x1))(终轮 E2E 25 帧真值:
        #   oracle 行0 y=56/114、行14 y=56/113、行15 y=55/112,x2-x1 浮点尾数决定
        #   int 边界翻位;R014 期旧重建的 max/min 钳位形被证伪)。
        #   反编译对应 FUN_18000f860 末段 (A-B)→int→(+B)+C-D 算术链
        #   (A=y1+y2、B=2、C=x2-x1)。
        x1, y1, x2, y2 = int(facebox[0]), int(facebox[1]), int(facebox[2]), int(facebox[3])
        xmin = max(0, x1)
        xmax = min(frame.shape[1], x2)
        y_center = (y1 + y2) / 2
        ymin = int(y_center)
        ymax = int(ymin + (x2 - x1))
        faceboxs.append([xmin, xmax, ymin, ymax])
        images.append(frame)
        i += 1
    landmarks = np.array(faceboxs)
    np.save(os.path.join(os.path.dirname(path), 'landmarks.npy'), landmarks)


def noise_reduction(wav_path):                             # [py=263][FUN_180011f80]
    # [I007 E2E 定谳 i007-fix-6] 终轮字节级定谳(i007_adjud2/3 探针):
    #   oracle = librosa.load(wav_path, sr=44100, mono=True) → denoise(wav, sr,
    #   batch_process_chunks=False) → **原地覆写 wav_path**(write(wav_path, sr, ...
    #   int16),scipy 极简头/44100Hz;Variant A 与 oracle 噪声 Reduction 输出
    #   sha256 逐字节一致 252e9e1d…;上一轮 fix-5 的 load(sr=None)+True+resample
    #   形(V-B)与 batch=False 单返回值形均被字节级证伪)。
    #   反编译出处:FUN_180011f80 kwargs dict {'sr': DAT_180028e50(=44100), 'mono':
    #   True}、denoise kwargs {'batch_process_chunks': False}、write 实参 = param_2。
    denoiser = ResembleDenoiser(model_path='denoiser.onnx', device=device)
    wav, sr = librosa.load(wav_path, sr=44100, mono=True)
    wav_denoised, _ = denoiser.denoise(wav, sr, batch_process_chunks=False)
    write(wav_path, sr, (wav_denoised * 32767).astype(np.int16))
    del denoiser
    gc.collect()


def main(video_file, asr_mode='hubert', bitrate='3000k'):   # [py=276][FUN_1800134e0]
    # [E1 probe_learn2] 缺 ffmpeg → FileNotFoundError (WinError 2)
    base_dir = os.path.dirname(video_file)
    wav_path = os.path.join(base_dir, 'aud.wav')
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    live_path = os.path.join(base_dir, 'live.mp4')
    if asr_mode == 'hubert':
        run_command('ffmpeg -y -i ' + str(video_file)
                    + ' -vf scale=1080:1920 -r 25 -b:v ' + str(bitrate) + ' ' + live_path)
    elif asr_mode == 'wenet':
        run_command('ffmpeg -y -i ' + str(video_file)
                    + ' -vf scale=1080:1920 -r 20 -b:v ' + str(bitrate) + ' ' + live_path)
    extract_audio(video_file, wav_path)
    extract_images(video_file, asr_mode)
    noise_reduction(wav_path)
    get_audio_feature(wav_path, asr_mode)
    get_landmark(video_file)
    # [I007 E2E 定谳 i007-fix-11] oracle main() 末尾**无**第二行
    #   'extracted audio' 打印(终轮 E2E stdout:candidate 多一行,oracle 无;
    #   该串常量 DAT_180028ee0 属 extract_audio 自身第二次打印,勿在 main 重复)。
