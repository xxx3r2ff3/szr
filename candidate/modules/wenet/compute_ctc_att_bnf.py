# -*- coding: utf-8 -*-
# R055 候选重建:modules/wenet/compute_ctc_att_bnf.py

# 源件(只读 oracle):modules/wenet/compute_ctc_att_bnf.cp310-win_amd64.pyd
#   sha256 0278c1442354e43781abc7093ffb23bc41c9beac1652642c20472a36c1fafef5

# 证据链(全部可在 evidence/modules/wenet__compute_ctc_att_bnf/ 复核):
#   * E1 探针 probe_ppg_1..9(oracle 侧实测,输出在 runtime/probe_ppg_*_out.json):
#     - 命名空间/签名/默认值/docstring;函数 code 对象(co_varnames/co_firstlineno);
#     - np/torch 代理追踪得到的逐语句调用序列与实参;
#     - 逐点制造失败得到 Cython 回溯里的源码行号(build_model:80、
#       load_ppg_model:97/99/101/105/109/126、compute_bnf:180 等);
#     - 反汇编行标记(r2,pyd 内 mov ebx,<line>)给出 load_ppg_model 有代码的行集合。
#   * 谱系对照(只读):/Volumes/A/数字人/szr2026/modules/heygem/wenet/
#     compute_ctc_att_bnf.pyc(py3.8,S003 恢复源,函数体残缺)与
#     candidate/modules/heygem/wenet/compute_ctc_att_bnf.py(仅结构参考,未沿用其残缺体)。
#   * 本 py3.10 版相对 py3.8 的本地改动(均有二进制证据):
#     - 删 plot_spectrogram / matplotlib 导入(pyd dir() 无 matplotlib/plt/plot_spectrogram);
#     - 删 get_parser / get_weget0(pyd dir() 无此二名);
#     - `from wenet.tools._extract_feats import wav2mfcc_v2, load_wav`(无 _extract_feature);
#     - hparams1['hop_length'] = 160(probe 实测 repr);
#     - 新增 now_dir = os.getcwd() 与 import sys(pyd dir() 有 now_dir/sys)。

# 未唯一还原项(见报告"未定谳"):load_ppg_model 源码第 112-121 行有代码
# (反汇编行标记 112/115/118/119/120/121),但在 12 个行为场景下与下面的实现观测等价;
# compute_bnf 源码体内另有约 34 行无代码内容(注释/空行,probe 行锚 compute_bnf=131 /
# get_weget=206 与本文件 firstlineno 存在偏差,已登记)。
import os
import time
import argparse
import torch
from pathlib import Path
import yaml
import numpy as np
from wenet.tools._extract_feats import wav2mfcc_v2, load_wav
import sys

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'          # [pyd strtab] environ/CUDA_DEVICE_ORDER/PCI_BUS_ID
hparams1 = {                                            # [E1 probe_ppg_1] hparams1 repr(hop_length=160)
    'sample_rate': 16000,
    'preemphasis': 0.97,
    'n_fft': 1024,
    'hop_length': 160,
    'win_length': 800,
    'num_mels': 80,
    'n_mfcc': 13,
    'window': 'hann',
    'fmin': 0.,
    'fmax': 8000.,
    'ref_db': 20,
    'min_db': -80.0,
    'iterations': 100,
    'silence_db': -28.0,
    'center': True,
}
SAMPLE_RATE = 16000                                     # [E1 probe_ppg_1] SAMPLE_RATE=16000
from wenet.transformer.encoder import ConformerEncoder  # [pyd strtab] wenet.transformer.encoder/ConformerEncoder
now_dir = os.getcwd()                                   # [E1 probe_ppg_1] now_dir == 进程 cwd

# 说明:py3.8 版此处还 import 了 matplotlib 并定义 plot_spectrogram(py3.8 行 34-48),
# py3.10 pyd 已删除该函数与 matplotlib:pyd dir() 无 matplotlib/plt/plot_spectrogram,
# 字符串表亦无 'matplotlib.pyplot'(对照 evidence/.../static/imports.txt)。
# 行号锚点(pyd co_firstlineno 实测):__init__=45 / forward=50 / _extract_feats=62 /
# build_model=78 / load_ppg_model=85 / compute_bnf=131 / get_weget=206;
# 本文件因证据注释与源码注释不可还原,不主张行号一致(已登记偏差)。


class PPGModel(torch.nn.Module):
    """Phoneme Posterior Gram (PPG) 模型"""     # [E1 probe_ppg_2] PPGModel.__doc__

    def __init__(self, encoder):
        super().__init__()                      # [E1 probe_ppg_6 G_init] 实例属性 encoder/frontend
        self.encoder = encoder
        self.frontend = None

    def forward(self, feats, feats_lengths):
        """
        Args:
            feats (tensor): (B, L, D)
            feats_lengths (tensor): (B,)

        Returns:
            bottle_neck_feats (tensor): (B, L//hop_size, D')
        """
        encoder_out, encoder_out_lens = self.encoder(feats, feats_lengths)   # [E1 probe_ppg_2 A_forward] 解包 2 元组
        return encoder_out                                                   # [E1 probe_ppg_6] forward 返回 ENC_OUT

    def _extract_feats(self, speech, speech_lengths):
        """提取特征"""
        assert speech_lengths.dim() == 1                # [E1 probe_ppg_2] 2 维 lens → AssertionError(行 64)
        max_length = speech_lengths.max()
        speech = speech[:, :max_length]
        if self.frontend is not None:                   # [E1 probe_ppg_2 A_extract_frontend] frontend 收到 (speech, lengths)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            feats = speech
            feats_lengths = speech_lengths
        return feats, feats_lengths


def build_model(args):
    """根据参数构建PPG模型"""
    encoder = ConformerEncoder(**args.encoder_conf, input_size=args.input_size)   # [E1 probe_ppg_ab] dict 入参先报 encoder_conf
    model = PPGModel(encoder)
    return model                                    # [E1 probe_ppg_9] 返回模型 training=True(未 eval)


def load_ppg_model(train_config, model_file, device):
    """加载预训练的PPG模型

    Args:
        train_config: 训练配置文件路径
        model_file: 模型权重文件路径
        device: 设备 ('cuda' 或 'cpu')

    Returns:
        加载好的模型
    """
    config_file = Path(train_config)                          # [pyd line 96]
    with config_file.open(encoding='utf-8') as f:              # [pyd line 97:pathlib open 帧]
        args = yaml.safe_load(f)                               # [pyd strtab] safe_load/utf-8
    args = argparse.Namespace(**args)                          # [pyd line 99:** 非 mapping 报错]

    model = build_model(args)                                  # [pyd line 101]
    model_state_dict = model.state_dict()                      # [pyd line 102]

    ckpt_state_dict = torch.load(model_file, map_location=device, weights_only=True)   # [pyd line 105;E1 torch 代理实测 kwargs]
    filtered_state_dict = {}                                   # [pyd line 108]
    for k, v in ckpt_state_dict.items():                       # [pyd line 109:list → 'no attribute items']
        if k in model_state_dict:
            filtered_state_dict[k] = v                         # [pyd line 111]
    # 注:pyd 源码此处(112-121 行)另有代码(反汇编行标记 112/115/118/119/120/121),
    # 但在下列 12 个场景下与本实现观测等价,未唯一还原(详见报告 §未定谳):
    #   全键/半键/无关键/空 ckpt/嵌套 state_dict|model_state_dict|model/
    #   'module.' 前缀|'.module' 尾缀/额外键/命中键取值非张量/global_cmvn 取 str|list|None。
    missing_keys = model.load_state_dict(filtered_state_dict, strict=False)   # [pyd line 126]
    model.eval()                                               # [E1 probe_ppg_7] eval→train(False)
    model.to(device)                                           # [E1 probe_ppg_7] eval 之后 to(device)
    return model


def compute_bnf(wav_dir, wenet_model, section=560000):
    """计算音频文件的瓶颈特征

    Args:
        wav_dir: 音频文件路径
        wenet_model: 预加载的模型
        section: 处理的音频段大小

    Returns:
        瓶颈特征数组
    """
    device = 'cuda'                                     # [E1 probe_ppg_2 D] 无 GPU 时 .to('cuda') 报 NVIDIA driver
    ppg_model_local = wenet_model
    wav_arr = load_wav(wav_dir, sr=hparams1['sample_rate'])          # [E1 probe_ppg_2] kwargs sr=16000
    zero = np.zeros(6400)                                            # [E1 probe_ppg_4] zeros(6400)
    wav_arr = np.concatenate((zero, wav_arr, zero))                  # [E1 probe_ppg_4] 前后各补 6400
    result = []
    add_silence_flag = False
    # 循环次数 = len(wav_arr)//section + 1(probe_ppg_4 扫参实测:切片数 = 商+1)
    for i in range(len(wav_arr) // section + 1):
        start_idx = i * section
        end_idx = start_idx + section
        wav_arr_ = wav_arr[start_idx:end_idx]                        # 末片可短/可空
        if len(wav_arr_) < 16000:                                    # [E1 probe_ppg_5] 边界实测:<16000 才补
            add_silence_flag = True
            wav_arr_ = np.append(wav_arr_, np.zeros(16000))          # 补零到 len+16000(非补到 16000)
        mel, x_stft = wav2mfcc_v2(wav_arr_, sr=hparams1['sample_rate'],
                                  n_mfcc=hparams1['n_mfcc'], n_fft=hparams1['n_fft'],
                                  hop_len=hparams1['hop_length'], win_len=hparams1['win_length'],
                                  window=hparams1['window'], num_mels=hparams1['num_mels'],
                                  center=hparams1['center'])         # [E1 probe_ppg_4] 8 个 kwargs 实测
        wav_tensor = torch.from_numpy(mel).float().to(device).unsqueeze(0)   # [E1 张量方法链] float→to→unsqueeze
        wav_length = torch.LongTensor([mel.shape[0]]).to(device)             # [E1] LongTensor([mel_len])
        start_time = time.time()
        with torch.no_grad():                                                # [E1] 每片一次 no_grad
            bnf = ppg_model_local(wav_tensor, wav_length)
        bnf = bnf.squeeze(0).cpu().numpy()                                   # [E1 张量方法链] squeeze(0)→cpu→numpy
        result.append(bnf)
    bnf_npy = np.concatenate(result, 0)                                      # [E1] 第二实参 0
    return bnf_npy


def get_weget(wavpath, wenet_model, section=560000):
    """获取音频的瓶颈特征(BNF)的包装函数

    Args:
        wavpath: 音频文件路径
        wenet_model: 预加载的模型
        section: 处理的音频段大小

    Returns:
        瓶颈特征数组
    """
    return compute_bnf(wavpath, wenet_model, section)   # [E1 probe_ppg_2 E] 位置传 3 参并原样返回
