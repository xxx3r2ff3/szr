# -*- coding: utf-8 -*-
# 出处:S003 恢复件(candidate/modules/heygem/wenet/compute_ctc_att_bnf.pyc,py3.8)。
# 反汇编:evidence/modules/_heygem_orphans/disasm/wenet_compute_ctc_att_bnf.pyc.dis
# 字节码全局名对照:AssertionError <- PPGModel._extract_feats 的 assert 语句
# 隐式加载(py3.8 编译 assert 时 LOAD_GLOBAL AssertionError)。
# 保真说明:compute_bnf/get_weget 的 section 形参在本 pyc 字节码中未被使用
# (分片切片用字面量 560000),按原样保留形参;其余函数体均按 [Disassembly] 逐条还原。
"""
Compute CTC-Attention Seq2seq ASR encoder bottle-neck features (BNF).
"""
import os
import time
import argparse
import torch
from pathlib import Path
import yaml
import numpy as np
from wenet.tools._extract_feats import wav2mfcc_v2, load_wav, _extract_feature

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
hparams1 = {
    'sample_rate': 16000,
    'preemphasis': 0.97,
    'n_fft': 1024,
    'hop_length': 160,
    'win_length': 800,
    'num_mels': 80,
    'n_mfcc': 13,
    'window': 'hann',
    'fmin': 0,
    'fmax': 8000,
    'ref_db': 20,
    'min_db': -80,
    'iterations': 100,
    'silence_db': -28,
    'center': True
}
SAMPLE_RATE = 16000

from wenet.transformer.encoder import ConformerEncoder

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def plot_spectrogram(input, path=None, info=None):
    spectrogram = input
    fig = plt.figure(figsize=(16, 10))
    plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    plt.colorbar()
    if info is not None:
        plt.xlabel(info)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return fig


class PPGModel(torch.nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, feats, feats_lengths):
        """
        Args:
            speech (tensor): (B, L)
            speech_lengths (tensor): (B, )

        Returns:
            bottle_neck_feats (tensor): (B, L//hop_size, 144)

        """
        encoder_out, encoder_out_lens = self.encoder(feats, feats_lengths)
        return encoder_out

    def _extract_feats(self, speech: torch.Tensor,
                       speech_lengths: torch.Tensor):
        assert speech_lengths.dim() == 1, speech_lengths.shape
        # for data augmentation
        speech = speech[:, :speech_lengths.max()]
        if self.frontend is not None:
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths


def build_model(args):
    encoder = ConformerEncoder(**{
        'input_size': 80,
        **args.encoder_conf,
    })
    model = PPGModel(encoder)
    return model


def load_ppg_model(train_config, model_file, device):
    config_file = Path(train_config)
    with config_file.open('r', encoding='utf-8') as f:
        args = yaml.safe_load(f)
    args = argparse.Namespace(**args)

    model = build_model(args)
    model_state_dict = model.state_dict()
    ckpt_state_dict = torch.load(model_file, map_location='cpu')
    ckpt_state_dict = {
        k: v
        for k, v in ckpt_state_dict.items()
        if 'encoder' in k and 'encoder.global_cmvn' not in k
    }
    model_state_dict.update(ckpt_state_dict)
    model.load_state_dict(model_state_dict)
    return model.eval().to(device)


def compute_bnf(wav_dir, wenet_model, section=560000):
    device = 'cuda'
    ppg_model_local = wenet_model
    wav_arr = load_wav(wav_dir, sr=hparams1['sample_rate'])
    zero = np.zeros(6400)
    wav_arr = np.concatenate((zero, wav_arr, zero))
    result = []
    add_silence_flag = False
    for i in range(len(wav_arr) // 560000 + 1):
        wav_arr_ = wav_arr[560000 * i:560000 * (i + 1)]
        if len(wav_arr_) < hparams1['sample_rate']:
            wav_arr_ = np.append(wav_arr_, np.zeros(hparams1['sample_rate']))
            add_silence_flag = True
        mel, x_stft = wav2mfcc_v2(wav_arr_,
                                  sr=hparams1['sample_rate'],
                                  n_mfcc=hparams1['n_mfcc'],
                                  n_fft=hparams1['n_fft'],
                                  hop_len=hparams1['hop_length'],
                                  win_len=hparams1['win_length'],
                                  window=hparams1['window'],
                                  num_mels=hparams1['num_mels'],
                                  center=hparams1['center'])
        wav_tensor = torch.from_numpy(mel).float().to(device).unsqueeze(0)
        wav_length = torch.LongTensor([mel.shape[0]]).to(device)
        start_time = time.time()
        with torch.no_grad():
            bnf = ppg_model_local(wav_tensor, wav_length)
        print('use_time:', time.time() - start_time, bnf.shape)
        if add_silence_flag:
            bnf_npy = bnf[0].squeeze(0).cpu().numpy()[:-25]
        else:
            bnf_npy = bnf[0].squeeze(0).cpu().numpy()
        result.append(bnf_npy)
        add_silence_flag = False
    bnf_npy = np.concatenate(result, 0)
    return bnf_npy


def get_parser():
    parser = argparse.ArgumentParser(
        description='compute ppg or ctc-bnf or ctc-att-bnf')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/data1/wangpeiyu/out/dg_interface_2_2/wrong/wenet')
    parser.add_argument(
        '--wav_dir',
        type=str,
        default='/data1/wangpeiyu/out/dg_interface_2_2/wrong/add_sil')
    parser.add_argument(
        '--train_config',
        type=str,
        default='examples/aishell/aidata/conf/train_conformer_multi_cn.yaml')
    parser.add_argument(
        '--model_file',
        type=str,
        default='examples/aishell/aidata/exp_3500/conformer/81.pt')
    return parser


def get_weget(wavpath, wenet_model, section=560000):
    return compute_bnf(wavpath, wenet_model, section)


def get_weget0(wavpath):
    return compute_bnf(
        wavpath,
        '/data1/wangpeiyu/APB2Face/wenet/examples/aishell/aidata/conf/train_conformer_multi_cn.yaml',
        '/data1/wangpeiyu/APB2Face/wenet/examples/aishell/aidata/exp/conformer/wenetmodel.pt'
    )


if __name__ == '__main__':
    result = get_weget(
        '/data1/wangpeiyu/out/dg_interface_2_2/wrong/add_sil/安.wav')
    print(result.shape)
