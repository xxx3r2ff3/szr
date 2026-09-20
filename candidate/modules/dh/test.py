# -*- coding: utf-8 -*-
"""modules.dh.test —— 语义重建候选(R015 / T4R)。

源 pyd : modules/dh/test.cp310-win_amd64.pyd
sha256 : 6841f85ff7f6f257b144d55d48ab32ecd7f3c5acb899d7f4bfccbb9f7e2caf88
证据   : evidence/modules/dh__test/static/pseudocode/[FUN_xxxxxxxx];E1 见行内注释。
"""
import argparse
import os
import platform
import subprocess

import cv2
import numpy as np
import onnxruntime
import torch
from tqdm import tqdm

from modules.dh.network import Model

# [FUN_18000e720] now_dir=dirname(abspath(__file__));root_dir=abspath(join(now_dir,'../../'))
now_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(now_dir, '../../'))


class DH(object):
    """[py=21][FUN_1800013d0] onnx 会话;inference/activate 见 [py=42]/[py=52]。"""

    def __init__(self, model_path, device='cuda', gpu_id=0):
        # [E1 probe_learn2] 缺 onnx 文件 → onnxruntime NoSuchFile(与 providers 配置无关)
        session_options = onnxruntime.SessionOptions()
        providers = ['CPUExecutionProvider']
        if device == 'cuda':
            providers = [('CUDAExecutionProvider',
                          {'cudnn_conv_algo_search': 'DEFAULT', 'device_id': gpu_id})]
        self.session = onnxruntime.InferenceSession(
            model_path, sess_options=session_options, providers=providers)

    def inference(self, img, audio):                       # [py=42][FUN_180001dc0]
        output = self.session.run(None, {'input': img, 'audio': audio})
        return output

    def activate(self):                                    # [py=52][FUN_1800020a0]
        img = torch.zeros(1, 6, 256, 256)
        audio = torch.zeros(1, 32, 56, 56)
        output = self.inference(img.cpu().numpy(), audio.cpu().numpy())
        return output


def run_command(command):                                  # [py=58][FUN_180002ac0]
    # [E1 probe_learn] Windows 下 shell=False('echo x' → FileNotFoundError)
    subprocess.check_call(command, shell=(platform.system() != 'Windows'))


def make_hubert(wav_file):                                 # [py=62][FUN_180002fa0]
    # [E1 probe_learn + diff_smoke3] 先拼命令(纯 str 拼接,非 str 入参先 TypeError),
    # 命令里用的是 wav_file 本身;随后才 .replace 得到返回值。
    run_command('python ' + os.path.join(root_dir, 'modules/facebook/hubert.py')
                + ' --wav ' + wav_file)
    hu_file = wav_file.replace('.wav', '_hu.npy')
    return hu_file


def get_args():                                            # [py=71][FUN_180003870]
    # [E1 probe_learn] Namespace 顺序 asr/dataset/audio/output/checkpoint;--checkpoint 必填
    parser = argparse.ArgumentParser(
        description='Test model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--asr', type=str, default='hubert')
    parser.add_argument('--dataset', type=str, default='000')
    parser.add_argument('--audio', type=str, default='test_v1.wav')
    parser.add_argument('--output', type=str, default='test.mp4')
    parser.add_argument('--checkpoint', type=str, required=True)
    args, unknown = parser.parse_known_args()
    return args, unknown


def get_audio_features(features, index):                   # [py=85][FUN_1800047b0]
    # [E1 probe_learn] 前后各 8 帧;越界用 zeros_like(auds[:pad]) 补(不足则只补已有长度)
    left = index - 8
    right = index + 8
    pad_left = 0
    pad_right = 0
    if left < 0:
        pad_left = -left
        left = 0
    if right > features.shape[0]:
        pad_right = right - features.shape[0]
        right = features.shape[0]
    auds = torch.from_numpy(features[left:right])
    if pad_left > 0:
        auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
    if pad_right > 0:
        auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0)
    return auds


def main(dataset_dir, checkpoint, audio_path, output_path, mode='hubert'):   # [py=104][FUN_180005e90]
    make_hubert(audio_path)
    audio_feat_path = audio_path.replace('.wav', '_hu.npy')
    audio_feats = np.load(audio_feat_path)
    img_dir = os.path.join(dataset_dir, 'frames')
    full_landmarks = np.load(os.path.join(dataset_dir, 'landmarks.npy'))
    len_img = len(os.listdir(img_dir))
    exm_img = cv2.imread(os.path.join(img_dir, '0.jpg'))
    h, w = exm_img.shape[:2]
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, (w, h))
    step_stride = 1
    net = DH(checkpoint, 'cuda')
    for img_idx in tqdm(range(len_img), desc='Gen video'):
        img_path = os.path.join(img_dir, str(img_idx) + '.jpg')
        img = cv2.imread(img_path)
        lms_list = full_landmarks[img_idx]
        xmin, xmax, ymin, ymax = lms_list
        crop_img = img[ymin:ymax, xmin:xmax]
        crop_img_ori = crop_img.copy()
        img_real_ex = cv2.resize(crop_img_ori, (256, 256))
        img_real_ex_ori = img_real_ex.copy()
        img_masked = img_real_ex.copy()
        img_real_ex_T = torch.from_numpy(img_real_ex.transpose(2, 0, 1)).float().unsqueeze(0)
        img_masked_T = torch.from_numpy(img_masked.transpose(2, 0, 1)).float().unsqueeze(0)
        img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=1)
        audio_feat = get_audio_features(audio_feats, img_idx)
        audio_feat = audio_feat[None, ...].transpose(1, 2).unsqueeze(-1).numpy()
        pred = net.inference(img_concat_T.numpy(), audio_feat)
        video_writer.write(pred)
    video_writer.release()
    command = 'ffmpeg -y -i result.avi -i ' + str(audio_path) + ' -strict -2 -q:v 1 ' + str(output_path)
    if platform.system() == 'Windows':
        subprocess.check_call(command, shell=False)
    else:
        subprocess.check_call(command, shell=True)
