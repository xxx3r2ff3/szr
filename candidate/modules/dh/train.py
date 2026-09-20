# -*- coding: utf-8 -*-
"""modules.dh.train —— 语义重建候选(R016 / T4R)。

源 pyd : modules/dh/train.cp310-win_amd64.pyd
sha256 : 2bffc421dcf48517c7f7fe7fe19054bd215b5529da51bff6d0153b8b81743075
证据   : evidence/modules/dh__train/static/pseudocode/[FUN_xxxxxxxx];E1 见行内注释。
"""
import argparse
import io
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# [I008 E2E 定谳 i008-fix-6] 双上下文导入:本模块在包内以 modules.dh.train 导入
# (G2 用例),在 start_train.py 子进程里以 sys.path[0]=modules/dh 裸名 `train`
# 导入(oracle pyd 同此形态)——后者无 `modules` 顶层包,须回退裸名 `network`。
try:
    from modules.dh.network import Model
except ImportError:
    from network import Model

# [E1 probe_q] logloss 是模块级 nn.BCELoss() 实例(不是函数)
logloss = nn.BCELoss()


class MyDataset(Dataset):
    """[py=20][FUN_1800013d0];[py=36] __len__;[py=42] get_audio_features;
    [py=62][FUN_180004d90] process_img;[py=93][FUN_180007c00] __getitem__。"""

    def __init__(self, img_dir, mode):
        self.img_dir = img_dir
        self.img_landmarks = np.load(os.path.join(img_dir, 'landmarks.npy'))
        self.mode = mode
        frames = os.path.join(img_dir, 'frames')
        self.img_path_list = []
        for i, img_path in enumerate(sorted(os.listdir(frames))):
            if img_path.endswith('.jpg'):
                self.img_path_list.append(os.path.join(frames, img_path))
        if mode == 'wenet':
            self.audio_feats = np.load(os.path.join(img_dir, 'aud_wenet.npy')).astype(np.float32)
        else:
            self.audio_feats = np.load(os.path.join(img_dir, 'aud_hu.npy')).astype(np.float32)

    def __len__(self):
        # [E1 probe_learn2] 4 帧目录 → len == 3
        return len(self.img_path_list) - 1

    def get_audio_features(self, features, index):          # [py=42][FUN_1800037b0]
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

    def process_img(self, img, lms, img_ex, lms_ex):        # [py=62][FUN_180004d90]
        lms_list = np.array(lms).astype(np.int32)
        xmin, xmax, ymin, ymax = lms_list
        crop_img = img[ymin:ymax, xmin:xmax]
        img_real = cv2.resize(crop_img, (256, 256))
        img_real_ex = cv2.resize(img_real, (128, 128), interpolation=cv2.INTER_AREA)
        img_real_ori = img_real.copy()
        if img_ex is not None:
            # [I008 E2E 定谳 i008-fix-4] img_masked 必须派生自 128×128 的
            # img_real_ex:img_concat_T = cat([img_real_ex_T, img_masked_T], axis=1)
            # 要求两支同空间尺寸,E2E dh_unit 实测候选(源 256 的副本)在装载期即抛
            # 'Sizes of tensors must match … Expected size 128 but got size 256'
            # 而 oracle 侧同夹具装载成功并到达 CUDA 门控。
            img_masked = img_real_ex.copy()
            img_masked[:, :, :] = 0
        else:
            img_masked = img_real_ex.copy()
        img_real_ex_T = torch.from_numpy(
            img_real_ex.transpose(2, 0, 1)).float().unsqueeze(0)
        img_real_T = torch.from_numpy(
            img_real.transpose(2, 0, 1)).float().unsqueeze(0)
        img_masked_T = torch.from_numpy(
            img_masked.transpose(2, 0, 1)).float().unsqueeze(0)
        img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=1)
        return img_concat_T, img_real_T

    def __getitem__(self, idx):                             # [py=93][FUN_180007c00]
        img_path = self.img_path_list[idx]
        img = cv2.imread(img_path)
        lms = self.img_landmarks[idx]
        ex_int = random.randint(0, len(self.img_path_list) - 1)
        img_ex = cv2.imread(self.img_path_list[ex_int])
        lms_ex = self.img_landmarks[ex_int]
        img_concat_T, img_real_T = self.process_img(img, lms, img_ex, lms_ex)
        if self.mode == 'wenet':
            audio_feat = self.get_audio_features(self.audio_feats, idx)
            audio_feat = audio_feat.reshape(256, 16, 32)
        else:
            audio_feat = self.get_audio_features(self.audio_feats, idx)
            audio_feat = audio_feat.reshape(32, 32, 32)
        return img_concat_T, img_real_T, audio_feat


class PerceptualLoss(object):
    """[py=143][FUN_18000acd0] contentFunc;[py=163] __init__;[py=167][FUN_18000c6b0] get_loss。"""

    def __init__(self, loss):
        self.cnn = self.contentFunc()
        self.criterion = loss

    def contentFunc(self):
        # [FUN_18000acd0] vgg19-dcbb9e9d.pth / models.vgg19(...) / load_state_dict / features
        conv_3_3_layer = 14
        # [FUN_18000acd0] os.path.exists('vgg19-dcbb9e9d.pth') —— cwd 相对(E1 实测:
        # 沙箱 cwd 下缺失 ⇒ 走 else 的 models.vgg19(pretrained=True) 真下载 548MiB)
        vgg_file = 'vgg19-dcbb9e9d.pth'
        if os.path.exists(vgg_file):
            vgg_model = models.vgg19(pretrained=False)
            vgg_model.load_state_dict(torch.load(vgg_file))
        else:
            vgg_model = models.vgg19(pretrained=True)
        vgg_model = vgg_model.features
        cnn = nn.Sequential()
        model = vgg_model
        i = 0
        for layer in model:
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            cnn.add_module(name, layer)
            if i >= conv_3_3_layer:
                break
        # [I008 E2E 定谳 i008-fix-4] oracle 的 train() 在**首行 PerceptualLoss
        # 构造期内**即触 CUDA——E2E dh_unit 双侧观测:train_resume_gate 与
        # train_exhausted(空循环)oracle 侧均在 'Loading checkpoint' 打印之前抛
        # RuntimeError('Found no NVIDIA driver …'),候选原实现迟至循环内
        # imgs.cuda() 才触 CUDA(且空循环路径先抛 UnboundLocalError),与 oracle
        # 不符。vgg 特征网随构造迁入 CUDA 即复现该门控点(本机无驱动 ⇒ 两侧同型
        # 异常;GPU 环境下即为 vgg 上卡,与反编译骨架相容)。
        cnn = cnn.cuda()
        return cnn

    def get_loss(self, fakeIm, realIm):                     # [py=167][FUN_18000c6b0]
        f_fake = self.cnn(fakeIm)
        f_real = self.cnn(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss


def cosine_loss(a, v, y):                                   # [py=178][FUN_18000cef0]
    # [E1 probe_learn] d=cosine_similarity(a,v);loss=logloss(d.unsqueeze(1), y)
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)
    return loss


def collate_fn(batch):                                      # [py=185][FUN_18000d500]
    # [E1 probe_learn] 三元组逐位 stack
    img_concat_T = torch.stack([item[0] for item in batch])
    img_real_T = torch.stack([item[1] for item in batch])
    audio_feat = torch.stack([item[2] for item in batch])
    return (img_concat_T, img_real_T, audio_feat)


def decrypt_checkpoint(encrypted_model_path):               # [py=199][FUN_18000e1d0]
    # [E1 probe_learn2] 合法 token → io.BytesIO(明文);非法 → cryptography InvalidToken
    from cryptography.fernet import Fernet
    # [COM-D002 语义替换] 原为固定 Fernet 密钥字面量(值已按纪律删除:不入库、
    # 不入报告)。密钥改由凭据保险箱/配置提供:环境变量 SZR_FERNET_KEY →
    # config.ini [Credentials] fernet_key → 缺失即显式 RuntimeError(硬失败),
    # 绝不回落到任何内置字面量(与 modules.core.train_util 同一口径)。
    key = os.environ.get('SZR_FERNET_KEY', '').strip()
    if not key:
        _parser = __import__('configparser').ConfigParser()
        _parser.read(os.path.join(os.getcwd(), 'config.ini'), encoding='utf-8')
        key = (_parser.get('Credentials', 'fernet_key', fallback='') or '').strip()
    if not key:
        raise RuntimeError(
            '缺少 Fernet 密钥:请通过凭据保险箱/配置提供(环境变量 SZR_FERNET_KEY '
            '或 config.ini [Credentials] fernet_key);客户端不内置任何密钥。')
    cipher_suite = Fernet(key)
    with open(encrypted_model_path, 'rb') as f:
        encrypted_data = f.read()
    decrypted_data = cipher_suite.decrypt(encrypted_data)
    return io.BytesIO(decrypted_data)


def get_args():                                             # [py=117][FUN_180009b50]
    # [E1 probe_learn] Namespace 顺序与默认值逐项实测
    parser = argparse.ArgumentParser(
        description='Train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_syncnet', action='store_true',
                        help="if use syncnet, you need to set 'syncnet_checkpoint'")
    parser.add_argument('--syncnet_checkpoint', type=str, default='')
    parser.add_argument('--dataset_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None,
                        help='trained model save path.')
    parser.add_argument('--sample_image', type=str, default=False)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--asr', type=str, default='hubert')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--best_loss', type=float, default=0.006)
    args, unknow = parser.parse_known_args()
    return args, unknow


def train(args, net, epoch, batch_size, lr):                # [py=211][FUN_180011070]
    # [FUN_180011070] 串序首个调用是 PerceptualLoss(nn.MSELoss())(E1 实测:该句会先触发
    # contentFunc,故缺失本地 vgg19 权重时才表现为下载/超时),其后才读 args.save_dir
    criterion = PerceptualLoss(nn.MSELoss())
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sample_image = args.sample_image
    dataset_dir = args.dataset_dir
    dataset = MyDataset(dataset_dir, args.asr)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=0, collate_fn=collate_fn)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    use_basemodel = False
    start_epoch = 0
    if args.checkpoint:
        print('Loading checkpoint {}'.format(args.checkpoint))
        ckpt = torch.load(args.checkpoint)
        start_epoch = ckpt['epoch']
        net.load_state_dict(ckpt['state_dict'])
    for e in range(start_epoch, epoch):
        avg_loss = 0.0
        for batch in tqdm(train_dataloader):
            imgs, labels, audio_feat = batch
            imgs = imgs.cuda()
            labels = labels.cuda()
            audio_feat = audio_feat.cuda()
            preds = net(imgs, audio_feat)
            loss_PerceptualLoss = criterion.get_loss(preds, labels)
            loss_pixel = nn.L1Loss()(preds, labels)
            loss = loss_PerceptualLoss + loss_pixel
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        print('epoch: %d loss: %f' % (e, avg_loss / len(train_dataloader)))
        save_best(net, save_dir, e, avg_loss)
    return avg_loss


def save_best(net, save_dir, epoch, loss):                  # [py=185-ish][FUN_18000f0d0]
    files = os.listdir(save_dir)
    best_file = None
    for file in files:
        if not os.path.isdir(os.path.join(save_dir, file)):
            continue
        try:
            _, ep = file.replace('.pth', '').split('_')
            if float(ep) == float(epoch):
                best_file = file
        except ValueError:
            continue
    if best_file is None:
        torch.save(net.state_dict(), os.path.join(save_dir, 'epoch_%d.pth' % epoch))


def main(video_file):                                       # [py=345][FUN_180019d40]
    # [E1 probe_learn2] 无 GPU → 早期 RuntimeError('Found no NVIDIA driver ...')
    args, unknow = get_args()
    dataset_dir = args.dataset_dir
    save_dir = args.save_dir
    checkpoint = args.checkpoint
    net = Model()
    if args.asr == 'wenet':
        net = net.cuda()
    else:
        net = net.cuda()
    train(args, net, args.epochs, args.batchsize, args.lr)
