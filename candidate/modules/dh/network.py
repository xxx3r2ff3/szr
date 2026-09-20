# -*- coding: utf-8 -*-
"""modules.dh.network —— 语义重建候选(R013 / T4R)。

源 pyd : modules/dh/network.cp310-win_amd64.pyd
sha256 : ebd5a267660cd911285cc7d6a4c4ec8f0fa3265dd9a0e0f00dad57dd6f553ca2

证据(行内注释给出出处):
  [FUN_xxxxxxxx]  Ghidra 无头反编译 `evidence/modules/dh__network/static/pseudocode/
                  dh__network__FUN_xxxxxxxx__xxxxxxxx.c`;DAT_ 槽位已按
                  szr2026_out/diff_harness/stringtab_dump.py 还原为 STR("...")
  [py=N]          Cython code 对象 co_firstlineno / 反编译内 `iVar = 0xN` 行号标记
  [E1 xxx]        VM 内 oracle 解释器实测(ns_w2dh 共享目录同名 *_out.json)
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """[py=8][FUN_1800012d0] CBAM 空间注意力;[py=14][FUN_180001d00] forward。"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # [E1 probe_dh_params] conv1=Conv2d(2,1,kernel=7,stride=1,padding=3,bias=False)
        # [FUN_1800012d0] 缓存整数 2/1 + kwargs padding/bias
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # [FUN_180001d00] STR 序: torch|mean|dim|keepdim|max|cat|conv1|sigmoid
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SAM(nn.Module):
    """[py=23][FUN_180002940] 空间注意力包装;[py=27][FUN_180001af0] forward。"""

    def __init__(self):
        super(SAM, self).__init__()
        self.sa = SpatialAttention()

    def forward(self, sp, se):
        # [FUN_180003080] py=28 `sp_att=self.sa(sp)`;py=29 `out=se*sp_att+se`
        # 反编译:PyNumber_Multiply(param_4=se, sa(sp)) 后 PyNumber_Add(..., se)
        sp_att = self.sa(sp)
        out = se * sp_att + se
        return out


class InvertedResidual(nn.Module):
    """[py=34][FUN_1800036e0] MobileNetV2 式倒残差;[py=60][FUN_180005510] forward。

    [FUN_1800036e0] `inp * expand_ratio` 内联展开 4 次(PyNumber_Multiply ×4,
    co_varnames 无 hidden_dim 局部);深度卷积 groups=inp*expand_ratio、
    全部 Conv2d bias=False、每次卷积后 BatchNorm2d、激活 nn.ReLU(inplace=True)。
    """

    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = use_res_connect
        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1,
                      groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        # [FUN_180005510] STR 序 use_res_connect|conv(先取属性再判真假)
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DoubleConvDW(nn.Module):
    """[py=69][FUN_180005b40];[py=89][FUN_1800065a0] forward。

    [FUN_180005b40] 两个 InvertedResidual:第一个位置传 `stride`,第二个显式
    stride=1;两者 use_res_connect=False、expand_ratio=2。
    """

    def __init__(self, in_channels, out_channels, stride=2):
        super(DoubleConvDW, self).__init__()
        self.double_conv = nn.Sequential(
            InvertedResidual(in_channels, out_channels, stride, False, 2),
            InvertedResidual(out_channels, out_channels, 1, True, 2),
        )

    def forward(self, x):
        return self.double_conv(x)


class InConvDw(nn.Module):
    """[py=94][FUN_180006b50];[py=106][FUN_1800072e0] forward。"""

    def __init__(self, in_channels, out_channels):
        super(InConvDw, self).__init__()
        # [FUN_180006b50] stride=1, use_res_connect=False, expand_ratio=2
        self.inconv = nn.Sequential(
            InvertedResidual(in_channels, out_channels, 1, False, 2),
        )

    def forward(self, x):
        return self.inconv(x)


class Down(nn.Module):
    """[py=112][FUN_180007890];[py=119][FUN_180007fc0] forward。"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        # [FUN_180007890] 仅一个 DoubleConvDW(显式 stride=2);无 MaxPool2d 串
        self.maxpool_conv = nn.Sequential(
            DoubleConvDW(in_channels, out_channels, stride=2),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """[py=125][FUN_180008570];[py=130][FUN_180008fb0] forward。

    [FUN_180008570] up=Upsample(scale_factor=2, mode='bilinear', align_corners=True);
                    conv=DoubleConvDW(in_channels, out_channels, stride=1)
    [FUN_180008fb0] x1=self.up(x1);按 x2 的 H/W 求 diffY/diffX → F.pad →
                    torch.cat([x1, x2], axis=1) → self.conv
    """

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConvDW(in_channels, out_channels, stride=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], axis=1)
        return self.conv(x)


class OutConv(nn.Module):
    """[py=142][FUN_180009f50];[py=146][FUN_18000a4a0] forward。"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # [FUN_180009f50] kernel_size=1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AudioConvWenet(nn.Module):
    """[py=151][FUN_18000a7b0];[py=180][FUN_18000c680] forward。

    [FUN_18000a7b0] ch=[32,64,128,256,512](PyList_New(5));
      conv1/conv2=IR(ch[3],ch[4],stride=1,...);conv3=Conv2d(ch[3],ch[3],k=3,p=1,
      stride=(1,2));bn3=BatchNorm2d(ch[3]);conv4=IR(ch[3],ch[4],stride=1,...);
      conv5=Conv2d(ch[3],ch[4],k=3,p=3,stride=2);bn5=BatchNorm2d(ch[4]);relu=ReLU();
      conv6/conv7=IR(ch[4],ch[4],stride=1,...)
    [E1 probe_dh_params] conv3.stride=(1,2) 元组、conv5.stride=2/padding=3
    """

    def __init__(self):
        super(AudioConvWenet, self).__init__()
        ch = [32, 64, 128, 256, 512]
        self.conv1 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True,
                                      expand_ratio=2)
        self.conv2 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True,
                                      expand_ratio=2)
        self.conv3 = nn.Conv2d(ch[3], ch[3], kernel_size=3, padding=1, stride=(1, 2))
        self.bn3 = nn.BatchNorm2d(ch[3])
        self.conv4 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True,
                                      expand_ratio=2)
        self.conv5 = nn.Conv2d(ch[3], ch[4], kernel_size=3, padding=3, stride=2)
        self.bn5 = nn.BatchNorm2d(ch[4])
        self.relu = nn.ReLU()
        self.conv6 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True,
                                      expand_ratio=2)
        self.conv7 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True,
                                      expand_ratio=2)

    def forward(self, x):
        # [FUN_18000c680] STR 序 conv1|conv2|relu|bn3|conv3|conv4|bn5|conv5|conv6|conv7
        # [E1 probe_dh_fix] 子模块 hook 次序:conv1,conv2,conv3,bn3,relu,conv4,conv5,bn5,relu,conv6,conv7
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        x = self.conv7(x)
        return x


class AudioConvHubert(nn.Module):
    """[py=198][FUN_18000d0f0];[py=227][FUN_18000efa0] forward。

    [FUN_18000d0f0] ch=[32,64,128,256,512];
      conv1=IR(ch[0],ch[1],stride=1,...);conv2=IR(ch[1],ch[2],stride=1,...);
      conv3=Conv2d(ch[2],ch[3],k=3,p=1,stride=(2,2));bn3=BatchNorm2d(ch[3]);
      conv4=IR(ch[3],ch[3],stride=1,...);conv5=Conv2d(ch[3],ch[4],k=3,p=3,stride=2);
      bn5=BatchNorm2d(ch[4]);relu=ReLU();conv6/conv7=IR(ch[4],ch[4],stride=1,...)
    [E1 probe_dh_params] conv3.stride=(2,2) 元组
    """

    def __init__(self):
        super(AudioConvHubert, self).__init__()
        ch = [32, 64, 128, 256, 512]
        self.conv1 = InvertedResidual(ch[0], ch[1], stride=1, use_res_connect=False,
                                      expand_ratio=2)
        self.conv2 = InvertedResidual(ch[1], ch[2], stride=1, use_res_connect=False,
                                      expand_ratio=2)
        self.conv3 = nn.Conv2d(ch[2], ch[3], kernel_size=3, padding=1, stride=(2, 2))
        self.bn3 = nn.BatchNorm2d(ch[3])
        self.conv4 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True,
                                      expand_ratio=2)
        self.conv5 = nn.Conv2d(ch[3], ch[4], kernel_size=3, padding=3, stride=2)
        self.bn5 = nn.BatchNorm2d(ch[4])
        self.relu = nn.ReLU()
        self.conv6 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True,
                                      expand_ratio=2)
        self.conv7 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True,
                                      expand_ratio=2)

    def forward(self, x):
        # [FUN_18000efa0] 与 Wenet 版同构
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        x = self.conv7(x)
        return x


class Model(nn.Module):
    """[py=245][FUN_18000fc80];[py=274][FUN_180012500] forward。

    [FUN_18000fc80] ch=[32,64,128,256,512];mode=='hubert'→AudioConvHubert,
      mode=='wenet'→AudioConvWenet;fuse_conv=Sequential(DoubleConvDW(ch[4]*2,ch[4],
      stride=1), DoubleConvDW(ch[4],ch[3],stride=1))(仅两处显式 stride=1 的 kwargs);
      outc=OutConv(ch[0],3)。
    [E1 probe_dh_patch] 实测:up1(256,256)→128、up2(128,128)→64、up3(64,64)→32、
      up4(32,32)→32、outc(32)→3;audio_model 输出 512ch 与 down4 输出 512ch 拼接。
    """

    def __init__(self, n_channels=6, mode='hubert'):
        super(Model, self).__init__()
        ch = [32, 64, 128, 256, 512]
        if mode == 'hubert':
            self.audio_model = AudioConvHubert()
        elif mode == 'wenet':
            self.audio_model = AudioConvWenet()
        self.fuse_conv = nn.Sequential(
            DoubleConvDW(ch[4] * 2, ch[4], stride=1),
            DoubleConvDW(ch[4], ch[3], stride=1),
        )
        self.n_channels = n_channels
        self.inc = InConvDw(n_channels, ch[0])
        self.down1 = Down(ch[0], ch[1])
        self.down2 = Down(ch[1], ch[2])
        self.down3 = Down(ch[2], ch[3])
        self.down4 = Down(ch[3], ch[4])
        self.up1 = Up(ch[4], ch[2])
        self.up2 = Up(ch[3], ch[1])
        self.up3 = Up(ch[2], ch[0])
        self.up4 = Up(ch[1], ch[0])
        self.outc = OutConv(ch[0], 3)

    def forward(self, x, audio_feat):
        # [FUN_180012500] 行号标记 0x114..0x11d = 276..285
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        audio_feat = self.audio_model(audio_feat)
        x5 = torch.cat([x5, audio_feat], axis=1)
        x5 = self.fuse_conv(x5)
        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.outc(out)
        return F.sigmoid(out)


def check_onnx(onnx_path, torch_in, audio):
    """[py=294][FUN_180013910] onnx 导出后自检。

    注:oracle 侧该函数体内**没有** onnx/onnxruntime/time 的模块级或函数级 import
    (E1:[E1 diff_smoke] oracle 实测 `NameError: name 'onnx' is not defined`),
    该函数在冻结版本中为不可用的死路径;候选按二进制原样保留(不得"修好"它)。
    """
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    providers = ['CUDAExecutionProvider']
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    ort_inputs = {}
    ort_inputs[ort_session.get_inputs()[0].name] = torch_in.cpu().numpy()
    ort_inputs[ort_session.get_inputs()[1].name] = audio.cpu().numpy()
    for i in range(10):
        t1 = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
        t2 = time.time()
        print(t2 - t1)


def export_onnx(pth_file):
    """[py=312][FUN_180014cc0] 加载 ckpt → 导出同目录 model.onnx。"""
    import onnx
    net = Model()
    net.eval()
    ckpt = torch.load(pth_file)
    net.load_state_dict(ckpt)
    img = torch.zeros(1, 6, 256, 256)
    audio = torch.zeros(1, 32, 56, 56)
    input_dict = (img, audio)
    onnx_path = os.path.join(os.path.dirname(pth_file), 'model.onnx')
    with torch.no_grad():
        torch_out = net(img, audio)
        onnx.export(net, input_dict, onnx_path, input_names=['input', 'audio'],
                    output_names=['output'], opset_version=11, export_params=True)

