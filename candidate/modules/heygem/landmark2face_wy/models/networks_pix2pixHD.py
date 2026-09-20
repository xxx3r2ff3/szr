# -*- coding: utf-8 -*-
# networks_pix2pixHD —— S004 逐字节码恢复(landmark2face_wy/models,py3.8)。
#
# 事实源:evidence/modules/_heygem_orphans/disasm/
#         modules_heygem_landmark2face_wy_models_networks_pix2pixHD.pyc.dis
#         (9552 行,31 个 code object;逐 code object 核对
#          Object Name / Arg Count / KW Only Arg Count / [Names] / [Var Names] /
#          [Constants] / [Disassembly])
# 交叉参考:/Volumes/A/vm_transfer/ns_s3/scratch_networks/pycdc/
#          models_networks_pix2pixHD.pyc.py(骨架;签名与默认值以 .dis 为准)
#
# 原模块 co_consts[0] == 0(非字符串)⇒ **原文件没有模块 docstring**,
# 故此处恢复说明全部写在 # 注释里;各 code object 的 co_consts[0] 亦均非字符串
# (首个常量是 None / 类名 / BUILD_LIST 之类),即原文件所有函数与类都没有 docstring。
#
# 模块级 IMPORT(见 .dis 模块 [Disassembly] 0-80):
#   import time / import torch / import torch.nn as nn / import functools /
#   from torch.autograd import Variable / import numpy as np /
#   from .base_function import ADAIN / from .face_model import *
#   判据:torch.nn 那条是 `LOAD_CONST 0; LOAD_CONST None; IMPORT_NAME torch.nn;
#   IMPORT_FROM nn; STORE_NAME nn`(fromlist=None ⇒ 不是 from-import,而是带点名的
#   as-import);base_function / face_model 两条的 level 常量为 1 ⇒ 相对 import。
#   (torchvision.models 的 import 位于 NLayerDiscriminator 与 Vgg19 之间,
#    见 .dis 9501-9506,下面按原位置保留。)
#
# 恢复方式:所有函数体均按 [Disassembly] 逐条还原(含分支、循环、
# MAKE_FUNCTION 前的默认值元组、CALL_FUNCTION_KW 的关键字名元组),
# 未使用任何外部先验知识补写逻辑。要点:
#   * 默认值来自类体里 MAKE_FUNCTION 前的常量元组,例如生成器类的
#     (64, 4, 9, nn.BatchNorm2d, 'reflect') ⇒
#     ngf=64, n_downsampling=4, n_blocks=9, norm_layer=nn.BatchNorm2d,
#     padding_type='reflect'。
#   * define_G 的 PirenderGenerator 三个配置 dict 由 BUILD_CONST_KEY_MAP 还原
#     (.dis 286-320)。
#   * VGGLoss.weights 常量折叠后为 [0.03125, 0.0625, 0.125, 0.25, 1.0]
#     (.dis 常量 0.03125/0.0625/0.125/0.25/1)。
#   * 'generator not implemented!' 在原文件里是 raise '...'(RAISE_VARARGS 1,
#     直接 raise 一个字符串常量),此处保持原样。
#   * 列表推导:<listcomp> code object 共 2 处
#     (LocalEnhancer.__init__ 的 [model_global[i] for i in range(len(model_global) - 3)]
#      与 MultiscaleDiscriminator.forward 的
#      [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]),
#     均按 LOAD_CLOSURE 元组 + MAKE_FUNCTION 8 还原;闭包自由变量
#     (model_global / i, num_D, self)与 [Cell Vars] 一致。
#   * __main__ 自测块(.dis 9517-9550)一并还原。
#
# 全局名对照:模块 [Names] 里的 'AssertionError' 只来自断言语句——
#   define_G/define_D 的 `len(gpu_ids) > 0` 分支里是
#   `POP_JUMP_IF_TRUE` + `LOAD_GLOBAL AssertionError` + `RAISE_VARARGS 1`(无断言消息),
#   各生成器 __init__ 的 `n_blocks >= 0` 同理;py3.8 编译 assert 时隐式
#   LOAD_GLOBAL AssertionError。源码保留 assert 原语句形态(与 networks.py /
#   networks_HD.py 同一处理口径),AssertionError 仅由这些 assert 隐式引用。
#
# 本文件无“语义重建/未还原”函数:全部函数体均由反汇编完整还原。
import time
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from .base_function import ADAIN
from .face_model import *


def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight'):
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=4, n_blocks_global=9,
             n_local_enhancers=1, n_blocks_local=3, norm='instance', gpu_ids=[], apex=False):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'wenet':
        netG = GlobalGeneratorwenet(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'globalaudio':
        netG = GlobalGeneratoraudio(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'globalaudio2':
        netG = GlobalGeneratoraudio2(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'global256to512audio':
        netG = Global256to512Generatoraudio(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'AFRSmall':
        netG = AFRSmall(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'AFR':
        netG = AFR(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'local':
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
                             n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    elif netG == 'pirender':
        netG = PirenderGenerator({'coeff_nc': 73, 'descriptor_nc': 256, 'layer': 3},
                                 {'layer': 3, 'num_res_blocks': 2, 'base_nc': 64},
                                 {'image_nc': 3, 'descriptor_nc': 256, 'max_nc': 256, 'use_spect': False})
    elif netG == 'pirenderhd':
        netG = GlobalGeneratorPirender(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'pirenderhdv2':
        netG = GlobalGeneratorPirenderv2(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'pirenderhdv3':
        netG = GlobalGeneratorPirenderv3(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    else:
        raise 'generator not implemented!'
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        if not apex:
            netG.to(gpu_ids[0])
            netG = torch.nn.DataParallel(netG, gpu_ids)
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1,
             getIntermFeat=False, gpu_ids=[], apex=False):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        if not apex:
            netD.to(gpu_ids[0])
            netD = torch.nn.DataParallel(netD, gpu_ids)
    netD.apply(weights_init)
    return netD


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [0.03125, 0.0625, 0.125, 0.25, 1.0]

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        # global generator model
        ngf_global = ngf * (2 ** n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global,
                                       n_blocks_global, norm_layer).model
        model_global = [model_global[i] for i in range(len(model_global) - 3)]
        self.model = nn.Sequential(*model_global)

        # local enhancer layers
        for n in range(1, n_local_enhancers + 1):
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.ReflectionPad2d(3),
                                nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global),
                                nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2),
                                nn.ReLU(True)]
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type,
                                               norm_layer=norm_layer)]
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global,
                                                  kernel_size=3, stride=2, padding=1, output_padding=1),
                               norm_layer(ngf_global),
                               nn.ReLU(True)]
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3),
                                   nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]
            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))
        output_prev = self.model(input_downsampled[-1])
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class AFR(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(AFR, self).__init__()
        activation = nn.ReLU(True)

        ngf_1 = ngf
        model_face = [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf_1, kernel_size=7, padding=0),
                      norm_layer(ngf_1),
                      activation]
        model_face += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                       norm_layer(128),
                       activation]
        model_face += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                       norm_layer(256),
                       activation]
        model_face += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                       norm_layer(256),
                       activation]
        model_face += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                       norm_layer(256),
                       activation]

        ngf_2 = ngf
        model_reference = [nn.ReflectionPad2d(3),
                           nn.Conv2d(input_nc, ngf_2, kernel_size=7, padding=0),
                           norm_layer(ngf_2),
                           activation]
        model_reference += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                            norm_layer(128),
                            activation]
        model_reference += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                            norm_layer(256),
                            activation]
        model_reference += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                            norm_layer(256),
                            activation]
        model_reference += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                            norm_layer(256),
                            activation]
        model_reference += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                            norm_layer(256),
                            activation]

        model2 = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model2 += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                   activation=activation, norm_layer=norm_layer)]

        model3 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / 2),
                                          kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)),
                       activation]
        model3 += [nn.ConvTranspose2d(int(ngf * mult / 2), int(ngf * mult / 2),
                                      kernel_size=3, stride=2, padding=1, output_padding=1),
                   norm_layer(int(ngf * mult / 2)),
                   activation]
        model3 += [nn.ReflectionPad2d(3),
                   nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                   nn.Tanh()]

        self.face_encoder = nn.Sequential(*model_face)
        self.reference_encoder = nn.Sequential(*model_reference)
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 64, kernel_size=3, stride=1, padding=(1, 3)),
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True))
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)

    def forward(self, audio_feature, face_feature, referenc_feature):
        a = self.audio_encoder(audio_feature)
        f = self.face_encoder(face_feature)
        r = self.reference_encoder(referenc_feature)
        x = torch.cat((a, f, r), dim=1)
        x = self.model2(x)
        out = self.model3(x)
        return out


class AFRSmall(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(AFRSmall, self).__init__()
        activation = nn.ReLU(True)

        ngf_1 = ngf // 2
        model_face = [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf_1, kernel_size=7, padding=0),
                      norm_layer(ngf_1),
                      activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                model_face += [nn.Conv2d(ngf_1 * mult, ngf_1 * mult,
                                         kernel_size=3, stride=2, padding=1),
                               norm_layer(ngf * mult),
                               activation]
                continue
            model_face += [nn.Conv2d(ngf_1 * mult, ngf_1 * mult * 2,
                                     kernel_size=3, stride=2, padding=1),
                           norm_layer(ngf_1 * mult * 2),
                           activation]

        ngf_2 = ngf // 2
        model_reference = [nn.ReflectionPad2d(3),
                           nn.Conv2d(input_nc, ngf_2, kernel_size=7, padding=0),
                           norm_layer(ngf_2),
                           activation]
        model_reference += [nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                            norm_layer(64),
                            activation]
        model_reference += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                            norm_layer(128),
                            activation]
        model_reference += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                            norm_layer(256),
                            activation]
        model_reference += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                            norm_layer(256),
                            activation]
        model_reference += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                            norm_layer(256),
                            activation]

        model2 = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model2 += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                   activation=activation, norm_layer=norm_layer)]

        model3 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / 2),
                                          kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)),
                       activation]
        model3 += [nn.ConvTranspose2d(int(ngf * mult / 2), int(ngf * mult / 2),
                                      kernel_size=3, stride=2, padding=1, output_padding=1),
                   norm_layer(int(ngf * mult / 2)),
                   activation]
        model3 += [nn.ReflectionPad2d(3),
                   nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                   nn.Tanh()]

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 64, kernel_size=3, stride=1, padding=(1, 3)),
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True))
        self.face_encoder = nn.Sequential(*model_face)
        self.reference_encoder = nn.Sequential(*model_reference)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)

    def forward(self, audio_feature, face_feature, referenc_feature):
        a = self.audio_encoder(audio_feature)
        f = self.face_encoder(face_feature)
        r = self.reference_encoder(referenc_feature)
        x = torch.cat((a, f, r), dim=1)
        x = self.model2(x)
        out = self.model3(x)
        return out


class Global256to512Generatoraudio(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Global256to512Generatoraudio, self).__init__()
        activation = nn.ReLU(True)

        model1 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                  norm_layer(ngf),
                  activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                model1 += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=2, padding=1),
                           norm_layer(ngf * mult),
                           activation]
                continue
            model1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                       norm_layer(ngf * mult * 2),
                       activation]

        model2 = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model2 += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                   activation=activation, norm_layer=norm_layer)]

        model3 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / 2),
                                          kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)),
                       activation]
        model3 += [nn.ConvTranspose2d(int(ngf * mult / 2), int(ngf * mult / 2),
                                      kernel_size=3, stride=2, padding=1, output_padding=1),
                   norm_layer(int(ngf * mult / 2)),
                   activation]
        model3 += [nn.ReflectionPad2d(3),
                   nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                   nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 64, kernel_size=3, stride=1, padding=(1, 3)),
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True))

    def forward(self, audio_feature, face_feature):
        audio_feature = self.audio_encoder(audio_feature)
        x = self.model1(face_feature)
        x = torch.cat((x, audio_feature), dim=1)
        x = self.model2(x)
        out = self.model3(x)
        return out


class GlobalGeneratoraudio2(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGeneratoraudio2, self).__init__()
        activation = nn.ReLU(True)

        model1 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                  norm_layer(ngf),
                  activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                model1 += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=2, padding=1),
                           norm_layer(ngf * mult),
                           activation]
                continue
            model1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                       norm_layer(ngf * mult * 2),
                       activation]

        model2 = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model2 += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                   activation=activation, norm_layer=norm_layer)]

        model3 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / 2),
                                          kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)),
                       activation]
        model3 += [nn.ReflectionPad2d(3),
                   nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                   nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)

        self.audio_encoder = [nn.Conv2d(1, ngf, kernel_size=7, padding=0),
                              norm_layer(ngf),
                              activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                self.audio_encoder += [nn.Conv2d(ngf * mult, ngf * mult,
                                                 kernel_size=3, stride=2, padding=1),
                                       norm_layer(ngf * mult),
                                       activation]
                continue
            self.audio_encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2,
                                             kernel_size=3, stride=2, padding=1),
                                   norm_layer(ngf * mult * 2),
                                   activation]
        self.audio_encoder = nn.Sequential(*self.audio_encoder)

    def forward(self, audio_feature, face_feature):
        x = self.model1(face_feature)
        audio_feature = self.audio_encoder(audio_feature)
        x = torch.cat((x, audio_feature), dim=1)
        x = self.model2(x)
        out = self.model3(x)
        return out


class GlobalGeneratorwenet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGeneratorwenet, self).__init__()
        activation = nn.ReLU(True)

        model1 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                  norm_layer(ngf),
                  activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                model1 += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=2, padding=1),
                           norm_layer(ngf * mult),
                           activation]
                continue
            model1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                       norm_layer(ngf * mult * 2),
                       activation]

        model2 = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model2 += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                   activation=activation, norm_layer=norm_layer)]

        model3 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / 2),
                                          kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)),
                       activation]
        model3 += [nn.ReflectionPad2d(3),
                   nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                   nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 64, kernel_size=3, stride=(1, 2), padding=(0, 0)),
            Conv2d(64, 128, kernel_size=3, stride=(1, 2), padding=(0, 0)),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=(1, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=(1, 0)),
            nn.ConvTranspose2d(256, 512, kernel_size=3, stride=(2, 1), padding=(1, 0),
                               output_padding=(1, 0)),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True))

    def forward(self, audio_feature, face_feature):
        x = self.model1(face_feature)
        x = torch.cat((x, audio_feature), dim=1)
        x = self.model2(x)
        out = self.model3(x)
        return out


class GlobalGeneratorPirender(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGeneratorPirender, self).__init__()
        activation = nn.ReLU(True)

        model1 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                  norm_layer(ngf),
                  activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                model1 += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=2, padding=1),
                           norm_layer(ngf * mult),
                           activation]
                continue
            model1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                       norm_layer(ngf * mult * 2),
                       activation]

        model2 = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model2 += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                   activation=activation, norm_layer=norm_layer)]

        model3 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / 2),
                                          kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)),
                       activation]
        model3 += [nn.ReflectionPad2d(3),
                   nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                   nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 64, kernel_size=3, stride=(1, 2), padding=(0, 0)),
            Conv2d(64, 128, kernel_size=3, stride=(1, 2), padding=(0, 0)),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=(1, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=(1, 0)),
            nn.ConvTranspose2d(256, 512, kernel_size=3, stride=(2, 1), padding=(1, 0),
                               output_padding=(1, 0)),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True))
        self.mapping_net = MappingNet(67, 256, 3)
        self.adain = ADAIN(512, 256)

    def forward(self, feature_3dmm, face_feature):
        feature_3dmm = self.mapping_net(feature_3dmm)
        x = self.model1(face_feature)
        adain_x = self.adain(x, feature_3dmm)
        x = torch.cat((x, adain_x), dim=1)
        x = self.model2(x)
        out = self.model3(x)
        return out


class Conv1d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm1d(cout)
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class GlobalGeneratorPirenderv2(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGeneratorPirenderv2, self).__init__()
        activation = nn.ReLU(True)

        model1 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                  norm_layer(ngf),
                  activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                model1 += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=2, padding=1),
                           norm_layer(ngf * mult),
                           activation]
                continue
            model1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                       norm_layer(ngf * mult * 2),
                       activation]

        model2 = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model2 += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                   activation=activation, norm_layer=norm_layer)]

        model3 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / 2),
                                          kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)),
                       activation]
        model3 += [nn.ReflectionPad2d(3),
                   nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                   nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.exp_3dmm_encoder = nn.Sequential(
            Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            Conv1d(256, 256, kernel_size=3, stride=1, padding=1, residual=True))
        self.adain = ADAIN(512, 256)

    def forward(self, feature_3dmm, face_feature):
        feature_3dmm = self.exp_3dmm_encoder(feature_3dmm)
        x = self.model1(face_feature)
        adain_x = self.adain(x, feature_3dmm)
        x = torch.cat((x, adain_x), dim=1)
        x = self.model2(x)
        out = self.model3(x)
        return out


class GlobalGeneratorPirenderv3(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGeneratorPirenderv3, self).__init__()
        activation = nn.ReLU(True)

        model1 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                  norm_layer(ngf),
                  activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                model1 += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=2, padding=1),
                           norm_layer(ngf * mult),
                           activation]
                continue
            model1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                       norm_layer(ngf * mult * 2),
                       activation]

        model2 = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model2 += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                   activation=activation, norm_layer=norm_layer)]

        model3 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / 2),
                                          kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)),
                       activation]
        model3 += [nn.ReflectionPad2d(3),
                   nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                   nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.exp_3dmm_encoder = nn.Sequential(
            Conv1d(323, 128, kernel_size=3, stride=2, padding=1),
            Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            Conv1d(256, 256, kernel_size=3, stride=1, padding=1, residual=True))
        self.adain = ADAIN(512, 256)

    def forward(self, feature_3dmm, face_feature):
        for one in self.exp_3dmm_encoder:
            feature_3dmm = one(feature_3dmm)
        x = self.model1(face_feature)
        adain_x = self.adain(x, feature_3dmm)
        x = torch.cat((x, adain_x), dim=1)
        x = self.model2(x)
        out = self.model3(x)
        return out


class GlobalGeneratoraudio(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGeneratoraudio, self).__init__()
        activation = nn.ReLU(True)

        model1 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                  norm_layer(ngf),
                  activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                model1 += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=2, padding=1),
                           norm_layer(ngf * mult),
                           activation]
                continue
            model1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                       norm_layer(ngf * mult * 2),
                       activation]

        model2 = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model2 += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                   activation=activation, norm_layer=norm_layer)]

        model3 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / 2),
                                          kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)),
                       activation]
        model3 += [nn.ReflectionPad2d(3),
                   nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                   nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 64, kernel_size=3, stride=1, padding=(1, 3)),
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True))

    def forward(self, audio_feature, face_feature):
        audio_feature = self.audio_encoder(audio_feature)
        x = self.model1(face_feature)
        x = torch.cat((x, audio_feature), dim=1)
        x = self.model2(x)
        out = self.model3(x)
        return out


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      activation]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                  activation=activation, norm_layer=norm_layer)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)),
                      activation]
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        self.output_nc = output_nc

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, inst):
        outputs = self.model(input)
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b + 1] == int(i)).nonzero()
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:, 0] + b, indices[:, 1] + j,
                                         indices[:, 2], indices[:, 3]]
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[indices[:, 0] + b, indices[:, 1] + j,
                                 indices[:, 2], indices[:, 3]] = mean_feat
        return outputs_mean


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j))
                         for j in range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != num_D - 1:
                input_downsampled = self.downsample(input_downsampled)
        return result


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                          norm_layer(nf),
                          nn.LeakyReLU(0.2, True)]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                      norm_layer(nf),
                      nn.LeakyReLU(0.2, True)]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


if __name__ == '__main__':
    a = GlobalGeneratorPirender(6, 3)
    b = torch.ones(1, 6, 256, 256)
    c = torch.ones(1, 153, 27)
    d = a(c, b)
    print(d.shape)
