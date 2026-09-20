# -*- coding: utf-8 -*-
# l2faceaudio_model —— S004 逆向恢复(landmark2face_wy/models/l2faceaudio_model.pyc,
# py3.8;873 行反汇编)。L2FaceAudioModel(BaseModel):音频驱动 L2 人脸生成的训练封装。
# 事实源:evidence/modules/_heygem_orphans/disasm/landmark2face_wy_models_l2faceaudio_model.pyc.dis
# 还原说明(全部函数体按 [Disassembly] 逐指令重建,未采用 pycdc 的近似输出):
#   1) 基类调用按字节码 LOAD_GLOBAL BaseModel + LOAD_METHOD __init__ + CALL_METHOD 2
#      还原为 BaseModel.__init__(self, opt)(类体无 __classcell__,故不是零参 super);
#   2) modify_commandline_options 为 staticmethod(类体 LOAD_NAME staticmethod + CALL_FUNCTION 1,
#      默认值元组 (True,)),体为 parser.set_defaults(pool_size=0, gan_mode='vanilla');
#   3) loss_names/visual_names/model_names 与 LOAD_CONST 顺序逐字一致;
#      0.999 为 Adam betas 的第二个元素(betas=(opt.beta1, 0.999)),100 为 lambda_L1,0.5 为 loss_D 系数;
#   4) define_G 调用为 9 个位置参数 + 关键字 input_size=opt.dataloader_size
#      (CALL_FUNCTION_KW 10 + names ('input_size',));
#   5) 模块 docstring:.dis 模块 [Constants] 首项为 0(非字符串),原码无模块 docstring,
#      此处按规范以 # 注释记录出处;
#   6) 未还原细节:无(9 个 code object 的常量表/名字表/分支顺序均已逐条落位)。
import torch
from .base_model import BaseModel
from . import networks


class L2FaceAudioModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'mask_B']
        if self.isTrain:
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      input_size=opt.dataloader_size)
        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain,
                                          self.gpu_ids)
        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.A_label = input['A_label'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.B_label = input['B_label'].to(self.device)
        self.mask_B = input['mask_B'].to(self.device)

    def forward(self):
        self.fake_B = self.netG(self.B_label, torch.cat((self.mask_B, self.real_A), 1))

    def backward_D(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        lambda_GAN = 1
        lambda_L1 = 100
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * lambda_GAN
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def eval_(self):
        lambda_GAN = 1
        lambda_L1 = 100
        self.forward()
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * lambda_GAN
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * lambda_L1
