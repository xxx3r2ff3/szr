# -*- coding: utf-8 -*-
# pirender_3dmm_mouth_hd_model —— S004 逆向恢复
# (landmark2face_wy/models/pirender_3dmm_mouth_hd_model.pyc,py3.8;1567 行反汇编)。
# Pirender3dmmmouthhdModel(BaseModel):3dmm 驱动 PiRender 高清口型训练封装。
# 事实源:evidence/modules/_heygem_orphans/disasm/landmark2face_wy_models_pirender_3dmm_mouth_hd_model.pyc.dis
# 还原说明(全部函数体按 [Disassembly] 逐指令重建,未采用 pycdc 的近似输出):
#   1) 基类调用按字节码 LOAD_GLOBAL BaseModel + LOAD_METHOD __init__ + CALL_METHOD 2
#      还原为 BaseModel.__init__(self, opt)(类体无 __classcell__,故不是零参 super);
#   2) name 是普通方法:类体为 LOAD_CONST <CODE> name → MAKE_FUNCTION 0 → STORE_NAME name,
#      没有 LOAD_NAME property + CALL_FUNCTION,故不加 @property;
#      modify_commandline_options 是 staticmethod(LOAD_NAME staticmethod + CALL_FUNCTION 1);
#   3) 口型裁剪框为元组解包赋值 (m_x1, m_x2, m_y1, m_y2),常量顺序
#      240 / 400 / 120 / -120 均乘 self.resize_size 再 // 512;
#   4) loss_names 9 项、visual_names 4 项与 LOAD_CONST 顺序逐字一致;
#      0.999 为 Adam betas 第二项,0.5 为 loss_D/loss_mouthD 系数,100 为 lambda_L1,
#      2 为口型 GAN/L1 损失倍率,6 为 loss_G_VGG 的除数;
#   5) apex 相关:__init__ 中 `import apex` 为函数内局部导入(STORE_FAST apex),
#      fp16 分支用 apex.amp.scale_loss(...) 上下文管理器(字节码 SETUP_WITH)包住 backward;
#      (self.netG, self.netD), (self.optimizer_G, self.optimizer_D) = apex.amp.initialize(...)
#      为嵌套元组解包(UNPACK_SEQUENCE 2 + UNPACK_SEQUENCE 2),与字节码逐一对应;
#   6) 模块 docstring:.dis 模块 [Constants] 首项为 0(非字符串),原码无模块 docstring,
#      此处按规范以 # 注释记录出处;
#   7) 未还原细节:无(10 个 code object 的常量表/名字表/分支顺序均已逐条落位)。
import torch
from .base_model import BaseModel
from . import networks_pix2pixHD as networks


class Pirender3dmmmouthhdModel(BaseModel):

    def name(self):
        return 'Pirender3dmmmouthhdModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.resize_size = opt.resize_size
        self.m_x1, self.m_x2, self.m_y1, self.m_y2 = (240 * self.resize_size // 512,
                                                     400 * self.resize_size // 512,
                                                     120 * self.resize_size // 512,
                                                     -120 * self.resize_size // 512)
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'mask_B']
        if self.isTrain:
            self.loss_names = ['G_GAN', 'G_L1', 'G_VGG', 'D_real', 'D_fake', 'G_mouthL1',
                               'G_mouthGAN', 'D_real_m', 'D_fake_m']
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        self.netG = networks.define_G(6, opt.output_nc, opt.ngf, opt.netG, opt.n_downsample_global,
                                      opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids,
                                      apex=opt.fp16)
        if self.isTrain:
            use_sigmoid = False
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.n_layers_D,
                                          opt.norm, use_sigmoid, opt.num_D,
                                          not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids,
                                          apex=opt.fp16)
        if self.isTrain:
            self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            self.criterionGAN = networks.GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            if self.opt.fp16:
                import apex
                (self.netG, self.netD), (self.optimizer_G, self.optimizer_D) = apex.amp.initialize(
                    [self.netG.to(self.device), self.netD.to(self.device)],
                    [self.optimizer_G, self.optimizer_D], opt_level='O1')
                if not opt.distributed:
                    self.netG = torch.nn.DataParallel(self.netG, device_ids=opt.gpu_ids)
                    self.netD = torch.nn.DataParallel(self.netD, device_ids=opt.gpu_ids)
                else:
                    self.netG = apex.parallel.DistributedDataParallel(self.netG,
                                                                       delay_allreduce=True)
                    self.netD = apex.parallel.DistributedDataParallel(self.netD,
                                                                       delay_allreduce=True)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device).float()
        self.A_label = input['A_label'].to(self.device).float()
        self.real_B = input['B'].to(self.device).float()
        self.B_label = input['B_label'].to(self.device).float()
        self.mask_B = input['mask_B'].to(self.device).float()

    def forward(self):
        self.fake_B = self.netG(self.B_label, torch.cat((self.mask_B, self.real_A), 1))

    def backward_D(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        if self.opt.fp16:
            import apex
            with apex.amp.scale_loss(self.loss_D, self.optimizer_D) as scaled_loss:
                scaled_loss.backward()
        else:
            self.loss_D.backward()
        m_fake_AB = torch.cat((self.real_A[:, :, self.m_x1:self.m_x2, self.m_y1:self.m_y2],
                               self.fake_B[:, :, self.m_x1:self.m_x2, self.m_y1:self.m_y2]), 1)
        m_pred_fake = self.netD(m_fake_AB.detach())
        self.loss_D_fake_m = self.criterionGAN(m_pred_fake, False)
        m_real_AB = torch.cat((self.real_A[:, :, self.m_x1:self.m_x2, self.m_y1:self.m_y2],
                               self.real_B[:, :, self.m_x1:self.m_x2, self.m_y1:self.m_y2]), 1)
        m_pred_real = self.netD(m_real_AB)
        self.loss_D_real_m = self.criterionGAN(m_pred_real, True)
        self.loss_mouthD = (self.loss_D_fake_m + self.loss_D_real_m) * 0.5
        if self.opt.fp16:
            import apex
            with apex.amp.scale_loss(self.loss_mouthD, self.optimizer_D) as scaled_loss:
                scaled_loss.backward()
        else:
            self.loss_mouthD.backward()

    def backward_G(self):
        lambda_GAN = 1
        lambda_L1 = 100
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * lambda_GAN
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * lambda_L1
        m_fake_AB = torch.cat((self.real_A[:, :, self.m_x1:self.m_x2, self.m_y1:self.m_y2],
                               self.fake_B[:, :, self.m_x1:self.m_x2, self.m_y1:self.m_y2]), 1)
        m_pred_fake = self.netD(m_fake_AB)
        self.loss_G_mouthGAN = self.criterionGAN(m_pred_fake, True) * lambda_GAN * 2
        self.loss_G_mouthL1 = self.criterionL1(
            self.fake_B[:, :, self.m_x1:self.m_x2, self.m_y1:self.m_y2],
            self.real_B[:, :, self.m_x1:self.m_x2, self.m_y1:self.m_y2]) * lambda_L1 * 2
        self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B) * lambda_L1 / 6
        self.loss_G = (self.loss_G_GAN + self.loss_G_L1 + self.loss_G_VGG + self.loss_G_mouthL1 +
                       self.loss_G_mouthGAN)
        if self.opt.fp16:
            import apex
            with apex.amp.scale_loss(self.loss_G, self.optimizer_G) as scaled_loss:
                scaled_loss.backward()
        else:
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
        self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B) * lambda_L1 / 6
        self.loss_G_mouthL1 = self.criterionL1(
            self.fake_B[:, :, self.m_x1:self.m_x2, self.m_y1:self.m_y2],
            self.real_B[:, :, self.m_x1:self.m_x2, self.m_y1:self.m_y2]) * lambda_L1 * 2
