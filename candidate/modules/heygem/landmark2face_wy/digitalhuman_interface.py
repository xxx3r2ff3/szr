# -*- coding: utf-8 -*-
# digitalhuman_interface.py —— landmark2face_wy(py3.8)静态恢复件
#
# 事实源:
#   evidence/modules/_heygem_orphans/disasm/landmark2face_wy_digitalhuman_interface.pyc.dis
# 以及只读原版 pyc 的逐 code object 转储(co_argcount / co_kwonlyargcount /
# co_varnames / co_names / co_consts,比 .dis 文本更精确)。
#
# 还原说明 / 语义重建与未还原细节:
#   * 模块级 [Constants][0] 是整数 0(不是字符串)⇒ 本模块无 docstring,出处只写在
#     本 # 注释块里(REPAIR_GUIDELINES §2)。各 code object 的 co_consts[0] 都是 None,
#     所以类/函数都没有 docstring。
#   * 导入块严格按模块 [Disassembly] 的 IMPORT_NAME / IMPORT_FROM / STORE_NAME 顺序还原。
#     其中 `import torchvision.transforms as transforms` 与 `import torch.nn.functional as F`
#     两条的 fromlist 常量在 co_consts 里是 None(不是元组),即 py3.8 的
#     `import a.b as c` / `import a.b.c as d` 形式(后者带 IMPORT_FROM nn; ROT_TWO; POP_TOP)。
#   * cv2 / np 没有出现在模块导入块里,由 `from landmark2face_wy.util.util import *` 提供;
#     按字节码照抄,不额外臆造 import。
#   * XsegNet / FaceAttr 是 __init__ 内部的条件 from-import(IMPORT_NAME xseg.dfl_xseg_api /
#     face_attr_detect.face_attr + IMPORT_FROM + STORE_FAST),不是模块级导入。
#   * __init__ 的第 4 个形参 face_blur_detect 默认值 False 来自类体常量 (False,);类体里
#     blend_dynamic / chaofen_before 没有 defaults(共 4 个 defaults 元组,分别属于
#     __init__(False,)、tensor_norm(None,)、tensor_norm_no_training(None,)、blendImages(0.1,))。
#   * get_face_mask2 末尾字节码是 `LOAD_FAST amask; RETURN_VALUE` —— 返回的是 amask,
#     不是上一行算出的 eroded_mask(反直觉,但按字节码照抄);形参 delta 在函数体内未被使用。
#   * inference / inference1 / inference_notraining 的形参 params 在函数体内未被使用。
#   * blendImages 使用 self.w_gpu(该类 __init__ 从未给这个属性赋值),featherAmount 未被使用,
#     src_gpu / dst_gpu 只是 src / dst 的别名 —— 全部按字节码照抄,不做"修正"。
#   * 切片常量按字节码原样还原为 `int(0 * (self.img_size / 256))` 这类写法(常量 0 / -10 /
#     5 / -5 与 self.img_size / 256 的乘法次序、以及 FLOOR_DIVIDE(`//`)与 TRUE_DIVIDE(`/`)
#     的区别都逐条对齐)。
#   * 索引形状同样按 BUILD_TUPLE 的实参个数逐条对齐(不是凭语义推测):
#     - `mask_B[:, :, [2, 1, 0]]` / `B_img[:, :, [2, 1, 0]]`:BUILD_TUPLE 3(3 维 numpy HWC);
#     - netG 输出 `[:, [2, 1, 0], :, :]`:BUILD_TUPLE 4(4 维 NCHW,列表在第 2 位);
#     - `landmarks[5, :]` / `landmarks[11, :]`:索引元组是 (5, slice(None, None)),
#       字节码压栈次序为 CONST 5 -> CONST None -> CONST None -> BUILD_SLICE 2 -> BUILD_TUPLE 2
#       (即取第 5 / 第 11 号关键点这一行,不是 `[5:]` 切片,也不是 newaxis)。
#   * 未能从字节码还原的部分:params / delta / featherAmount 的语义用途本文件无从推断
#     (它们确实未被使用);self.w_gpu 的赋值处不在本文件内。除此之外无语义缺口。
from landmark2face_wy.options.test_options import TestOptions
import torchvision.transforms as transforms
from landmark2face_wy.models.l2faceaudio_model import L2FaceAudioModel
from landmark2face_wy.util.util import *
import torch
import time
import math
import torch.nn.functional as F
from face_lib.face_restore import GFPGAN
from y_utils.config import GlobalConfig


class DigitalHumanModel:

    def __init__(self, blend_dynamic, chaofen_before, face_blur_detect=False):
        self.blend = True
        self.opt = TestOptions().parse()
        self.isTrain = False
        temp_model = torch.load(self.opt.model_path)
        self.netG = temp_model['model_name']
        self.opt.netG = temp_model['model_name']
        self.opt.dataloader_size = temp_model['model_input_size'][0]
        self.ngf = temp_model['model_ngf']
        self.img_size = temp_model['model_input_size'][0]
        self.fuse_mask = temp_model['fuse_mask']
        self.fuse_mask = cv2.resize(self.fuse_mask, (self.img_size, self.img_size))
        self.mask_re_cuda = torch.tensor(temp_model['input_mask_re']).unsqueeze(0).unsqueeze(0).cuda().half()
        self.mask_cuda = torch.tensor(temp_model['input_mask']).unsqueeze(0).unsqueeze(0).cuda().half()
        self.fuse_mask_cuda = torch.tensor(self.fuse_mask).unsqueeze(0).unsqueeze(0).cuda().repeat(1, 3, 1, 1).half()
        self.nblend = temp_model['nblend']
        self.model = L2FaceAudioModel(self.opt)
        self.drivered_wh = temp_model['wh']
        self.model.netG.load_state_dict(temp_model['face_G'])
        self.model.netG.cuda()
        self.model.eval()
        if blend_dynamic == "xseg":
            from xseg.dfl_xseg_api import XsegNet
            self.xseg = XsegNet(model_name='xseg_net_private', provider='gpu')
        if chaofen_before == 1:
            self.gfpgan = GFPGAN(model_type='GFPGANv1.4', provider='gpu')
        self.face_blur_detect = face_blur_detect
        if self.face_blur_detect:
            from face_attr_detect.face_attr import FaceAttr
            self.face_attr = FaceAttr(model_name='face_attr_mbnv3', provider='gpu')

    def tensor_norm(self, img_tensor, mask=None):
        img_tensor = img_tensor / 127.5 - 1
        if mask is not None:
            return (img_tensor + 1) * mask - 1
        return img_tensor

    def tensor_norm_no_training(self, img_tensor, mask=None):
        img_tensor = img_tensor / 255.0
        if mask is not None:
            return img_tensor * mask
        return img_tensor

    def inference(self, audio_info, face_data_dict, this_batch, start_idx, params):
        audio_idx, wenet_feature = audio_info
        B_img_list = []
        B_img__list = []
        mask_B_list = []
        mask_B_pre_list = []
        lab_list = []
        for i in range(this_batch):
            img_idx = start_idx + i
            mask_B_pre = self.gfpgan.forward(face_data_dict[img_idx]['crop_img'])
            mask_B = mask_B_pre[int(0 * (self.img_size / 256)):int(-10 * (self.img_size / 256)),
                                int(5 * (self.img_size / 256)):int(-5 * (self.img_size / 256))]
            B_img = mask_B.copy()
            mask_B = torch.from_numpy(mask_B[:, :, [2, 1, 0]].transpose(2, 0, 1))
            B_img_ = B_img.copy()
            B_img_ = torch.from_numpy(B_img_[:, :, [2, 1, 0]].transpose(2, 0, 1))
            B_img_list.append(torch.from_numpy(B_img.transpose(2, 0, 1)))
            B_img__list.append(B_img_)
            mask_B_pre_list.append(mask_B_pre)
            mask_B_list.append(mask_B)
            lab = wenet_feature.transpose(1, 0)[audio_idx[start_idx + i][0]:audio_idx[start_idx + i][1]][np.newaxis, ...]
            lab_list.append(lab)
        model_st = time.time()
        torch.cuda.synchronize()
        if this_batch > 0:
            lab = torch.tensor(lab_list).cuda().half()
            mask_B = torch.stack(mask_B_list).cuda().half()
            mask_B = self.tensor_norm(mask_B, mask=self.mask_cuda.repeat(this_batch, 3, 1, 1))
            B_img_ = torch.stack(B_img__list).cuda().half()
            B_img_ = self.tensor_norm(B_img_, mask=self.mask_re_cuda.repeat(this_batch, 3, 1, 1))
            B_img = torch.stack(B_img_list).cuda().half()
            B_img = self.tensor_norm(B_img)
            fake_B = self.model.netG(lab, torch.cat((mask_B, B_img_), 1))[:, [2, 1, 0], :, :]
            if self.nblend:
                fake_B = torch.where(self.mask_re_cuda == 0, B_img, fake_B)
            self.fuse_mask_cuda_copy = self.fuse_mask_cuda.repeat(this_batch, 1, 1, 1)
            fuse_res = fake_B * self.fuse_mask_cuda_copy + (1 - self.fuse_mask_cuda_copy) * B_img
        torch.cuda.synchronize()
        model_et = time.time()
        output_img_list = []
        for i in range(this_batch):
            mask_B_pre_list[i][int(0 * (self.img_size / 256)):int(-10 * (self.img_size / 256)),
                               int(5 * (self.img_size / 256)):int(-5 * (self.img_size / 256))] = \
                ((fuse_res[i] + 1) * 127.5).permute(1, 2, 0).byte().cpu().numpy()
            output_img_list.append(mask_B_pre_list[i])
        return output_img_list

    def inference1(self, audio_info, face_data_dict, this_batch, start_idx, params):
        B_img_list = []
        B_img__list = []
        mask_B_list = []
        mask_B_pre_list = []
        lab_list = []
        for i in range(this_batch):
            img_idx = start_idx + i
            mask_B_pre = face_data_dict[img_idx]['crop_img']
            mask_B = mask_B_pre[int(0 * (self.img_size / 256)):int(-10 * (self.img_size / 256)),
                                int(5 * (self.img_size / 256)):int(-5 * (self.img_size / 256))]
            B_img = mask_B.copy()
            mask_B = torch.from_numpy(mask_B[:, :, [2, 1, 0]].transpose(2, 0, 1))
            B_img_ = B_img.copy()
            B_img_ = torch.from_numpy(B_img_[:, :, [2, 1, 0]].transpose(2, 0, 1))
            B_img_list.append(torch.from_numpy(B_img.transpose(2, 0, 1)))
            B_img__list.append(B_img_)
            mask_B_pre_list.append(mask_B_pre)
            mask_B_list.append(mask_B)
            lab = audio_info[start_idx + i][np.newaxis, ...]
            lab_list.append(lab)
        model_st = time.time()
        torch.cuda.synchronize()
        if this_batch > 0:
            lab = torch.tensor(lab_list).cuda().half()
            mask_B = torch.stack(mask_B_list).cuda().half()
            mask_B = self.tensor_norm(mask_B, mask=self.mask_cuda.repeat(this_batch, 3, 1, 1))
            B_img_ = torch.stack(B_img__list).cuda().half()
            B_img_ = self.tensor_norm(B_img_, mask=self.mask_re_cuda.repeat(this_batch, 3, 1, 1))
            B_img = torch.stack(B_img_list).cuda().half()
            B_img = self.tensor_norm(B_img)
            fake_B = self.model.netG(lab, torch.cat((mask_B, B_img_), 1))[:, [2, 1, 0], :, :]
            if self.nblend:
                fake_B = torch.where(self.mask_re_cuda == 0, B_img, fake_B)
            self.fuse_mask_cuda_copy = self.fuse_mask_cuda.repeat(this_batch, 1, 1, 1)
            fuse_res = fake_B * self.fuse_mask_cuda_copy + (1 - self.fuse_mask_cuda_copy) * B_img
        torch.cuda.synchronize()
        model_et = time.time()
        output_img_list = []
        for i in range(this_batch):
            mask_B_pre_list[i][int(0 * (self.img_size / 256)):int(-10 * (self.img_size / 256)),
                               int(5 * (self.img_size / 256)):int(-5 * (self.img_size / 256))] = \
                ((fuse_res[i] + 1) * 127.5).permute(1, 2, 0).byte().cpu().numpy()
            output_img_list.append(mask_B_pre_list[i])
        return output_img_list

    def get_face_mask(self, img, landmarks):
        imgshape = img.shape[0]
        landmarks = landmarks.astype(int)
        wanted_numpy = np.concatenate([landmarks[2:15], landmarks[29:30]])
        mask = np.zeros((imgshape, imgshape), dtype=np.uint8)
        wanted_numpy = cv2.convexHull(wanted_numpy)
        cv2.fillConvexPoly(mask, wanted_numpy, 255)
        mid = (landmarks[5, :] + landmarks[11, :]) // 2
        cv2.ellipse(mask, (mid[0], mid[1]),
                    ((landmarks[11, 0] - landmarks[5, 0] + 3 * (imgshape // 266)) // 2,
                     60 * (imgshape // 266)),
                    0, 0, 180, (255, 255, 255), -1)
        amask = (mask > 0).astype(np.uint8) * 255
        kernel_size = (5 * (imgshape // 266) + 1, 5 * (imgshape // 266) + 1)
        iterations = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        eroded_mask = cv2.dilate(amask, kernel, iterations=iterations)
        return eroded_mask

    def get_face_mask2(self, img, landmarks, delta):
        wanted_numpy = landmarks.astype(int)
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        wanted_numpy = cv2.convexHull(wanted_numpy)
        cv2.fillConvexPoly(mask, wanted_numpy, 255)
        amask = (mask > 0).astype(np.uint8) * 255
        kernel_size = (5, 5)
        iterations = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        amask = cv2.dilate(amask, kernel, iterations=iterations)
        eroded_mask = cv2.GaussianBlur(amask, (5, 5), 0)
        return amask

    def inference_notraining(self, audio_info, face_data_dict, this_batch, start_idx,
                             blend_dynamic, params, frameId):
        model_st = time.time()
        B_img_list = []
        B_img__list = []
        mask_B_list = []
        mask_B_pre_list = []
        blend_mask_list = []
        lab_list = []
        for i in range(this_batch):
            img_idx = start_idx + i
            mask_B_pre = face_data_dict[img_idx]['crop_img']
            mask_B = mask_B_pre[int(5 * (self.img_size / 256)):int(-5 * (self.img_size / 256)),
                                int(5 * (self.img_size / 256)):int(-5 * (self.img_size / 256))]
            B_img = mask_B.copy()
            mask_B = torch.from_numpy(mask_B[:, :, [2, 1, 0]].transpose(2, 0, 1))
            B_img_ = B_img.copy()
            B_img_ = torch.from_numpy(B_img_[:, :, [2, 1, 0]].transpose(2, 0, 1))
            if blend_dynamic == "lmk":
                lm = face_data_dict[img_idx]['crop_lm']
                blend_mask = self.get_face_mask(mask_B_pre, lm)
                blend_mask = blend_mask[int(5 * (self.img_size / 256)):int(-5 * (self.img_size / 256)),
                                        int(5 * (self.img_size / 256)):int(-5 * (self.img_size / 256))]
                blend_mask = cv2.resize(blend_mask, (self.img_size, self.img_size))
                blend_mask_list.append(blend_mask)
            elif blend_dynamic == "xseg":
                xseg_mask_out = self.xseg.forward(B_img)
                xseg_mask_out = np.where(xseg_mask_out >= 0.1, 1, 0).astype(np.float32)
                blend_mask = xseg_mask_out
                blend_mask = cv2.resize(blend_mask, (self.img_size, self.img_size))
                blend_mask_list.append(blend_mask)
            B_img_list.append(torch.from_numpy(B_img[:, :, [2, 1, 0]].transpose(2, 0, 1)))
            B_img__list.append(B_img_)
            mask_B_pre_list.append(mask_B_pre)
            mask_B_list.append(mask_B)
            lab = audio_info[start_idx + i].transpose(1, 0)
            lab_list.append(lab)
        torch.cuda.synchronize()
        if this_batch > 0:
            lab = torch.tensor(lab_list).cuda()
            mask_B = torch.stack(mask_B_list).cuda()
            mask_B = self.tensor_norm_no_training(mask_B, mask=self.mask_cuda.repeat(this_batch, 3, 1, 1))
            B_img_ = torch.stack(B_img__list).cuda()
            B_img_ = self.tensor_norm_no_training(B_img_, mask=self.mask_re_cuda.repeat(this_batch, 3, 1, 1))
            B_img = torch.stack(B_img_list).cuda()
            B_img = self.tensor_norm_no_training(B_img)
            fake_B = self.model.netG(mask_B, B_img_, lab)
            if self.nblend:
                fake_B = torch.where(self.mask_re_cuda == 0, B_img, fake_B)
            if blend_dynamic in ('xseg', 'lmk'):
                weights = []
                for i in range(this_batch):
                    weight = cv2.stackBlur(blend_mask_list[i],
                                           (16 * int(self.img_size / 256) + 1,
                                            16 * int(self.img_size / 256) + 1), 0) / 255
                    weight = torch.tensor(weight).unsqueeze(0).cuda().repeat(3, 1, 1)
                    weights.append(weight)
                self.fuse_mask_cuda_copy = torch.stack(weights).cuda()
            else:
                self.fuse_mask_cuda_copy = self.fuse_mask_cuda.repeat(this_batch, 1, 1, 1)
            if blend_dynamic == "xseg":
                fuse_res = B_img * self.fuse_mask_cuda_copy + (1 - self.fuse_mask_cuda_copy) * fake_B
            else:
                fuse_res = fake_B * self.fuse_mask_cuda_copy + (1 - self.fuse_mask_cuda_copy) * B_img
        torch.cuda.synchronize()
        model_et = time.time()
        output_img_list = []
        for i in range(this_batch):
            mask_B_pre_list[i][int(5 * (self.img_size / 256)):int(-5 * (self.img_size / 256)),
                               int(5 * (self.img_size / 256)):int(-5 * (self.img_size / 256))] = \
                (fuse_res[i] * 255).permute(1, 2, 0).byte().cpu().numpy()[..., ::-1]
            output_img_list.append(mask_B_pre_list[i])
        return output_img_list

    def blendImages(self, src, dst, featherAmount=0.1):
        torch.cuda.synchronize()
        src_gpu = src
        dst_gpu = dst
        composed_gpu = self.w_gpu * src_gpu + (1 - self.w_gpu) * dst_gpu
        composedImg = composed_gpu
        return composedImg
