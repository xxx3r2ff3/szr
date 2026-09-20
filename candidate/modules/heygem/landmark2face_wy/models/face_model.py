# -*- coding: utf-8 -*-
# face_model —— S004 逆向恢复(landmark2face_wy/models/face_model.pyc,py3.8;
# 1229 行反汇编)。PiRender 谱系:FaceGenerator/PirenderGenerator/MappingNet/
# WarpingNet/EditingNet(基类均为 nn.Module,按类体 LOAD_NAME nn + LOAD_ATTR Module 判定)。
# 事实源:evidence/modules/_heygem_orphans/disasm/landmark2face_wy_models_face_model.pyc.dis
# 还原说明(全部函数体按 [Disassembly] 逐指令重建,未采用 pycdc 的近似输出):
#   1) __init__ 使用显式 super(类名, self).__init__()
#      (字节码为 LOAD_GLOBAL super + LOAD_GLOBAL 类名 + LOAD_FAST self + CALL_FUNCTION 2);
#   2) 字典展开按字节码还原:MappingNet(**mapping_net) 为 BUILD_TUPLE 0 + CALL_FUNCTION_EX 1;
#      WarpingNet(**warpping_net, **common) / EditingNet(**editing_net, **common) 为
#      BUILD_MAP_UNPACK_WITH_CALL 2 + CALL_FUNCTION_EX 1;
#   3) MappingNet.forward 的残差项是 out[:, :, 3:-3](BUILD_SLICE 3 / BUILD_SLICE -3);
#   4) FaceGenerator.forward 的最后一个参数 stage 有默认值 None
#      (MAKE_FUNCTION 1 + 默认值元组 (None,));其余方法均为 MAKE_FUNCTION 0/8,无默认值;
#   5) 导入语句按 .dis 逐字还原:`from landmark2face_wy.util import flow_util`
#      (level 0 + IMPORT_NAME landmark2face_wy.util + IMPORT_FROM flow_util);
#      该交叉引用已在 candidate 树落地:landmark2face_wy/util/__init__.py 与
#      landmark2face_wy/util/flow_util.py 均存在(缺少 __init__.py 时静态检查会报“缺内部 import”);
#   6) 模块 docstring:.dis 模块 [Constants] 首项为 0(非字符串),原码无模块 docstring,
#      此处按规范以 # 注释记录出处;
#   7) 未还原细节:无(7 个 code object 的常量表/名字表/分支顺序均已逐条落位)。
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from landmark2face_wy.util import flow_util
from .base_function import LayerNorm2d, ADAINHourglass, FineEncoder, FineDecoder


class FaceGenerator(nn.Module):

    def __init__(self, mapping_net, warpping_net, editing_net, common):
        super(FaceGenerator, self).__init__()
        self.mapping_net = MappingNet(**mapping_net)
        self.warpping_net = WarpingNet(**warpping_net, **common)
        self.editing_net = EditingNet(**editing_net, **common)

    def forward(self, input_image, driving_source, stage=None):
        if stage == 'warp':
            descriptor = self.mapping_net(driving_source)
            output = self.warpping_net(input_image, descriptor)
        else:
            descriptor = self.mapping_net(driving_source)
            output = self.warpping_net(input_image, descriptor)
            output['fake_image'] = self.editing_net(input_image, output['warp_image'], descriptor)
        return output


class PirenderGenerator(nn.Module):

    def __init__(self, mapping_net, editing_net, common):
        super(PirenderGenerator, self).__init__()
        self.mapping_net = MappingNet(**mapping_net)
        self.editing_net = EditingNet(**editing_net, **common)

    def forward(self, ref_img, mask_img, driving_source):
        descriptor = self.mapping_net(driving_source)
        return self.editing_net(ref_img, mask_img, descriptor)


class MappingNet(nn.Module):

    def __init__(self, coeff_nc, descriptor_nc, layer):
        super(MappingNet, self).__init__()
        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)
        self.first = nn.Sequential(
            torch.nn.Conv1d(coeff_nc, descriptor_nc, kernel_size=7, padding=0, bias=True))
        for i in range(layer):
            net = nn.Sequential(
                nonlinearity,
                torch.nn.Conv1d(descriptor_nc, descriptor_nc, kernel_size=3, padding=0, dilation=3))
            setattr(self, 'encoder' + str(i), net)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = descriptor_nc

    def forward(self, input_3dmm):
        out = self.first(input_3dmm)
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) + out[:, :, 3:-3]
        out = self.pooling(out)
        return out


class WarpingNet(nn.Module):

    def __init__(self, image_nc, descriptor_nc, base_nc, max_nc, encoder_layer, decoder_layer,
                 use_spect):
        super(WarpingNet, self).__init__()
        nonlinearity = nn.LeakyReLU(0.1)
        norm_layer = functools.partial(LayerNorm2d, affine=True)
        kwargs = {'nonlinearity': nonlinearity, 'use_spect': use_spect}
        self.descriptor_nc = descriptor_nc
        self.hourglass = ADAINHourglass(image_nc, self.descriptor_nc, base_nc, max_nc, encoder_layer,
                                        decoder_layer, **kwargs)
        self.flow_out = nn.Sequential(
            norm_layer(self.hourglass.output_nc), nonlinearity,
            nn.Conv2d(self.hourglass.output_nc, 2, kernel_size=7, stride=1, padding=3))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input_image, descriptor):
        final_output = {}
        output = self.hourglass(input_image, descriptor)
        final_output['flow_field'] = self.flow_out(output)
        deformation = flow_util.convert_flow_to_deformation(final_output['flow_field'])
        final_output['warp_image'] = flow_util.warp_image(input_image, deformation)
        return final_output


class EditingNet(nn.Module):

    def __init__(self, image_nc, descriptor_nc, layer, base_nc, max_nc, num_res_blocks, use_spect):
        super(EditingNet, self).__init__()
        nonlinearity = nn.LeakyReLU(0.1)
        norm_layer = functools.partial(LayerNorm2d, affine=True)
        kwargs = {'norm_layer': norm_layer, 'nonlinearity': nonlinearity, 'use_spect': use_spect}
        self.descriptor_nc = descriptor_nc
        self.encoder = FineEncoder(image_nc * 2, base_nc, max_nc, layer, **kwargs)
        self.decoder = FineDecoder(image_nc, self.descriptor_nc, base_nc, max_nc, layer,
                                   num_res_blocks, **kwargs)

    def forward(self, input_image, warp_image, descriptor):
        x = torch.cat([input_image, warp_image], 1)
        x = self.encoder(x)
        gen_image = self.decoder(x, descriptor)
        return gen_image


if __name__ == '__main__':
    g = PirenderGenerator({'coeff_nc': 64, 'descriptor_nc': 256, 'layer': 3},
                          {'layer': 3, 'num_res_blocks': 2, 'base_nc': 64},
                          {'image_nc': 3, 'descriptor_nc': 256, 'max_nc': 256, 'use_spect': False})
    semantic_data = torch.randn(2, 64, 27)
    img = torch.randn(2, 3, 512, 512)
    img1 = torch.randn(2, 3, 512, 512)
    output = g(img, img1, semantic_data)
    print(output.size())
