# -*- coding: utf-8 -*-
# GFPGAN ONNX 推理封装 —— S004 逐指令重写
# (face_lib/face_restore/gfpgan_onnx/gfpgan_onnx_api.pyc,py3.8;
#  反汇编 evidence/modules/_heygem_orphans/disasm/
#  face_lib_face_restore_gfpgan_onnx_gfpgan_onnx_api.pyc.dis,318 行)
#
# 结构与出处(全部照 [Disassembly]/[Constants] 落位):
#   - 模块导入:`from cv2box import CVImage, MyFpsCounter`;`from model_lib import ModelBase`
#     (IMPORT_NAME cv2box/model_lib,fromlist 分别为 ('CVImage','MyFpsCounter')/('ModelBase',));
#     模块 Constants 首项为 0(import level),故 .dis 无模块 docstring。
#   - MODEL_ZOO = {'GFPGANv1.4': {'model_path':
#     'pretrain_models/face_lib/face_restore/gfpgan/GFPGANv1.4.onnx'}}(BUILD_MAP 嵌套单键)。
#   - class GFPGAN(ModelBase):类体带 __classcell__(零参 super),默认值元组 ('GFPGANv1.4','gpu')。
#   - __init__:super().__init__(MODEL_ZOO[model_type], provider) →
#     self.model_type = model_type;self.input_std = self.input_mean = 127.5;
#     self.input_size = (512, 512)(DUP_TOP 双 STORE_ATTR 的同值链)。
#   - forward:docstring 逐字;CVImage(face_image).blob(input_size, input_mean, input_std, rgb=True)
#     → self.model.forward(image_in) → 后处理 ((image_out[0][0] + 1) / 2)[::-1]
#     .transpose(1, 2, 0).clip(0, 1)(BINARY_ADD → TRUE_DIVIDE → BUILD_SLICE(None,None,-1)
#     → transpose(1,2,0) → clip(0,1),偏移量见 .dis 40..90)。
#   - __main__ 块:face_img_p / GFPGAN(model_type='GFPGANv1.4', provider='gpu') /
#     with MyFpsCounter() as mfc: for i in range(10): face = fa.forward(face_img_p) /
#     CVImage(face, image_format='cv2').show()。
# 置信度:高(模块体与两个方法逐指令对齐,无未还原分支)。
# 依赖缺口(不在本批可改范围,仅登记):`from model_lib import ModelBase` 要求 model_lib 包导出
# ModelBase;候选树 candidate/modules/heygem/model_lib/__init__.py 为空,
# model_lib/base_wrapper/onnx_model.py 只有 ONNXModel,无 ModelBase ⇒ 该内部 import 目前不可解析。
from cv2box import CVImage, MyFpsCounter
from model_lib import ModelBase

MODEL_ZOO = {'GFPGANv1.4': {'model_path': 'pretrain_models/face_lib/face_restore/gfpgan/GFPGANv1.4.onnx'}}


class GFPGAN(ModelBase):

    def __init__(self, model_type='GFPGANv1.4', provider='gpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type
        self.input_std = self.input_mean = 127.5
        self.input_size = (512, 512)

    def forward(self, face_image):
        """
        Args:
            face_image: cv2 image 0-255 BGR
        Returns:
            BGR 512x512x3 0-1
        """
        image_in = CVImage(face_image).blob(self.input_size, self.input_mean,
                                            self.input_std, rgb=True)
        image_out = self.model.forward(image_in)
        output_face = ((image_out[0][0] + 1) / 2)[::-1].transpose(1, 2, 0).clip(0, 1)
        return output_face


if __name__ == '__main__':
    face_img_p = 'resource/cropped_face/512.jpg'
    fa = GFPGAN(model_type='GFPGANv1.4', provider='gpu')
    with MyFpsCounter() as mfc:
        for i in range(10):
            face = fa.forward(face_img_p)
    CVImage(face, image_format='cv2').show()
