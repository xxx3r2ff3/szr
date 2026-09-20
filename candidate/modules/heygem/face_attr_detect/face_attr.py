# -*- coding: utf-8 -*-
# 出处:candidate/modules/heygem/face_attr_detect/face_attr.py ← face_attr_detect/face_attr.pyc
# 反汇编:evidence/modules/_heygem_orphans/disasm/face_attr_detect_face_attr.pyc.dis(py3.8)
# 模块 [Constants] 第 0 项为 0(不是字符串)⇒ 本模块**没有**模块级 docstring,
#   故此处只保留出处注释块。
# 结构/签名/常量按 .dis 对齐:
#   导入:import numpy as np / from cv2box import CVImage / from apstone import ModelBase
#   MODEL_ZOO = {'face_attr_mbnv3': {'model_path': ...onnx, 'input_dynamic_shape': (1, 3, 512, 512)}}
#   class FaceAttr(ModelBase):
#       __init__(self, model_name='face_attr_mbnv3', provider='gpu')   —— 3 参
#           (类体 MAKE_FUNCTION 9 = DEFAULTS|CLOSURE,defaults=('face_attr_mbnv3', 'gpu'))
#       forward(self, image_p_)                                        —— 2 参
#       show_label()                                                   —— @staticmethod,0 参
# 语义重建说明:三个函数体均由 .dis 控制流逐条还原。
#   forward 里 blob_innormal 的 CALL_FUNCTION_KW 3 + ('input_mean', 'input_std')
#   ⇒ 源码形态为 blob_innormal(512, input_mean=[...], input_std=[...]);
#   model.forward 的 CALL_FUNCTION_KW 2 + ('trans',) ⇒ forward(blob, trans=False)[0]。
# 未还原细节:浮点常量按 .dis 的 LOAD_CONST 显示值逐字保留(pycdas 只印 6 位有效数字;
#   原始 double 为 132.38155592 / 110.99284567 / 102.62942472 / 68.5106407 /
#   61.65929394 / 58.61700102)。此处按规范以 [Disassembly] 显示值为准。
import numpy as np
from cv2box import CVImage
from apstone import ModelBase

MODEL_ZOO = {
    'face_attr_mbnv3': {
        'model_path': './face_attr_detect/face_attr_epoch_12_220318.onnx',
        'input_dynamic_shape': (1, 3, 512, 512),
    },
}


class FaceAttr(ModelBase):

    def __init__(self, model_name='face_attr_mbnv3', provider='gpu'):
        super().__init__(MODEL_ZOO[model_name], provider)

    def forward(self, image_p_):
        blob = CVImage(image_p_).blob_innormal(
            512,
            input_mean=[132.382, 110.993, 102.629],
            input_std=[68.5106, 61.6593, 58.617])
        result = self.model.forward(blob, trans=False)[0]
        return np.around(result, 3)

    @staticmethod
    def show_label():
        print('female male front side clean occlusion super_hq hq blur nonhuman')
