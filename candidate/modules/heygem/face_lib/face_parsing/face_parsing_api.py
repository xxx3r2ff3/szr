# -*- coding: utf-8 -*-
# @Time : 2021/11/25
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

"""
https://github.com/zllrunning/face-parsing.PyTorch
"""

# ---- R072 `heygem__face_lib__face_parsing__face_parsing_api` 候选实现说明 ----
# 门控 T3 上游恢复件;源 pyd: modules/heygem/face_lib/face_parsing/face_parsing_api.pyd
#   sha256 = 01fd5feb722fd31f5b85de105010d096208a84f956262ce4e6f1b6b940351934
# 上游锚定: ykk648/AI_power @ a895b6f1c191 (2023-07-19)
#   face_lib/face_parsing/face_parsing_api.py
#   sha256(upstream) = 03904681285b078fbbe8f064060496b9a34e42ca39a586acf9a861b91138c6fc
#   许可证 = GPL-3.0(仓库 LICENSE);同源副本 ykk648/face_power@92f68c845738
# 本地谱系证据: modules/heygem/model_lib/model_base.py(pyd 实际导入的 model_lib 包,
#   文件头 "# @Time : 2022/7/29");site-packages 内 apstone-0.0.8(上游后续改名)、
#   cv2box-0.5.9(GPL-3.0,作者 ykk648)。
# token 级一致性: 与上游逐 token 相同,唯二差异 = 本注释块 + 导入包名(见 static/token_diff.md)
# 差异(详见 reports/modules/heygem__face_lib__face_parsing__face_parsing_api-impl.md §5):
#   1) 上游导入行 `from apstone import ModelBase`;pyd 串池只有 model_lib(无 apstone)
#      → 本候选改用 `from model_lib import ModelBase`(与冻结树 modules/heygem/model_lib 一致)。
#   2) pyd 的 get_face_mask 代码区存在一处 `cv2.ellipse` 调用(串池 ellipse @0x31838,
#      调用点 RVA 0x8640),上游该修订无此调用,参数无法从静态材料恢复 → 未臆造,
#      已登记为未解决差异项。
#
# 证据出处(逐项可回溯):
#   [ST 0x..] evidence/modules/heygem__face_lib__face_parsing__face_parsing_api/static/
#            pe_scan.md §Cython 字面量池(87 条,由 .rdata __pyx_k_* 池精确提取)与
#            pyx_string_table.json(84 条,栈上 __Pyx_StringTabEntry 序列精确还原);
#            覆盖核对见同目录 pool_coverage.md。
#   [MP]     同目录 pe_scan.md §PyMethodDef —— FaceParsing.__init__ / forward /
#            get_face_mask / show 四条(均 METH_FASTCALL|METH_KEYWORDS = 0x82);
#            forward 的 ml_doc 与上游 docstring 逐字节一致。
#   [PYLINE] 同目录 line_markers.md —— forward(48-52)/show(78-90) 错误出口 py 行号
#            与上游完全一致;get_face_mask 段见差异项 2)。
#   [GATE]   evidence/imports/gated_dll_dependencies.json + probe_gated_dll.py ——
#            pyd 导入表为 python38.dll,3.10 运行时不可装载(无 oracle 黄金)。
import numpy as np                                            # [ST 0x317a4 'numpy']
import cv2                                                    # [ST 0x316fc 'cv2']
from cv2box import CVImage                                    # [ST 0x317f4/0x31830]
from model_lib import ModelBase                               # [ST 0x318d8/0x318f8]

MODEL_ZOO = {                                                 # [ST 0x318c8]
    # input_name: ['x'], shape: [[1, 3, 512, 512]]
    # output_name: ['feat_out'], shape: [[1, 19, 512, 512]]
    'face_parse_onnx': {                                      # [ST 0x31a88]
        'model_path': 'pretrain_models/face_lib/face_parsing/79999_iter.onnx'
    },                                                        # [ST 0x31bd8 逐字节一致]
    # no more support for tjm model
    # 'face_parse_tjm': {
    #     'model_path': 'pretrain_models/face_lib/face_parsing/79999_iter.tjm'
    # },
}

# Colors for all 20 parts
PART_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 0, 85], [255, 0, 170],
               [0, 255, 0], [85, 255, 0], [170, 255, 0], [0, 255, 85], [0, 255, 170],
               [0, 0, 255], [85, 0, 255], [170, 0, 255], [0, 85, 255], [0, 170, 255],
               [255, 255, 0], [255, 255, 85], [255, 255, 170], [255, 0, 255], [255, 85, 255],
               [255, 170, 255], [0, 255, 255], [85, 255, 255], [170, 255, 255]]
# Face Mask
MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]


class FaceParsing(ModelBase):                                 # [ST 0x31998 FaceParsing]
    def __init__(self, model_name='face_parse_onnx', provider='gpu'):
        # [ST 0x31978 'model_name' / 0x31888 'provider' / 0x31968 'model_info' / 0x31a88]
        super().__init__(model_info=MODEL_ZOO[model_name], provider=provider)
        self.parsing_results = None                           # [ST 0x31a98]
        self.face_image = None                                # [ST 0x31928]
        self.input_size = 512                                 # [ST 0x31948]
        self.input_mean = (0.485, 0.456, 0.406)               # [ST 0x31938]
        self.input_std = (0.229, 0.224, 0.225)                # [ST 0x318e8]

    def forward(self, face_image):
        """
        Args:
            face_image: cv2 0-255 (3,h,w)
        Returns: (512,512)
        """
        # [MP] 本 docstring 与 pyd ml_doc 逐字节一致;py 行号 48-52 亦一致
        self.face_image = face_image
        face_image_in = CVImage(self.face_image).blob_innormal(self.input_size, self.input_mean, self.input_std,
                                                               rgb=False)
        # [ST 0x31a50 'face_image_in' / 0x31a28 'blob_innormal' / 0x31710 'rgb']
        self.parsing_results = self.model.forward(face_image_in)[0].squeeze(0).argmax(0)
        # [ST 0x3179c 'model' / 0x31848 'forward' / 0x31870 'squeeze' / 0x317e4 'argmax']
        return self.parsing_results

    def get_face_mask(self, mask_shape):
        # [ST 0x31958 'mask_shape' / 0x317dc 'zeros' / 0x31840 'float32' / 0x31a18 MASK_COLORMAP]
        mask = np.zeros((512, 512)).astype(np.float32)
        for idx, color in enumerate(MASK_COLORMAP):
            mask[self.parsing_results == idx] = color
        mask = cv2.stackBlur(mask, (201, 201))                # [ST 0x31918 'stackBlur']

        # remove the black borders
        thres = 10                                            # [ST 0x317c4 'thres']
        mask[:thres, :] = 0
        mask[-thres:, :] = 0
        mask[:, :thres] = 0
        mask[:, -thres:] = 0
        mask = mask / 255.
        mask = cv2.resize(mask, mask_shape)                   # [ST 0x3181c 'resize']
        return mask[..., np.newaxis]                          # [ST 0x31850 'newaxis']

    def show(self):
        vis_im = CVImage(self.face_image).bgr.copy().astype(np.uint8)
        # [ST 0x31824 'vis_im' / 0x316f8 'bgr' / 0x31714 'copy' / 0x317ec 'astype' / 0x317cc 'uint8']
        vis_parsing_anno = self.parsing_results.copy().astype(np.uint8)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
        # [ST 0x31ad8 'vis_parsing_anno' / 0x31b50 'vis_parsing_anno_color' / 0x317b4 'shape']

        num_of_class = np.max(vis_parsing_anno)               # [ST 0x319f8 'num_of_class' / 0x3170c 'max']
        for pi in range(1, num_of_class + 1):                 # [ST 0x317ac 'range' / 0x316ec 'pi']
            index = np.where(vis_parsing_anno == pi)          # [ST 0x31794 'index' / 0x317d4 'where']
            vis_parsing_anno_color[index[0], index[1], :] = PART_COLORS[pi]
            # [ST 0x319a8 'PART_COLORS']

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
        # [ST 0x319b8 'addWeighted' / 0x31878 'cvtColor' / 0x31a08 'COLOR_RGB2BGR']

        CVImage(vis_im).show()                                # [ST 0x3176c 'show']


if __name__ == "__main__":
    test_img = 'resource/cropped_face/512.jpg'                # [ST 0x318b8 / 0x31b88 逐字节一致]
    fp = FaceParsing(model_name='face_parse_onnx', provider='gpu')

    parsing = fp.forward(test_img)
    # mask = fp.get_face_mask((512, 512))
    fp.show()
