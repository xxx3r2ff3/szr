# -*- coding: utf-8 -*-
# S003/S004 恢复件:heygem/face_detect_utils/face_detect.pyc(Python 3.8,magic 3413)
# 源 pyc  : /Volumes/A/数字人/szr2026/modules/heygem/face_detect_utils/face_detect.pyc
#           (源文件长度 5003 字节;pycdas 反汇编 1148 行)
# 证据链  : evidence/modules/_heygem_orphans/disasm/face_detect_utils_face_detect.pyc.dis
# 谱系    : heygem 自有工具模块(SCRFD 人脸检测 + pfpld 98 点关键点精化);
#           同目录 scrfd.pyc(insightface SCRFD)为其直接依赖,反汇编见
#           modules_heygem_face_detect_utils_scrfd.pyc.dis。
# 还原度  : 结构/常量/控制流逐指令对齐(函数清单 Object Name/Arg Count 与 [Constants]/
#           [Names]/[Disassembly] 三方核对):FaceDetect.__init__(4)、FaceDetect.get_bboxes(4)、
#           pfpld.__init__(3)、pfpld.forward(2)、convert98to68(1) 全部落齐。
# 原码缺陷(忠实保留,未修正):
#   1) pfpld.__init__ 的 `raise e('load onnx failed')`:字节码为
#      LOAD_FAST e;LOAD_CONST 'load onnx failed';CALL_FUNCTION 1;RAISE_VARARGS 1,
#      即原码对异常实例做调用后抛出(原码如此,非恢复讹误)。
#   2) convert98to68 首个循环下标写作 points[j * 2 * 2 + 0](字节码两次 BINARY_MULTIPLY),
#      其余循环为 points[j * 2 + 0]。

import numpy as np
import cv2

from scrfd import SCRFD

import onnxruntime as ort


class FaceDetect:

    def __init__(self, mode='scrfd_500m', cpu=False, model_path='./resources/'):
        if 'scrfd' in mode:
            if mode == 'scrfd_500m':
                scrfd_model_path = model_path + 'scrfd_500m_bnkps_shape640x640.onnx'
            elif mode == 'scrfd_10g':
                scrfd_model_path = model_path + 'scrfd_10g_bnkps.onnx'
        self.det_model = SCRFD(scrfd_model_path, cpu=cpu)
        self.det_model.prepare(ctx_id=0, input_size=(640, 640))

    def get_bboxes(self, image, thresh=0.5, max_num=0):
        if type(image) == str:
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        elif type(image) == np.ndarray:
            pass
        bboxes_, kpss_ = self.det_model.detect(image, thresh=thresh, max_num=max_num, metric='max')
        return bboxes_, kpss_


class pfpld:

    def __init__(self, cpu=False, model_path='./resources'):
        onnx_path = f'{model_path}/pfpld_robust_sim_bs1_8003.onnx'
        try:
            self.ort_session = ort.InferenceSession(
                onnx_path,
                providers=['CPUExecutionProvider'] if cpu else ['CUDAExecutionProvider'])
        except Exception as e:
            raise e('load onnx failed')
        self.input_name = self.ort_session.get_inputs()[0].name

    def forward(self, input):
        size = input.shape
        ort_inputs = {
            self.input_name:
                (cv2.resize(input, (112, 112)) / 255).astype(np.float32).transpose(2, 0, 1)[None]
        }
        pred = self.ort_session.run(None, ort_inputs)
        pred = convert98to68(pred[1])
        return pred.reshape(-1, 68, 2) * size[:2][::-1]


def convert98to68(list_info):
    points = list_info[0, 0:196]
    info_68 = []
    for j in range(17):
        x = points[j * 2 * 2 + 0]
        y = points[j * 2 * 2 + 1]
        info_68.append(x)
        info_68.append(y)
    for j in range(33, 38):
        x = points[j * 2 + 0]
        y = points[j * 2 + 1]
        info_68.append(x)
        info_68.append(y)
    for j in range(42, 47):
        x = points[j * 2 + 0]
        y = points[j * 2 + 1]
        info_68.append(x)
        info_68.append(y)
    for j in range(51, 61):
        x = points[j * 2 + 0]
        y = points[j * 2 + 1]
        info_68.append(x)
        info_68.append(y)
    point_38_x = (float(points[120]) + float(points[124])) / 2
    point_38_y = (float(points[121]) + float(points[125])) / 2
    point_39_x = (float(points[124]) + float(points[128])) / 2
    point_39_y = (float(points[125]) + float(points[129])) / 2
    point_41_x = (float(points[128]) + float(points[132])) / 2
    point_41_y = (float(points[129]) + float(points[133])) / 2
    point_42_x = (float(points[120]) + float(points[132])) / 2
    point_42_y = (float(points[121]) + float(points[133])) / 2
    point_44_x = (float(points[136]) + float(points[140])) / 2
    point_44_y = (float(points[137]) + float(points[141])) / 2
    point_45_x = (float(points[140]) + float(points[144])) / 2
    point_45_y = (float(points[141]) + float(points[145])) / 2
    point_47_x = (float(points[144]) + float(points[148])) / 2
    point_47_y = (float(points[145]) + float(points[149])) / 2
    point_48_x = (float(points[136]) + float(points[148])) / 2
    point_48_y = (float(points[137]) + float(points[149])) / 2
    info_68.append(point_38_x)
    info_68.append(point_38_y)
    info_68.append(point_39_x)
    info_68.append(point_39_y)
    info_68.append(points[128])
    info_68.append(points[129])
    info_68.append(point_41_x)
    info_68.append(point_41_y)
    info_68.append(point_42_x)
    info_68.append(point_42_y)
    info_68.append(points[136])
    info_68.append(points[137])
    info_68.append(point_44_x)
    info_68.append(point_44_y)
    info_68.append(point_45_x)
    info_68.append(point_45_y)
    info_68.append(points[144])
    info_68.append(points[145])
    info_68.append(point_47_x)
    info_68.append(point_47_y)
    info_68.append(point_48_x)
    info_68.append(point_48_y)
    for j in range(76, 96):
        x = points[j * 2 + 0]
        y = points[j * 2 + 1]
        info_68.append(x)
        info_68.append(y)
    for j in range(len(list_info[196:])):
        info_68.append(list_info[196 + j])
    return np.array(info_68)
