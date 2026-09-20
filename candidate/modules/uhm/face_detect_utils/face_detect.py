# -*- coding: utf-8 -*-
"""uhm.face_detect_utils.face_detect —— T4R 私有语义重建(R025)。

证据(oracle 侧只读实测):
  * P1 probe_surface_out.json —— dir()=['FaceDetect','SCRFD','convert98to68','cv2','np','ort',
    'pfpld'];SCRFD 的 code object co_filename='scrfd.py'、__module__='scrfd'
    (与本包 scrfd 同源,此处直接复用候选 scrfd.py 的 SCRFD)。
  * P5 probe_behavior4_out.json / P6 probe_behavior5_out.json —— convert98to68 输入形状
    与 68 点索引映射(逐点解码,见下)。
  * P4 probe_behavior3_out.json —— FaceDetect(mode='scrfd_500m', cpu=True, model_path=
    './resources/') 构造成功且仅含 det_model;mode='bogus' 无属性;'scrfd_10g' 缺文件 →
    AssertionError;get_bboxes 空图 → ((0,5) float32, (0,5,2) float32);pfpld.forward →
    (1,68,2) float64。
  * ST constants.txt —— 'scrfd_500m_bnkps_shape640x640.onnx'、'scrfd_10g_bnkps.onnx'、
    './resources/'、'./resources'、'/pfpld_robust_sim_bs1_8003.onnx'、'load onnx failed: '。
"""
import cv2
import numpy as np
import onnxruntime as ort

from modules.uhm.face_detect_utils.scrfd import SCRFD


def convert98to68(list_info):
    # P1 co_varnames=['list_info','points','info_68','j','x','y',
    #                 'point_38_x','point_38_y', … 'point_48_x','point_48_y']
    # 输入形态(P5/P6/P9):points = list_info[0, :](第二维 ≥196;仅第 0 行参与,
    #   shape_2x196_shift 与 shape_1x196 同值;1-D/2 行太短/3-D/list 均报错,
    #   报错行号 122/130/151/152/167 与分段一致);2 行输入只取第 0 行。
    # 线性反解(P11 基向量 196 次调用,linearity_max_err=0.0):该函数是 196→136 的
    #   **精确线性映射**,系数矩阵逐行给出:
    #   out 0..16  ← src 点 0,2,…,32(扁平 j*4 / j*4+1)          [17 点,直取]
    #   out 17..21 ← src 点 33..37                              [5 点,直取]
    #   out 22..26 ← src 点 42..46                              [5 点,直取]
    #   out 27..36 ← src 点 51..60                              [10 点,直取]
    #   out 37     ← mid(60,62)   out 38 ← mid(62,64)   out 39 ← 64
    #   out 40     ← mid(64,66)   out 41 ← mid(60,66)   out 42 ← 68
    #   out 43     ← mid(68,70)   out 44 ← mid(70,72)   out 45 ← 72
    #   out 46     ← mid(72,74)   out 47 ← mid(68,74)   out 48 ← 76
    #     (mid(a,b) = (x_a+x_b)/2,系数 0.5/0.5,即 P11 输出的 16 个非直取项)
    #   out 49..67 ← src 点 77..95                              [19 点,直取]
    # 输出:P8 实测恒为 np.ndarray(dtype=float64,shape=(136,))——即使输入 int32/i64
    #   也是 float64 → 末尾 np.array(info_68, dtype=np.float64)。
    # W6-4 probe_w64j8:中点在 float64 域求解(f32 输入按位提升,否则中点差 1 ULP)
    points = np.asarray(list_info[0, :], dtype=np.float64)
    info_68 = []
    j = 0
    x = 0
    y = 0
    for j in range(17):
        x = points[j * 4]
        y = points[j * 4 + 1]
        info_68.append(x)
        info_68.append(y)
    for j in range(33, 38):
        x = points[j * 2]
        y = points[j * 2 + 1]
        info_68.append(x)
        info_68.append(y)
    for j in range(42, 47):
        x = points[j * 2]
        y = points[j * 2 + 1]
        info_68.append(x)
        info_68.append(y)
    for j in range(51, 61):
        x = points[j * 2]
        y = points[j * 2 + 1]
        info_68.append(x)
        info_68.append(y)
    # 右眼区(60..66)/左眼区(68..74)补点:共 11 点(中点为算术平均)
    info_68.append((points[120] + points[124]) / 2)
    info_68.append((points[121] + points[125]) / 2)
    point_38_x = (points[124] + points[128]) / 2
    point_38_y = (points[125] + points[129]) / 2
    info_68.append(point_38_x)
    info_68.append(point_38_y)
    point_39_x = points[128]
    point_39_y = points[129]
    info_68.append(point_39_x)
    info_68.append(point_39_y)
    info_68.append((points[128] + points[132]) / 2)
    info_68.append((points[129] + points[133]) / 2)
    point_41_x = (points[120] + points[132]) / 2
    point_41_y = (points[121] + points[133]) / 2
    info_68.append(point_41_x)
    info_68.append(point_41_y)
    point_42_x = points[136]
    point_42_y = points[137]
    info_68.append(point_42_x)
    info_68.append(point_42_y)
    info_68.append((points[136] + points[140]) / 2)
    info_68.append((points[137] + points[141]) / 2)
    point_44_x = (points[140] + points[144]) / 2
    point_44_y = (points[141] + points[145]) / 2
    info_68.append(point_44_x)
    info_68.append(point_44_y)
    point_45_x = points[144]
    point_45_y = points[145]
    info_68.append(point_45_x)
    info_68.append(point_45_y)
    info_68.append((points[144] + points[148]) / 2)
    info_68.append((points[145] + points[149]) / 2)
    point_47_x = (points[136] + points[148]) / 2
    point_47_y = (points[137] + points[149]) / 2
    info_68.append(point_47_x)
    info_68.append(point_47_y)
    point_48_x = points[152]
    point_48_y = points[153]
    info_68.append(point_48_x)
    info_68.append(point_48_y)
    for j in range(77, 96):
        x = points[j * 2]
        y = points[j * 2 + 1]
        info_68.append(x)
        info_68.append(y)
    return np.array(info_68, dtype=np.float64)


class FaceDetect:
    def __init__(self, mode='scrfd_500m', cpu=False, model_path='./resources/'):
        # P1 co_varnames=['self','mode','cpu','model_path','scrfd_model_path'];
        # P4 实测:mode='scrfd_500m' + './resources/' 生效 → 仅 det_model 属性;
        # mode='bogus' → 无任何属性(无 else 分支);mode='scrfd_10g' 缺文件 →
        # AssertionError(无消息)→ assert os.path.exists(scrfd_model_path)。
        import os
        if mode == 'scrfd_500m':
            scrfd_model_path = model_path + 'scrfd_500m_bnkps_shape640x640.onnx'
            assert os.path.exists(scrfd_model_path)
            self.det_model = SCRFD(scrfd_model_path)
            # W6-4 probe_w64e/j:get_bboxes 构造路径 stdout 含 det_size 警告 →
            # FaceDetect 构造后调用 prepare(input_size=(640,640))(ctx 不可观测)
            self.det_model.prepare(-1, input_size=(640, 640))
        elif mode == 'scrfd_10g':
            scrfd_model_path = model_path + 'scrfd_10g_bnkps.onnx'
            assert os.path.exists(scrfd_model_path)
            self.det_model = SCRFD(scrfd_model_path)
            self.det_model.prepare(-1, input_size=(640, 640))

    def get_bboxes(self, image, thresh=0.5, max_num=0):
        # P1 co_varnames=['self','image','thresh','max_num','bboxes_','kpss_'];
        # P4 实测空图/白块图均 → ((0,5) float32, (0,5,2) float32)。
        bboxes_, kpss_ = self.det_model.detect(image, thresh=thresh, max_num=max_num)
        return bboxes_, kpss_


class pfpld:
    def __init__(self, cpu=False, model_path='./resources'):
        # P1 co_varnames=['self','cpu','model_path','onnx_path','cache_dir','providers','e'];
        # P4 实测:model_path='./resources' → 报错串内路径 './resources/pfpld_robust_sim_bs1_8003.onnx'
        # (字符串拼接,非 os.path.join);失败抛 Exception('load onnx failed: ' + str(e))。
        import os
        onnx_path = model_path + '/pfpld_robust_sim_bs1_8003.onnx'
        # W6-4 probe_w64b/j:v1 op 构造路径 stdout 内嵌 pfpld providers 字典,
        # cache_dir 为相对字面量 '.trtcache'、键序 fp16→path→enable(逐字证据)
        cache_dir = '.trtcache'
        if cpu:
            providers = ['CPUExecutionProvider']
        else:
            providers = [
                ('TensorrtExecutionProvider', {
                    'trt_fp16_enable': True,
                    'trt_engine_cache_path': cache_dir,
                    'trt_engine_cache_enable': True,
                }),
                'CUDAExecutionProvider',
            ]
        try:
            self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception as e:
            raise Exception("load onnx failed: " + str(e))

    def forward(self, input_image):
        # P1 co_varnames=['self','input_image','size','img_resized','img_tensor',
        #                 'ort_inputs','pred']
        # W6-4 probe_w64i2 位级破解(bitwise_equal=true):取模型第 2 输出 landms
        # (1,196) → convert98to68 → (1,68,2) float64 → x×w、y×h(归一化坐标放大回
        # 原图尺寸);输入 resize 112x112 + /255 + HWC→CHW。
        size = input_image.shape[:2]
        h, w = size
        img_resized = cv2.resize(input_image, (112, 112))
        img_tensor = img_resized.astype(np.float32) / 255.0
        img_tensor = img_tensor.transpose(2, 0, 1)[np.newaxis, :, :, :]
        ort_inputs = {self.ort_session.get_inputs()[0].name: img_tensor}
        pred = self.ort_session.run(None, ort_inputs)[1].reshape(-1, 196)
        pred = convert98to68(pred).reshape((1, 68, 2))
        pred[:, :, 0] *= w
        pred[:, :, 1] *= h
        return pred
