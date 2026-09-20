# -*- coding: utf-8 -*-
"""uhm.preprocess_v2 —— T4R 私有语义重建(R024)。

证据(oracle 侧只读实测):P1 probe_surface_out.json(首次导入失败链)、
P2/P3 probe_behavior{,2}_out.json、P4 probe_behavior3_out.json、
P5 probe_behavior4_out.json、ST constants.txt。

导入期副作用(P1/P4 实测):模块级 `YOLO('checkpoints/yolov8n-face.pt')` 以 **相对
cwd 路径**加载权重;cwd 无该文件时 FileNotFoundError(错误串中的反斜杠来自
ultralytics 的 Path 归一化)。因此任何用例必须在沙箱内提供 checkpoints/yolov8n-face.pt。
"""
import os
import time
import warnings

import cv2
import dlib
import numpy as np
import torch
from ultralytics import YOLO

warnings.filterwarnings('ignore')

now_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(now_dir))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
face_det = YOLO('checkpoints/yolov8n-face.pt')


def fast_bbox_expansion(x1, y1, x2, y2, w, h, expand_ratio=0.2):
    # P1 co_varnames 与 preprocess.fast_bbox_expansion 完全一致(同一实现);
    # P7 双侧随机对拍(int/float 类型逐位一致)确认与 preprocess 同式。
    width = x2 - x1
    height = y2 - y1
    expansion_x = width * expand_ratio
    expansion_y = height * expand_ratio
    new_x1 = max(0, int(x1 - expansion_x))
    new_y1 = max(0, y1)
    new_x2 = min(w, int(x2 + expansion_x))
    new_y2 = min(h, int(y2 + expansion_y))
    return new_x1, new_y1, new_x2, new_y2


def get_max_face(face_boxes):
    # P1 co_varnames=['face_boxes','areas','max_idx'](probe_behavior2/3/fix_out.json)。
    # P8 实测守卫存在:shape (1,5)(含全 NaN 行)→ 直接返回 face_boxes[0].astype(int32)
    #   (nan1 → 全 INT_MIN,未抛 All-NaN);shape (0,5) → ValueError: All-NaN slice
    #   encountered(表明确实调用 np.nanargmax);多行 → 面积最大行、并列取首个、
    #   NaN 面积被忽略(nan_row → 取第 1 行);list → AttributeError: 'list' object
    #   has no attribute 'shape'(第 18 行,先于任何索引)。
    if face_boxes.shape[0] == 1:
        return face_boxes[0].astype(np.int32)
    areas = (face_boxes[:, 2] - face_boxes[:, 0]) * (face_boxes[:, 3] - face_boxes[:, 1])
    max_idx = np.nanargmax(areas)
    return face_boxes[max_idx].astype(np.int32)


def fast_3dmm_bounds(xmin, ymin, w_rect, h_rect, img_w, img_h):
    # 同 preprocess.fast_3dmm_bounds(P2 fast_3dmm_bounds_a 双侧同值)。
    wh_ratio = w_rect / h_rect
    x_c = xmin + w_rect / 2
    half_width = h_rect * 0.8
    height_factor = h_rect * 0.45
    Xmin_3dmm = max(0, int(x_c - half_width))
    Xmax_3dmm = min(img_w, int(x_c + half_width))
    Ymin_3dmm = max(0, int(ymin + height_factor - half_width))
    Ymax_3dmm = min(img_h, int(ymin + height_factor + half_width))
    return Xmin_3dmm, Xmax_3dmm, Ymin_3dmm, Ymax_3dmm


def fast_crop_bounds(xmin, ymin, w, img_w, img_h, wh_ratio_factor):
    # 同 preprocess.fast_crop_bounds(P2 fast_crop_bounds_a 双侧同值)。
    x_c = xmin + w / 2
    half_width = w * 0.75 * wh_ratio_factor
    height_factor = w * 0.6 * wh_ratio_factor
    Xmin = max(0, int(x_c - half_width))
    Xmax = min(img_w, int(x_c + half_width))
    Ymin = max(0, int(ymin + height_factor - half_width))
    Ymax = min(img_h, int(ymin + height_factor + half_width))
    return Xmin, Xmax, Ymin, Ymax


def fast_landmark_transform(landmarks, xmin, ymin, scale_x, scale_y):
    # 同 preprocess.fast_landmark_transform(P2 双侧同值)。
    result = np.array(landmarks)
    result[:, 0] = (result[:, 0] - xmin) * scale_x
    result[:, 1] = (result[:, 1] - ymin) * scale_y
    return result


def fast_pose_check(head_poses, pose_threshold):
    # 同 preprocess.fast_pose_check(严格不等式);P7 双侧对拍含边界例一致。
    return np.all((head_poses > pose_threshold[:, 0]) & (head_poses < pose_threshold[:, 1]))


def warp_imgs(imgs_data):
    # 同 preprocess.warp_imgs;P3/P4 实测 v2 侧返回 {idx: {'imgs_data':…, 'idx':…}}。
    return {idx: {'imgs_data': img, 'idx': idx} for idx, img in enumerate(imgs_data)}


class op:
    """preprocess_v2 的人脸预处理类(R024;W6-4 深度加固轮按 oracle 实测重写)。

    W6-4 证据(evidence/vm_sessions/probe_w64{b,c,d,e,f}_out.json + pyd 反汇编
    slotmap_v2/cfa_trace):
      * __init__(wh=0.97, img_size=256)(签名实测):实例属性 = wh/img_size/
        pose_threshold(恒 [[-70,50],[-100,100],[-70,70]])/target_size(
        img_size + 10*(img_size//256))/target_size_float/wh_ratio_factor(1.0/wh)/
        no_face([])/dlib_predictor(shape_predictor('checkpoints/
        shape_predictor_68_face_landmarks.dat'))/face_detector(dlib frontal)。
      * calculate_face_angles:反汇编 GetItemInt 序列为 landmarks[1]/[2]/[3]/[4]
        (行重排关键点),eye_center=(L[1][1]+L[2][1])/2、mouth_center=(L[3][1]+
        L[4][1])/2;数值公式仅部分定谳(登记未定谳),可差分行为 = 形状异常路径
        (转置输入 → IndexError: index 2 is out of bounds for axis 0 with size 2,
        probe_w64e 实测)。
      * face_detect(无人脸合成图,probe_w64b):返回 (int32 (N,4) 全零,
        int32 (N,5,2) 全零),N = len(images)。检测器内部实现未定谳
        (灰度输入报 torch conv 错误 → oracle 内部为 torch CNN;候选按 dlib
        frontal 检测器重建,零图无人脸路径行为一致,登记偏差)。
      * flow(caped_img)(probe_w64c):返回 (data_dict, no_face);条目 =
        {'imgs_data': img, 'idx': idx, 'bounding_box_p': int32 空数组};data_dict
        挂到 self。
      * loc_detect_face/flow_optimized 经 self.data_dict 协作;无人脸时
        loc 返回 {}、flow_optimized 返回 None(probe_w64c 实测)。
    """

    def __init__(self, wh=0.97, img_size=256):
        # P1 co_varnames=['self','wh','img_size'](属性赋值不占局部名)
        self.wh = wh
        self.img_size = img_size
        # W6-4 probe_w64b:pose_threshold 恒定(与 v1 不同表)
        self.pose_threshold = np.array([[-70.0, 50.0], [-100.0, 100.0], [-70.0, 70.0]])
        # W6-4 probe_w64c/f:target_size = img_size + 10*(img_size//256)
        self.target_size = img_size + 10 * (img_size // 256)
        self.target_size_float = float(self.target_size)
        self.wh_ratio_factor = 1.0 / (wh * 1.0)
        self.no_face = []
        # ST 'checkpoints/shape_predictor_68_face_landmarks.dat'(相对 cwd)
        self.dlib_predictor = dlib.shape_predictor(
            'checkpoints/shape_predictor_68_face_landmarks.dat')
        self.face_detector = dlib.get_frontal_face_detector()

    def calculate_face_angles(self, landmarks, w, h):
        # P1 co_varnames=['self','landmarks','w','h','left_eye','right_eye','nose',
        #                 'mouth_left','mouth_right','eye_center_y','mouth_center_y',
        #                 'nose_to_eye','nose_to_mouth','pitch_rate','pitch','yaw_rate',
        #                 'yaw','eye_dist_x','eye_dist_y','roll']
        # W6-4 反汇编:GetItemInt 序 1,2,3,4(行重排关键点);eye/mouth 中心取
        # L[1][1]/L[2][1] 与 L[3][1]/L[4][1] 的均值。数值公式未完全定谳(登记),
        # 以下实现为反汇编可证部分 + 同构补全;形状异常路径与 oracle 一致
        # (转置 (2,68) 输入 → IndexError: index 2 is out of bounds …,两侧逐字同)。
        left_eye = landmarks[1]
        right_eye = landmarks[2]
        nose = landmarks[3]
        mouth_left = landmarks[4]
        mouth_right = landmarks[5]
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        mouth_center_y = (nose[1] + mouth_left[1]) / 2
        nose_to_eye = abs(nose[1] - eye_center_y)
        nose_to_mouth = abs(nose[1] - mouth_center_y)
        eye_dist_x = mouth_right[0] - nose[0]
        eye_dist_y = mouth_right[1] - nose[1]
        pitch_rate = nose_to_mouth / (w / 4)
        pitch = np.arctan(pitch_rate) / np.pi * 180
        yaw_rate = nose_to_eye / (h / 4)
        yaw = np.arctan(yaw_rate) / np.pi * 180
        roll = np.arctan(eye_dist_y / eye_dist_x) / np.pi * 180
        return pitch, yaw, roll

    def face_detect(self, images):
        # P1 co_varnames=['self','images','keypoints','predictions','frame','det_ret',
        #                 'box','keypoint','e','results','pady1','pady2','padx1','padx2',
        #                 'rect','image','y1','y2','x1','x2','boxes']
        # W6-4 probe_w64b:无人脸合成图 → (int32 (N,4) 全零, int32 (N,5,2) 全零)。
        boxes = []
        keypoints = []
        for image in images:
            det_ret = self.face_detector(image, 1)
            if len(det_ret) == 0:
                boxes.append((0, 0, 0, 0))
                keypoints.append(((0, 0), (0, 0), (0, 0), (0, 0), (0, 0)))
            else:
                rect = det_ret[0]
                parts = self.dlib_predictor(image, rect)
                box = (rect.left(), rect.top(), rect.right(), rect.bottom())
                keypoint = [(parts.part(i).x, parts.part(i).y) for i in range(5)]
                boxes.append(box)
                keypoints.append(tuple(keypoint))
        results = (np.array(boxes, dtype=np.int32).reshape(-1, 4),
                   np.array(keypoints, dtype=np.int32).reshape(-1, 5, 2))
        return results

    def loc_detect_face(self, idx_batch):
        # P1 co_varnames 见 probe_surface;W6-4:经 data_dict[idx]['imgs_data'] 取图
        # (probe_w64b:对 ndarray data_dict → IndexError),无人脸 → 返回 {}。
        results = {}
        for idx in idx_batch:
            loc_dict = self.data_dict[idx]
            img = loc_dict['imgs_data']
            boxes, keypoints = self.face_detect([img])
            if boxes.shape[0] == 0 or int(boxes[0].sum()) == 0:
                loc_dict['no_face'] = True
                continue
            loc_dict['boxes'] = boxes
            results[idx] = loc_dict
        return results

    def flow_optimized(self):
        # P1 co_varnames=['self','keys','detection_results']
        # W6-4 probe_w64c:flow 之后调用返回 None 且 data_dict 不变(无人脸条目
        # 无可优化路径)。
        detection_results = self.data_dict
        keys = list(detection_results.keys())
        for idx in keys:
            entry = detection_results[idx]
            if entry.get('bounding_box_p', None) is not None and \
                    getattr(entry['bounding_box_p'], 'size', 0) > 0:
                entry['optimized'] = True
        return None

    def flow(self, caped_img):
        # P1 co_varnames=['self','caped_img']
        # W6-4 probe_w64c:flow([img]) → (data_dict, no_face);无人脸条目 =
        # {'imgs_data': img, 'idx': idx, 'bounding_box_p': int32 空数组}。
        self.data_dict = {}
        self.imgs_data = []
        no_face = []
        for idx, img in enumerate(caped_img):
            self.imgs_data.append(img)
            boxes, keypoints = self.face_detect([img])
            if boxes.shape[0] == 0 or int(boxes[0].sum()) == 0:
                self.data_dict[idx] = {
                    'imgs_data': img, 'idx': idx,
                    'bounding_box_p': np.array([], dtype=np.int32),
                }
                no_face.append(idx)
            else:
                self.data_dict[idx] = {
                    'imgs_data': img, 'idx': idx,
                    'bounding_box_p': boxes[0],
                }
        return self.data_dict, no_face
