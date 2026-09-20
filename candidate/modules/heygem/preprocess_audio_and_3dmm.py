# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# S004 静态还原(唯一事实源):
#   evidence/modules/_heygem_orphans/disasm/modules_heygem_preprocess_audio_and_3dmm.pyc.dis
# 并用原版 pyc 的 xdis 精确表复核 co_argcount / co_kwonlyargcount / co_nlocals /
# co_varnames / co_cellvars / co_consts / co_lnotab。pycdc 参考件为空(0 行),未采用。
#
# 逐项对齐的硬证据:
#   * 模块级:docstring 逐字;导入顺序与形态 = IMPORT_NAME/IMPORT_FROM 序列:
#       import re / from PIL import Image / from scipy import signal / import time /
#       import numpy as np / import cv2 / video_time = [] / import multiprocessing.dummy as mp
#     末条 .dis 为 LOAD_CONST None(fromlist=None)+ IMPORT_NAME multiprocessing.dummy +
#     IMPORT_FROM dummy + STORE_NAME mp ⇒ 点号导入别名形式(不是 from ... import ...)。
#   * class op 六个方法签名严格按 Arg Count(含 self):
#       __init__(9) show(1) get_max_face(2) loc_detect_face(2) loc_crop_face(2)
#       smooth_(1) flow(1)
#     因此 show / flow / smooth_ 只收 self;max_len、bbx_smooth、keylist、it、conv_core、
#     loc_dict、idx 等全部是函数内局部量(co_varnames),不是形参。S003 版本把
#     loc_detect_face/loc_crop_face 写成 6 参、smooth_ 写成 6 参,均与 .dis 不符,已改正。
#   * loc_detect_face 的 x1/y1 是 cellvars(被内层 <listcomp> 用 LOAD_DEREF 捕获)⇒ 保留闭包写法。
#   * 常量逐字取自 LOAD_CONST:256 / 10 / -70 / 50 / -100 / 100 / 70 / 0.1 / 0.8 / 0.35 / 1.25 /
#     0.75 / 0.15 / 1.35 / 12 / (5, 1) / 5 / 100.0 / 0.0 / 4 / 2 / 3 等。
#   * dict 键串与日志串:imgs_data / bounding_box_p / bounding_box / landmarks / crop_lm /
#     crop_img / '侧脸检测不通过' / 'symm' / 'same'。
#
# 未能从字节码还原 / 属语义重建的部分(诚实声明):
#   1) 原文件的注释文字与空行不可恢复(字节码不含注释);本文件行号与原文件行号不等。
#      co_lnotab 只留下"哪些行有语句"的痕迹,例如 flow 的 def 在 L202、首条语句在 L206,
#      说明 L203-L205 原为注释,文字无法还原。
#   2) loc_detect_face 的姿态判定:字节码是**单个 BoolOp + 三条链式比较**
#      (DUP_TOP / ROT_THREE / COMPARE_OP,链中断失败走 POP_TOP 再跳到同一个 else 块 698),
#      不是三层嵌套 if(嵌套 if 会为每个 else 各生成一份 print 块,而 .dis 里只有一份)。
#      故写成 `if A and B and C:`,其中每条是 a < x < b 的链式比较。
#   3) 括号续行与反斜杠续行在字节码上不可区分;本文件对长表达式自行选择括号续行。
#   4) `t0 = time.time()`(loc_detect_face、loc_crop_face 各一处)按字节码保留:
#      STORE_FAST 之后再无 LOAD_FAST t0,属原码"存而不取"。
#   5) 数值字面量类型按 co_consts 还原:loc_crop_face 的哨兵框是 float(0.0 / 100.0),
#      而索引 0 是 int;`bbx_smooth + 0` 中的 0 是 int。
#   6) `self.manager = mp.Manager`(无调用)后接 `self.manager().dict()`,是 .dis 的真实序列
#      (LOAD_ATTR 后直接 STORE_ATTR,无 CALL_FUNCTION),按原样保留。
# ---------------------------------------------------------------------------
"""
Created on Thu Jan 21 11:27:22 2021

@author: guiji
"""
import re
from PIL import Image
from scipy import signal
import time
import numpy as np
import cv2

video_time = []

import multiprocessing.dummy as mp


class op:

    def __init__(self, caped_img2, wh, scrfd_detector, scrfd_predictor, hp, lm3d_std, img_size, driver_flag):
        self.manager = mp.Manager
        self.mp_dict = self.manager().dict()
        self.img_size = img_size

        self.target_size = self.img_size + int(self.img_size / 256) * 10
        for idx in caped_img2.keys():
            self.mp_dict[idx] = caped_img2[idx]
        self.wh = wh
        self.scrfd_detector = scrfd_detector
        self.scrfd_predictor = scrfd_predictor
        self.hp = hp
        self.pose_threshold = [[-70, 50], [-100, 100], [-70, 70]]
        self.driver_flag = driver_flag
        self.no_face = []

    def show(self):
        for idx in self.mp_dict.keys():
            print(self.mp_dict[idx], idx)

    def get_max_face(self, face_boxes):
        if face_boxes.shape[0] == 1:
            return face_boxes[0].astype(int)

        return face_boxes[np.nanargmax(
            np.abs(face_boxes[:, 2] - face_boxes[:, 0]) * np.abs(face_boxes[:, 3] - face_boxes[:, 1]))].astype(int)

    def loc_detect_face(self, idx):
        loc_dict = self.mp_dict[idx]
        img = loc_dict['imgs_data']
        t0 = time.time()
        face_boxes, _ = self.scrfd_detector.get_bboxes(img)

        h, w = img.shape[:2]
        if face_boxes.shape[0] > 0:
            x1, y1, x2, y2, score = self.get_max_face(face_boxes)
            x1 = max(0, x1 - int((x2 - x1) * 0.1))
            y1 = max(0, y1)
            x2 = min(w, x2 + int((x2 - x1) * 0.1))
            y2 = min(h, y2 + int((y2 - y1) * 0.1))

            face_img = img[int(y1):int(y2), int(x1):int(x2)]
            pots = self.scrfd_predictor.forward(face_img)[0]
            landmarks = np.array([[x1 + x, y1 + y] for x, y in pots.astype(np.int32)])
            xmin, ymin, w, h = cv2.boundingRect(np.array(landmarks))
            x_c = xmin + w / 2
            wh = w / h
            Xmin_3dmm = int(x_c - w / wh * 0.8)
            Xmax_3dmm = int(x_c + w / wh * 0.8)
            Ymin_3dmm = int(ymin - w / wh * 0.35)
            Ymax_3dmm = int(ymin + w / wh * 1.25)
            if Xmin_3dmm <= 0:
                Xmin_3dmm = 0
            if Ymin_3dmm <= 0:
                Ymin_3dmm = 0
            if Xmax_3dmm >= img.shape[1]:
                Xmax_3dmm = img.shape[1]
            if Ymax_3dmm >= img.shape[0]:
                Ymax_3dmm = img.shape[0]
            head_poses = self.hp.get_head_pose(img[int(Ymin_3dmm):int(Ymax_3dmm), int(Xmin_3dmm):int(Xmax_3dmm)])
            if (self.pose_threshold[0][0] < head_poses[0] < self.pose_threshold[0][1] and
                    self.pose_threshold[1][0] < head_poses[1] < self.pose_threshold[1][1] and
                    self.pose_threshold[2][0] < head_poses[2] < self.pose_threshold[2][1]):
                loc_dict['bounding_box_p'] = np.array((y1, y2, x1, x2))
            else:
                print('侧脸检测不通过', head_poses)
                loc_dict['bounding_box_p'] = []

        else:
            loc_dict['bounding_box_p'] = []
        self.mp_dict[idx] = loc_dict

    def loc_crop_face(self, idx):
        loc_dict = self.mp_dict[idx]
        img = loc_dict['imgs_data']
        dets = self.mp_dict[idx]['bounding_box_p']

        if len(dets) == 0 or max(dets) == 0.0:
            self.no_face.append(idx)
            dets = [0.0, 100.0, 0.0, 100.0]
            self.mp_dict[idx]['bounding_box_p'] = [0.0, 0.0, 0.0, 0.0]
        if len(dets) > 0:
            t0 = time.time()
            face_landmarks = self.scrfd_predictor.forward(img[int(dets[0]):int(dets[1]), int(dets[2]):int(dets[3])])[0]

            landmarks = face_landmarks + np.array([dets[2], dets[0]])[np.newaxis]
            loc_dict['landmarks'] = landmarks
            xmin, ymin, w, h = cv2.boundingRect(np.array(landmarks).astype(np.int32))
            x_c = xmin + w / 2
            Xmin = int(x_c - w / self.wh * 0.75)
            Xmax = int(x_c + w / self.wh * 0.75)
            Ymin = int(ymin - w / self.wh * 0.15)
            Ymax = int(ymin + w / self.wh * 1.35)

            if Xmin <= 0:
                Xmin = 0
            if Ymin <= 0:
                Ymin = 0
            if Xmax >= img.shape[1]:
                Xmax = img.shape[1]
            if Ymax >= img.shape[0]:
                Ymax = img.shape[0]
            loc_dict['bounding_box'] = np.array([Ymin, Ymax, Xmin, Xmax])

            lm_crop = np.zeros(landmarks.shape)
            lm_crop[..., 0] = self.target_size * (landmarks[..., 0] - Xmin) / (Xmax - Xmin)
            lm_crop[..., 1] = self.target_size * (landmarks[..., 1] - Ymin) / (Ymax - Ymin)

            img_crop = img[Ymin:Ymax, Xmin:Xmax]
            if self.driver_flag:
                img_crop = cv2.cvtColor(cv2.resize(img_crop, (self.target_size, self.target_size),
                                                   interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
            else:
                img_crop = cv2.resize(img_crop, (self.target_size, self.target_size),
                                      interpolation=cv2.INTER_CUBIC)

            loc_dict['crop_lm'] = lm_crop
            loc_dict['crop_img'] = img_crop
            self.mp_dict[idx] = loc_dict

    def smooth_(self):
        max_len = np.array([it for it in self.mp_dict.keys()]).max()
        bbx_smooth = np.zeros((max_len, 4))
        keylist = list(self.mp_dict.keys())

        keylist.sort()
        for it in keylist:

            if len(self.mp_dict[it]['bounding_box_p']) != 4:
                bbx_smooth[it - 1:, :] = bbx_smooth[it - 2:, :]
                continue

            self.mp_dict[it]['bounding_box_p'] = bbx_smooth[it - 1:, :]
        conv_core = np.ones((5, 1)) / 5

        bbx_smooth2 = signal.convolve2d(bbx_smooth, conv_core, boundary='symm', mode='same')
        bbx_smooth_dif = np.where(np.abs(bbx_smooth2 - bbx_smooth).sum(1) > 12)[0]
        bbx_smooth3 = bbx_smooth + 0
        bbx_smooth[bbx_smooth_dif] = bbx_smooth3[bbx_smooth_dif]
        bbx_smooth4 = signal.convolve2d(bbx_smooth3, conv_core, boundary='symm', mode='same')

        for it in self.mp_dict:
            loc_dict = self.mp_dict[it]
            loc_dict['bounding_box_p'] = bbx_smooth4[it - 1:, :]
            self.mp_dict[it] = loc_dict

    def flow(self):
        for idx in self.mp_dict.keys():
            self.loc_detect_face(idx)

        for idx in self.mp_dict.keys():
            self.loc_crop_face(idx)
