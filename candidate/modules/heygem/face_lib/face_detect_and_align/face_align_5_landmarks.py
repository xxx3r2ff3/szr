# -*- coding: utf-8 -*-
# FaceDetect5Landmarks(SCRFD/MTCNN 五点检测 + 对齐裁剪) —— S004 逐指令重建
# (face_lib/face_detect_and_align/face_align_5_landmarks.pyc,py3.8;反汇编
#  evidence/modules/_heygem_orphans/disasm/
#  modules_heygem_face_lib_face_detect_and_align_face_align_5_landmarks.pyc.dis,1069 行;
#  8 个 code object:FaceDetect5Landmarks + __init__/get_bboxes/tracking_filter/
#  bboxes_filter/get_single_face/get_multi_face/draw_face)
#
# 签名(Arg Count / defaults 元组照 [Disassembly] 的 MAKE_FUNCTION):
#   __init__(3, ('scrfd_500m', False)) / get_bboxes(5, (0.5, 0, None)) /
#   tracking_filter(1) / bboxes_filter(2) / get_single_face(4, ('mtcnn_512', False)) /
#   get_multi_face(3, ('mtcnn_512',)) / draw_face(1)
#   类内各函数 co_consts[0] 为 docstring 的三处已逐字补回(get_bboxes/get_single_face/
#   get_multi_face);tracking_filter/bboxes_filter/draw_face/__init__ 原版无 docstring。
#   模块 Constants 首项为 0(import level)⇒ 无模块 docstring。
#
# 出处判读要点:
#   - 模块导入与 level:`from cv2box.utils.math import Normalize`、`from cv2box import CVImage`
#     (level 0);`from .scrfd_insightface import SCRFD`(**level 1,相对导入**);
#     `from face_lib.face_detect_and_align.face_align_utils import norm_crop, apply_roi_func`
#     (level 0);SCRFD_MODEL_PATH/MTCNN_MODEL_PATH 两个常量串(MTCNN_MODEL_PATH 原版未使用)。
#   - __init__:mode/tracking → dis_list=[] → last_bboxes_=[] →
#     `assert self.mode in ('scrfd','scrfd_500m','mtcnn')` →
#     `self.bboxes = self.kpss = self.image = None`(DUP_TOP 链式赋值)→ 'scrfd' in mode 时按
#     mode 选 scrfd_500m_bnkps_shape640x640.onnx / scrfd_10g_bnkps.onnx,构造 SCRFD 后
#     `prepare(ctx_id=0, input_size=(640, 640))`(关键字调用)。
#   - get_bboxes:先 `self.image = CVImage(image).rgb()`(未截取返回值 → None);
#     tracking 分支:last_bboxes_ 为空 → detect_faces(**参数 image**, max_num=1);
#     否则 detect_faces(image, max_num=0) 后 tracking_filter();elif 'scrfd' in self.mode →
#     detect_faces(**self.image**, max_num=max_num);三支均 metric='default'。
#     min_bbox_size 形参原版未使用(bboxes_filter 亦未被调用)。
#   - tracking_filter:逐框 np.linalg.norm(Normalize(...).np_norm() - Normalize(last[0]).np_norm())
#     追加进 dis_list;空表返回 ([], []);否则 argmin 取最佳框,dis_list 清空,
#     last_bboxes_ 只留最佳框,返回 (last_bboxes_, [kpss[best_index]])。
#   - bboxes_filter:min_area = np.power(min_bbox_size, 2);
#     area_list=(bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1]);
#     np.where(area_list < min_area) → np.delete(..., axis=0) 同步删 bboxes/kpss。
#   - get_single_face:mode 断言六值;bboxes.shape[0]==0 → (None, None);det_score=bboxes[...,4];
#     tracking 分支用 np.array(self.dis_list).argmax(),否则 np.argmax(det_score);
#     两分支各自 kpss=None + `if self.kpss is not None` 取 kpss(字节码重复两段);
#     apply_roi=True 时 apply_roi_func(self.image, bboxes[best_index], kpss) → norm_crop(roi,
#     roi_kpss, crop_size, mode=mode) → cvtColor(COLOR_RGB2BGR) → 返回三元组
#     (align_img, mat_rev, roi_box);否则 norm_crop(self.image, kpss, crop_size, mode=mode)
#     → cvtColor → 返回 (align_img, M)。
#   - draw_face:逐框 `x1, y1, x2, y2, score = bbox.astype(int)`,
#     cv2.rectangle(self.image, (x1,y1), (x2,y2), (255,0,0), 2);
#     kpss 非空时逐点 `kp = kp.astype(int)` + cv2.circle(self.image, tuple(kp), 1, (0,0,255), 2);
#     末尾 CVImage(self.image, image_format='cv2').show()(无 return,原版无返回值)。
#   - assert 的字节码形态为 `LOAD_GLOBAL AssertionError; RAISE_VARARGS 1`(无消息,失败即抛
#     AssertionError),见 __init__ 与 get_single_face。
# 置信度:高(8 个 code object 逐指令对齐;drawn 颜色/半径、跟踪分支的 image/self.image
#   差异等"看着奇怪"的细节均按字节码原样保留)。
import numpy as np
import cv2

from cv2box.utils.math import Normalize
from cv2box import CVImage

from .scrfd_insightface import SCRFD
from face_lib.face_detect_and_align.face_align_utils import norm_crop, apply_roi_func

SCRFD_MODEL_PATH = 'pretrain_models/face_lib/face_detect/scrfd_onnx/'
MTCNN_MODEL_PATH = 'pretrain_models/face_lib/face_detect/mtcnn_weights/'


class FaceDetect5Landmarks:

    def __init__(self, mode='scrfd_500m', tracking=False):
        self.mode = mode
        self.tracking = tracking
        self.dis_list = []
        self.last_bboxes_ = []
        assert self.mode in ('scrfd', 'scrfd_500m', 'mtcnn')
        self.bboxes = self.kpss = self.image = None
        if 'scrfd' in self.mode:
            if self.mode == 'scrfd_500m':
                scrfd_model_path = SCRFD_MODEL_PATH + 'scrfd_500m_bnkps_shape640x640.onnx'
            else:
                scrfd_model_path = SCRFD_MODEL_PATH + 'scrfd_10g_bnkps.onnx'
            self.det_model_scrfd = SCRFD(scrfd_model_path)
            self.det_model_scrfd.prepare(ctx_id=0, input_size=(640, 640))

    def get_bboxes(self, image, nms_thresh=0.5, max_num=0, min_bbox_size=None):
        """
        Args:
            image: RGB image path or Numpy array load by cv2
            nms_thresh:
            max_num:
            min_bbox_size:
        Returns:
        """
        self.image = CVImage(image).rgb()
        if self.tracking:
            if len(self.last_bboxes_) == 0:
                self.bboxes, self.kpss = self.det_model_scrfd.detect_faces(
                    image, thresh=nms_thresh, max_num=1, metric='default')
                self.last_bboxes_ = self.bboxes
            else:
                self.bboxes, self.kpss = self.det_model_scrfd.detect_faces(
                    image, thresh=nms_thresh, max_num=0, metric='default')
                self.bboxes, self.kpss = self.tracking_filter()
        elif 'scrfd' in self.mode:
            self.bboxes, self.kpss = self.det_model_scrfd.detect_faces(
                self.image, thresh=nms_thresh, max_num=max_num, metric='default')
        return self.bboxes, self.kpss

    def tracking_filter(self):
        for i in range(len(self.bboxes)):
            self.dis_list.append(np.linalg.norm(
                Normalize(self.bboxes[i]).np_norm() -
                Normalize(self.last_bboxes_[0]).np_norm()))
        if not self.dis_list:
            return [], []
        best_index = np.array(self.dis_list).argmin()
        self.dis_list = []
        self.last_bboxes_ = [self.bboxes[best_index]]
        return self.last_bboxes_, [self.kpss[best_index]]

    def bboxes_filter(self, min_bbox_size):
        min_area = np.power(min_bbox_size, 2)
        area_list = (self.bboxes[:, 2] - self.bboxes[:, 0]) * \
            (self.bboxes[:, 3] - self.bboxes[:, 1])
        min_index = np.where(area_list < min_area)
        self.bboxes = np.delete(self.bboxes, min_index, axis=0)
        self.kpss = np.delete(self.kpss, min_index, axis=0)

    def get_single_face(self, crop_size, mode='mtcnn_512', apply_roi=False):
        """
        Args:
            crop_size:
            mode: default mtcnn_512 arcface_512 arcface default_95
        Returns: cv2 image
        """
        assert mode in ('default', 'mtcnn_512', 'mtcnn_256',
                        'arcface_512', 'arcface', 'default_95')
        if self.bboxes.shape[0] == 0:
            return None, None
        det_score = self.bboxes[..., 4]
        if self.tracking:
            best_index = np.array(self.dis_list).argmax()
            kpss = None
            if self.kpss is not None:
                kpss = self.kpss[best_index]
        else:
            best_index = np.argmax(det_score)
            kpss = None
            if self.kpss is not None:
                kpss = self.kpss[best_index]
        if apply_roi:
            roi, roi_box, roi_kpss = apply_roi_func(self.image,
                                                    self.bboxes[best_index], kpss)
            align_img, mat_rev = norm_crop(roi, roi_kpss, crop_size, mode=mode)
            align_img = cv2.cvtColor(align_img, cv2.COLOR_RGB2BGR)
            return align_img, mat_rev, roi_box
        align_img, M = norm_crop(self.image, kpss, crop_size, mode=mode)
        align_img = cv2.cvtColor(align_img, cv2.COLOR_RGB2BGR)
        return align_img, M

    def get_multi_face(self, crop_size, mode='mtcnn_512'):
        """
        Args:
            crop_size:
            mode: default mtcnn_512 arcface_512 arcface
        Returns:
        """
        if self.bboxes.shape[0] == 0:
            return None
        align_img_list = []
        M_list = []
        for i in range(self.bboxes.shape[0]):
            kps = None
            if self.kpss is not None:
                kps = self.kpss[i]
            align_img, M = norm_crop(self.image, kps, crop_size, mode=mode)
            align_img_list.append(align_img)
            M_list.append(M)
        return align_img_list, M_list

    def draw_face(self):
        for i_ in range(self.bboxes.shape[0]):
            bbox = self.bboxes[i_]
            x1, y1, x2, y2, score = bbox.astype(int)
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if self.kpss is not None:
                kps = self.kpss[i_]
                for kp in kps:
                    kp = kp.astype(int)
                    cv2.circle(self.image, tuple(kp), 1, (0, 0, 255), 2)
        CVImage(self.image, image_format='cv2').show()
