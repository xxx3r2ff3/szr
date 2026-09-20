# -*- coding: utf-8 -*-
"""uhm.face_detect_utils.scrfd —— T4R 私有语义重建(R027)。

谱系:insightface `detection/scrfd`(onnxruntime 版)改写;本文件按 oracle 侧
只读证据逐项核对:
  * P1 probe_surface_out.json —— dir()/签名/co_varnames/co_firstlineno
  * P4 probe_behavior3_out.json —— softmax/distance2bbox/distance2kps/get_scrfd/
    scrfd_2p5gkps 实测 + SCRFD(model_file=…) 属性表(nms_thresh=0.35/input_size=(640,640)/
    output_names 9 项/use_kps=True/_num_anchors=2/fmc=3/_feat_stride_fpn=[8,16,32])
  * ST constants.txt —— '~/.insightface/models'、'insightface.model_zoo.model_store'、
    'Please install insightface to use the download feature.'、'init scrfd'
"""
import datetime
import os
import os.path as osp
import sys

import cv2
import numpy as np
import onnxruntime


def softmax(z):
    # P1 co_varnames=['z','s','e_x','div'];P4 实测 2D 输入按 axis=1 归一,
    # 1D 输入 → AssertionError(空消息)→ 与 assert len(z.shape)==2 一致。
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div


def distance2bbox(points, distance, max_shape=None):
    # P1 co_varnames=['points','distance','max_shape','x1','y1','x2','y2'];
    # P4 d2bbox: (px-d0, py-d1, px+d2, py+d3);d2bbox_max max_shape=(50,60) 时
    # 第 2 行 x 钳到 60、y 钳到 50 → 上界用 max_shape[1]/max_shape[0](不减 1)。
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clip(min=0, max=max_shape[1])
        y1 = y1.clip(min=0, max=max_shape[0])
        x2 = x2.clip(min=0, max=max_shape[1])
        y2 = y2.clip(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    # P1 co_varnames=['points','distance','max_shape','preds','i','px','py'];
    # P4 d2kps 实测 px 用 points[:, i%2]、py 用 points[:, i%2+1],
    # 结果按 px,py 交替 np.stack(axis=-1);d2kps_max 钳位同上。
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clip(min=0, max=max_shape[1])
            py = py.clip(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class SCRFD:
    def __init__(self, model_file=None, session=None, cpu=False):
        # P1 co_varnames=['self','model_file','session','cpu','cache_dir','providers'];
        # P4 构造实测属性:model_file(str)/session(InferenceSession)/taskname='detection'/
        # batched=True/center_cache={}/nms_thresh=0.35/input_size=(640,640)/
        # input_name='input.1'/output_names 9 项/use_kps=True/_num_anchors=2/fmc=3/
        # _feat_stride_fpn=[8,16,32];cpu=True 时 get_providers()=['CPUExecutionProvider']。
        self.model_file = model_file
        self.session = session
        self.taskname = 'detection'
        self.batched = False
        # P12 实测:session is None 且默认 cpu=False 时先打印 GPU 提示(与 heygem 同谱系实现);
        # 该提示在 cpu=True 时不出现(providers 仅 CPU,无 TRT/CUDA 初始化失败)。
        if self.session is None:
            # W6-4 深度加固:P12 实测 cpu=True 时无该提示(providers 仅 CPU),
            # 仅 cpu=False 路径打印(get_available_providers 恒列 CUDA,与可载入无关)
            if not cpu:
                print('检测人脸使用GPU' if any('CUDA' in p for p in
                      onnxruntime.get_available_providers()) else '')
        if self.session is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file)
            # W6-4 probe_w64j:get_bboxes 构造路径 stdout 内嵌 providers 字典,
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
            self.session = onnxruntime.InferenceSession(self.model_file, providers=providers)
        self.center_cache = {}
        self.nms_thresh = 0.35
        self._init_vars()

    def _init_vars(self):
        # P1 co_varnames=['self','input_cfg','input_shape','input_name','outputs',
        #                 'output_names','o'](与 P4 属性表逐项吻合)。
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        self.input_shape = input_shape
        self.input_name = input_name
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.output_names = output_names
        self.use_kps = False
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def prepare(self, ctx_id, **kwargs):
        # P1 co_varnames=['self','ctx_id','kwargs','nms_thresh','input_size'];
        # P4 实测 ctx_id=-1 + input_size=(640,640) 时打印
        # 'warning: det_size is already set in scrfd model, ignore'。
        if ctx_id < 0:
            self.session.set_providers(['CPUExecutionProvider'])
        nms_thresh = kwargs.get('nms_thresh', None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        input_size = kwargs.get('input_size', None)
        if input_size is not None:
            if self.input_size is not None:
                print('warning: det_size is already set in scrfd model, ignore')
                return
            else:
                self.input_size = input_size

    def forward(self, img, thresh):
        # P1 co_varnames 27 项与 insightface onnxruntime 版逐项一致
        # (blob/net_outs/input_height/input_width/fmc/idx/stride/…/pos_kpss)。
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128, input_size, (127.5, 127.5, 127.5),
                                     swapRB=True)
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        # W6-4 深度加固:batched 模型(outputs[0].shape==3)输出含批次维 (1,K,C),
        # oracle 前向空结果形状 (0,1)/(0,4)/(0,5,2) 唯一确定此处按 axis0 压缩
        # (probe_w64_e 前向网格;缺失该步候选在 stride-8 广播处报错)
        if self.batched:
            net_outs = [out[0] for out in net_outs]

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + fmc]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = net_outs[idx + fmc * 2] * stride
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors,
                                              axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers
            pos_inds = np.where(scores >= thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, thresh=0.7, input_size=None, max_num=0, metric='default'):
        # P1 co_varnames 30 项与 insightface onnxruntime 版逐项一致;
        # P4 实测空图 → (shape (0,5) float32, shape (0,5,2) float32)。
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size

        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, thresh)

        # [I007 E2E 定谳 i007-fix-2] 检出框/关键点必须回除 det_scale(映射回原图坐标)。
        # 整机 E2E oracle 真值(共享目录 i007_truth*_console.log + 旁路探针
        # i007_probe_truth*.py):512×512 人脸图 oracle det=[180.12,60.19,271.6,179.74],
        # 候选漏除后恰为其 1.25 倍(det_scale)→ 画布 512×900 非方形图同样回除后
        # 与粘贴几何吻合。cosmetic:与 insightface 上游一致,vstack 处回除。
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0],
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0
            bindex = np.argsort(values)[::-1][:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets):
        # P1 co_varnames=['self','dets','nms_thresh','x1','y1','x2','y2','scores','areas',
        #                 'order','keep','i','xx1','yy1','xx2','yy2','w','h','inter','ovr',
        #                 'inds'];P4 SCRFD_nms 实测 [[0,0,10,10,.9],[1,1,11,11,.8],
        #                 [50,50,60,60,.7]] → keep=[0, 2](thresh=0.35)。
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep


def get_scrfd(name, download=False, root='~/.insightface/models', **kwargs):
    # P1 co_varnames=['name','download','root','kwargs','get_model_file','_file'];
    # P4 实测:get_scrfd('nosuchmodel.onnx') → AssertionError(空消息,os.path.exists 断言);
    # get_scrfd('nosuchmodel', download=True) → ImportError:
    # 'Please install insightface to use the download feature.'(ST 逐字命中)。
    # P12 实测:get_scrfd(model, cpu=True) 仍打印 '检测人脸使用GPU' 与 TensorRT EP 报错
    #   → kwargs **未**下传给 SCRFD(SCRFD 取默认 cpu=False,providers 走 TRT/CUDA 列表后回退 CPU)。
    if not download:
        assert os.path.exists(name)
        return SCRFD(name)
    else:
        try:
            from insightface.model_zoo.model_store import get_model_file
        except ImportError:
            raise ImportError('Please install insightface to use the download feature.')
        _file = get_model_file('scrfd_%s' % name, root=root)
        return SCRFD(_file)


def scrfd_2p5gkps(**kwargs):
    # P1 co_varnames=['kwargs'];P4 实测 scrfd_2p5gkps() 走 download 分支 →
    # 同一 ImportError;模型名实参取自 ST 中的 '2p5gkps' 字面量(表内无 '2.5g')。
    return get_scrfd('2p5gkps', download=True, **kwargs)
