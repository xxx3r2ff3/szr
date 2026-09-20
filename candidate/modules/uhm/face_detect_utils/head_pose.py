# -*- coding: utf-8 -*-
"""uhm.face_detect_utils.head_pose —— T4R 私有语义重建(R026)。

证据(oracle 侧只读实测):
  * P1 probe_surface_out.json —— dir()=['Headpose','cv2','np','ort','os'];
    类方法表 {__init__(cpu=False, onnx_path=''), get_head_pose(self, image),
    softmax(self, x)};各方法 co_varnames/co_firstlineno:
    __init__ L12 ['self','cpu','onnx_path','cache_dir','providers','e','idx','idx','output','output'],
    softmax L73 ['self','x','a','b'], get_head_pose L82 ['self','image',
    'croped_resized_frame','rgb','chw','nchw','yaw','roll','pitch']。
  * P4 probe_behavior3_out.json —— 构造成功;实例属性 idx_tensor_yaw(120 个 0-d float32)、
    idx_tensor(66)、whenet_H/W=224、whenet_input_name='input_1'、
    whenet_output_names=['tf.identity','tf.identity_1','tf.identity_2']、
    whenet_output_shapes=[[1,120],[1,66],[1,66]]、mean/std=ImageNet;
    onnx_path='' 或不存在 → Exception('load head pose onnx failed: ' + str(e));
    softmax 1D/标量 → AxisError(axis=1),2D → axis=1 归一。
  * P5 probe_behavior5_out.json —— 用同一 session 复算:归一化取
    (chw/255 - mean)/std 的 float32 NCHW 时,逐输出 softmax 加权均值与 oracle 返回值
    满足 yaw=(m120-60)*3、pitch/roll=(m66-33)*3,返回序为 (pitch, roll, yaw)
    (与 ST 的 ', Pose (Pitch, Roll, Yaw): ' 一致);两次同图调用完全一致(确定性)。
  * ST constants.txt —— '.trtcache'、'TensorrtExecutionProvider'、'trt_engine_cache_enable'、
    'trt_engine_cache_path'、'trt_fp16_enable'、'CUDAExecutionProvider'、'CPUExecutionProvider'、
    'load head pose onnx failed: '。
"""
import os

import cv2
import numpy as np
import onnxruntime as ort


class Headpose:
    def __init__(self, cpu=False, onnx_path=''):
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.trtcache')
        if cpu:
            providers = ['CPUExecutionProvider']
        else:
            providers = [
                ('TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': cache_dir,
                    'trt_fp16_enable': True,
                }),
                'CUDAExecutionProvider',
            ]
        try:
            self.whenet_session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception as e:
            raise Exception('load head pose onnx failed: ' + str(e))

        self.whenet_input_name = self.whenet_session.get_inputs()[0].name
        self.whenet_output_names = [output.name for output in self.whenet_session.get_outputs()]
        self.whenet_output_shapes = [output.shape for output in self.whenet_session.get_outputs()]
        self.whenet_H = 224
        self.whenet_W = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.idx_tensor_yaw = [np.array(idx, dtype=np.float32)
                               for idx, output in enumerate(range(self.whenet_output_shapes[0][1]))]
        self.idx_tensor = [np.array(idx, dtype=np.float32)
                           for idx, output in enumerate(range(self.whenet_output_shapes[1][1]))]

    def softmax(self, x):
        # P1 co_varnames=['self','x','a','b'];P4 实测 2D → axis=1 归一
        # ([[1,2,3],[0,0,0]] → [[0.090,0.245,0.665],[1/3,1/3,1/3]]),
        # 1D/标量 → AxisError: axis 1 is out of bounds → 与 keepdims 形式一致。
        a = np.max(x, axis=1, keepdims=True)
        b = np.exp(x - a)
        return b / np.sum(b, axis=1, keepdims=True)

    def get_head_pose(self, image):
        # P1 co_varnames=['self','image','croped_resized_frame','rgb','chw','nchw',
        #                 'yaw','roll','pitch'](赋值序 yaw→roll→pitch 与输出序对应关系
        # 由 P5 反解确定,见文件头)。
        croped_resized_frame = cv2.resize(image, (self.whenet_W, self.whenet_H))
        rgb = croped_resized_frame[:, :, ::-1]
        chw = rgb.transpose(2, 0, 1)
        nchw = ((chw / 255.0 - np.array(self.mean).reshape(3, 1, 1))
                / np.array(self.std).reshape(3, 1, 1))[np.newaxis].astype(np.float32)
        yaw = self.whenet_session.run([self.whenet_output_names[0]],
                                      {self.whenet_input_name: nchw})[0]
        roll = self.whenet_session.run([self.whenet_output_names[1]],
                                       {self.whenet_input_name: nchw})[0]
        pitch = self.whenet_session.run([self.whenet_output_names[2]],
                                        {self.whenet_input_name: nchw})[0]
        # P8 dtype 判别:oracle 返回 np.float32 标量,而本环境 np.float32 标量 - int
        #   会升为 float64 → 说明原式先沿 axis=1 求和得 (1,) float32 数组、再取 [0]。
        yaw = (np.sum(self.softmax(yaw) * self.idx_tensor_yaw, axis=1) - 60) * 3
        roll = (np.sum(self.softmax(roll) * self.idx_tensor, axis=1) - 33) * 3
        pitch = (np.sum(self.softmax(pitch) * self.idx_tensor, axis=1) - 33) * 3
        return pitch[0], roll[0], yaw[0]
