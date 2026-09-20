# -*- coding: utf-8 -*-
# S003/S004 恢复件:heygem/face_detect_utils/head_pose.pyc(Python 3.8,magic 3413)
# 源 pyc  : /Volumes/A/数字人/szr2026/modules/heygem/face_detect_utils/head_pose.pyc
#           (源文件长度 3243 字节;pycdas 反汇编 768 行)
# 证据链  : evidence/modules/_heygem_orphans/disasm/face_detect_utils_head_pose.pyc.dis
# 谱系    : heygem 自有工具模块;WHENet 头部姿态(H/Y/W 三输出)→ yaw/pitch/roll。
#           路径常量自证上游来源:HeadPoseEstimation-WHENet-yolov4-onnx-openvino
#           (作者 zml),onnx_path 默认 '' 由调用方传入。
# 还原度  : 结构/常量/控制流逐指令对齐:Headpose.__init__(3)、Headpose.softmax(2)、
#           Headpose.get_head_pose(2) 及 4 个 listcomp code object 全部落齐;
#           [Constants]/[Names]/[Var Names] 逐项核对(含 'croped_resized_frame' 原码拼写)。
# 原码缺陷(忠实保留):`raise e('load head pose onnx failed')` —— 字节码为
#           LOAD_FAST e;CALL_FUNCTION 1;RAISE_VARARGS 1(对异常实例调用后抛出)。

import numpy as np
import cv2
import onnxruntime as ort


class Headpose:

    def __init__(self, cpu=False, onnx_path=''):
        self.idx_tensor_yaw = [np.array(idx, dtype=np.float32) for idx in range(120)]
        self.idx_tensor = [np.array(idx, dtype=np.float32) for idx in range(66)]
        self.whenet_H = 224
        self.whenet_W = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        try:
            self.whenet_session = ort.InferenceSession(
                onnx_path,
                providers=['CUDAExecutionProvider'] if not cpu else ['CPUExecutionProvider'])
        except Exception as e:
            raise e('load head pose onnx failed')
        if cpu:
            self.whenet_session.set_providers(['CPUExecutionProvider'])
        else:
            self.whenet_session.set_providers(['CUDAExecutionProvider'])
        self.whenet_input_name = self.whenet_session.get_inputs()[0].name
        self.whenet_output_names = [output.name for output in self.whenet_session.get_outputs()]
        self.whenet_output_shapes = [output.shape for output in self.whenet_session.get_outputs()]
        assert self.whenet_output_shapes[0] == [1, 120]
        assert self.whenet_output_shapes[1] == [1, 66]
        assert self.whenet_output_shapes[2] == [1, 66]

    def softmax(self, x):
        x -= np.max(x, axis=1, keepdims=True)
        a = np.exp(x)
        b = np.sum(np.exp(x), axis=1, keepdims=True)
        return a / b

    def get_head_pose(self, image):
        croped_resized_frame = cv2.resize(image, (self.whenet_W, self.whenet_H))
        rgb = croped_resized_frame[..., ::-1]
        rgb = (rgb / 255 - self.mean) / self.std
        chw = rgb.transpose(2, 0, 1)
        nchw = np.asarray(chw[np.newaxis, :, :, :], dtype=np.float32)
        yaw, roll, pitch = self.whenet_session.run(
            output_names=self.whenet_output_names,
            input_feed={self.whenet_input_name: nchw})
        yaw = np.sum(self.softmax(yaw) * self.idx_tensor_yaw, axis=1) * 3 - 180
        pitch = np.sum(self.softmax(pitch) * self.idx_tensor, axis=1) * 3 - 99
        roll = np.sum(self.softmax(roll) * self.idx_tensor, axis=1) * 3 - 99
        yaw, pitch, roll = np.squeeze([yaw, pitch, roll])
        return pitch, roll, yaw


if __name__ == '__main__':
    import os

    hp = Headpose(
        cpu=False,
        onnx_path=
        '/home/zml/my_code/HeadPoseEstimation-WHENet-yolov4-onnx-openvino/saved_model_224x224/model_float32.onnx'
    )
    base_dir = '/home/zml/dataset/项目/测试数据/测试模板/测试缩放/zhengyuqi/532_dlib_crop'
    for img in sorted(os.listdir(base_dir)):
        img = cv2.imread(os.path.join(base_dir, img))
        print(hp.get_head_pose(img))
