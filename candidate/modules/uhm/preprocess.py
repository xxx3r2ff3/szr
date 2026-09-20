# -*- coding: utf-8 -*-
"""uhm.preprocess —— T4R 私有语义重建(R023)。

证据(oracle 侧只读实测,文件在共享目录 /Volumes/A/vm_transfer/ns_w2uhm/,
关键件抄入 evidence/modules/uhm__preprocess/runtime/):
  * P1 probe_surface_out.json   —— dir()/签名/co_varnames/co_firstlineno(每函数局部变量表)
  * P2 probe_behavior_out.json  —— 纯函数行为网格
  * P3 probe_behavior2_out.json —— 判别性输入(常数与截断/类型)
  * P4 probe_behavior3_out.json —— warp_imgs 返回结构、fast_pose_check 阈值形态
  * ST szr2026_out/pyd_static/uhm__preprocess/constants.txt(StringTab 槽位字符串)
每个函数体后的注释给出该函数全部行为证据的出处(探针点),不含任何猜测值。
"""
import os
import time
import warnings

import cv2
import numpy as np

warnings.filterwarnings('ignore')


def fast_bbox_expansion(x1, y1, x2, y2, w, h, expand_ratio=0.2):
    # P1 co_varnames=['x1','y1','x2','y2','w','h','expand_ratio','width','height',
    #                 'expansion_x','expansion_y','new_x1','new_y1','new_x2','new_y2']
    # P2/P3/P8 实测(int/float 返回类型逐位判别):
    #   new_x1 = max(0, int(x1-expansion_x)) 恒为 int;
    #   new_y1 = max(0, y1) 保持 y1 原类型(P3 y_keep_300: y1=300,h=200,r=0.5 → 300 而非 200;
    #     P8 all_float_noclamp y1=100.0 → 100.0 float);
    #   new_x2 = min(w, int(x2+expansion_x))、new_y2 = min(h, int(y2+expansion_y)):
    #     int() 在内层,P8 tie_x2_floatw(…,250.0,250.0,0.5) → 250.0(w 为 float 且取到 w),
    #     P3 frac → int(130.84)=130,less_x2_floatw → int 220。
    width = x2 - x1
    height = y2 - y1
    expansion_x = width * expand_ratio
    expansion_y = height * expand_ratio
    new_x1 = max(0, int(x1 - expansion_x))
    new_y1 = max(0, y1)
    new_x2 = min(w, int(x2 + expansion_x))
    new_y2 = min(h, int(y2 + expansion_y))
    return new_x1, new_y1, new_x2, new_y2


def fast_3dmm_bounds(xmin, ymin, w_rect, h_rect, img_w, img_h):
    # P1 co_varnames=['xmin','ymin','w_rect','h_rect','img_w','img_h','wh_ratio','x_c',
    #                 'half_width','height_factor','Xmin_3dmm','Xmax_3dmm','Ymin_3dmm','Ymax_3dmm']
    # P3 3dmm_* 全部 8 例:half_width 只随 h_rect(=0.8*h_rect),与 w_rect 无关;
    #   纵向中心 = ymin + 0.45*h_rect;半宽/半高相同 → 正方形框。wh_ratio 计算但无观测效应
    #   (P3 3dmm_w200_h100 / w400_h100 与 h_odd 系列交叉验证)。
    wh_ratio = w_rect / h_rect
    x_c = xmin + w_rect / 2
    # P8 int/float 判别:int() 在内层 → min(img_w, int(...));w/h 为 float 且更小时
    #   上界保持 float(3dmm_tie_float → Xmax 280.0,而 Ymax 仍为 int 387)。
    half_width = h_rect * 0.8
    height_factor = h_rect * 0.45
    Xmin_3dmm = max(0, int(x_c - half_width))
    Xmax_3dmm = min(img_w, int(x_c + half_width))
    Ymin_3dmm = max(0, int(ymin + height_factor - half_width))
    Ymax_3dmm = min(img_h, int(ymin + height_factor + half_width))
    return Xmin_3dmm, Xmax_3dmm, Ymin_3dmm, Ymax_3dmm


def fast_crop_bounds(xmin, ymin, w, img_w, img_h, wh_ratio_factor):
    # P1 co_varnames=['xmin','ymin','w','img_w','img_h','wh_ratio_factor','x_c',
    #                 'half_width','height_factor','Xmin','Xmax','Ymin','Ymax']
    # P2/P3 crop_* 全部 12 例:half_width = 0.75*w*r(crop 边长 1.5*w*r,方形),
    #   纵向中心 = ymin + 0.6*w*r(与 half_width 之比恒 0.8,r2/r05/r15_odd 交叉验证)。
    # P8 int/float 判别同 3dmm:min(img_w, int(...))(crop_tie_xmax_float → 1000.0 float)。
    # [I007 E2E 定谳 i007-fix-3] 本函数公式保持 W6-4 原样(模块级 12 例 golden 全过);
    #   Ymin 1px 差的修正点在 op.get_bounding_box 的**逐项 int 截断**(见该处注释),
    #   证据链:i007_truth6/i007_bbox2/i007_vfix7 探针 + 终轮 E2E 产物逐字节一致。
    x_c = xmin + w / 2
    half_width = w * 0.75 * wh_ratio_factor
    height_factor = w * 0.6 * wh_ratio_factor
    Xmin = max(0, int(x_c - half_width))
    Xmax = min(img_w, int(x_c + half_width))
    Ymin = max(0, int(ymin + height_factor - half_width))
    Ymax = min(img_h, int(ymin + height_factor + half_width))
    return Xmin, Xmax, Ymin, Ymax


def fast_landmark_transform(landmarks, xmin, ymin, scale_x, scale_y):
    # P1 co_varnames=['landmarks','xmin','ymin','scale_x','scale_y','result']
    # P2/P3 实测:np.array 复制(dtype 保持:int32 输入 → int32 输出且按 C 截断:
    #   lm_int_trunc (1,2)@int32 → 0.75→0;输入对象不被就地改写:lm_after 不变)。
    result = np.array(landmarks)
    result[:, 0] = (result[:, 0] - xmin) * scale_x
    result[:, 1] = (result[:, 1] - ymin) * scale_y
    return result


def fast_pose_check(head_poses, pose_threshold):
    # P1 co_varnames 仅两个形参(无局部变量)→ 单表达式向量化实现。
    # P4/P5/P8 实测:pose_threshold 必须是二维数组(取 [:,0]/[:,1]);
    #   1D/标量/list 报 "too many indices"/"not subscriptable"/"list indices ... tuple";
    #   P5 pc_np3x2_* 用 np.array([[-0.5,0.5]]*3) 得 True/False 与逐轴区间一致;
    #   广播错误信息顺序证明 head_poses 在比较式左侧(np6x2_np: (1,3) (6,));
    #   区间为**严格**不等式:P8 at_max/at_min(0.5 / -0.5,阈值 [-0.5,0.5])→ False,
    #   just_in(0.49999) → True;head_poses=None 时报 "'<' not supported" 亦指向 `<`/`>`。
    return np.all((head_poses > pose_threshold[:, 0]) & (head_poses < pose_threshold[:, 1]))


def warp_imgs(imgs_data):
    # P1 co_varnames=['imgs_data','idx','img'](无 result 局部 → 字典推导式)。
    # P4 warp_* 实测:返回 {idx: {'imgs_data': 元素, 'idx': idx}};输入含 dict 时元素取键
    #   (warp_dict {0: {'imgs_data': 'a', 'idx': 0}}),空输入 → {}。
    return {idx: {'imgs_data': img, 'idx': idx} for idx, img in enumerate(imgs_data)}



class op:
    """人脸预处理流水线(R023;W6-4 深度加固轮已按 oracle 实测修正)。

    W6-4 证据(evidence/vm_sessions/probe_w64{b,c,d,e,f}_out.json + 反汇编
    preprocess_v2/preprocess pyd):
      * __init__ 属性全表(vars(op)):pose_threshold 恒 [[-50,30],[-70,70],[-30,30]];
        wh_ratio_factor = 1.0/wh(位级:1/0.97=1.0309278350515465);target_size =
        img_size + 10*(img_size//256)(128/220→不变,256→266,300→310,384→394,512→532,
        probe_w64f 全中);target_size_float = float(target_size)。
      * get_top_faces 返回 int32(probe_w64b top1: 分数 2.5 → 2,坐标 50.7 → 50)。
      * get_bounding_box = cv2.boundingRect(int32 截断点云) + fast_crop_bounds +
        np.clip([Xmin,Xmax],0,w)/np.clip([Ymin,Ymax],0,h) + np.array([Ymin,Ymax,Xmin,Xmax],
        int32) + scale = target_size_float/(crop_w|crop_h) + fast_landmark_transform;
        41/41 网格样本逐位吻合(含"同 min/max 异输出"的 int32 截断效应)。
      * flow/flow_optimized/loc_detect_face 经 self.data_dict 协作:flow 建
        data_dict[idx]={'idx':idx,'no_face':True}(无人脸)并返回 (data_dict, no_face);
        loc_detect_face/flow_optimized 读 data_dict[idx]['imgs_data'](缺键 → KeyError)。
    """

    def __init__(self, wh=0.97, img_size=256, pose_check=True, double_face=False):
        # P1 co_varnames=['self','wh','img_size','pose_check','double_face',
        #                 'face_detect_path','head_path','FaceDetect','pfpld','Headpose',
        #                 'scrfd_detector','scrfd_predictor','hp']
        # W6-4 probe_w64b:实例属性含 pose_threshold/target_size/target_size_float/
        # wh_ratio_factor(co_varnames 只列局部变量,属性赋值不占位)
        self.wh = wh
        self.img_size = img_size
        self.pose_check = pose_check
        self.double_face = double_face
        # W6-4 probe_w64b:默认与自定义 wh/img_size 下取值恒定 → 常量表
        self.pose_threshold = np.array([[-50.0, 30.0], [-70.0, 70.0], [-30.0, 30.0]])
        # W6-4 probe_w64c/f:target_size = img_size + 10*(img_size//256)(10 点全中)
        self.target_size = img_size + 10 * (img_size // 256)
        self.target_size_float = float(self.target_size)
        # W6-4:1.0/wh 位级吻合(probe_w64b/c/f:0.97→1.0309278350515465,0.8→1.25,0.5→2.0)
        # [I007 E2E 定谳 i007-fix-3] E2E 实测 oracle attr = 1.0309278350515465
        # (=1/0.97;e2e/i007_products.uhm_chain.oracle.json facts),W6-4 位级记录正确,
        # 维持 1.0/(wh*1.0)。Ymin 1px 之谜的修正点在 get_bounding_box 的逐项 int 截断
        # (见 get_bounding_box 注释)。
        self.wh_ratio_factor = 1.0 / (wh * 1.0)
        face_detect_path = os.path.dirname(os.path.realpath(__file__))
        head_path = os.path.join(face_detect_path, 'face_detect_utils/resources/model_float32.onnx')
        from modules.uhm.face_detect_utils.face_detect import FaceDetect, pfpld
        from modules.uhm.face_detect_utils.head_pose import Headpose  # noqa: F401(oracle 同源导入)

        self.scrfd_detector = FaceDetect(model_path='./face_detect_utils/resources/')
        self.scrfd_predictor = pfpld(model_path='./face_detect_utils/resources')
        # W6-4:oracle Headpose 构造期打印 '.trtcache' 相对路径的 EP 文本且缺文件不抛
        # (probe_w64b);head_pose.py 属 R026,其构造文本/异常行为与 oracle 不一致且本轮
        # 不可改,故内联等价构造(真实 TRT 尝试)+ get_head_pose 惰性委托,登记偏差。
        class _Whenet:
            """oracle head_pose.Headpose 构造行为内联复刻(W6-4,登记偏差)。

            probe_w64b:缺文件不抛;构造文本 = TRT providers('.trtcache'
            相对字面量、键序 fp16→path→enable)。get_head_pose 惰性委托
            head_pose.Headpose(cpu=True)(构造期无 stdout)。
            """

            def __init__(self, onnx_path):
                import onnxruntime as ort
                providers = [
                    ('TensorrtExecutionProvider', {
                        'trt_fp16_enable': True,
                        'trt_engine_cache_path': '.trtcache',
                        'trt_engine_cache_enable': True,
                    }),
                    'CUDAExecutionProvider',
                ]
                try:
                    self.whenet_session = ort.InferenceSession(
                        onnx_path, providers=providers)
                except Exception:
                    self.whenet_session = None
                self._onnx_path = onnx_path
                self._impl = None

            def get_head_pose(self, image):
                if self._impl is None:
                    import modules.uhm.face_detect_utils.head_pose as _hp_mod
                    self._impl = _hp_mod.Headpose(cpu=True,
                                                  onnx_path=self._onnx_path)
                return self._impl.get_head_pose(image)

        self.hp = _Whenet(head_path)

    def get_top_faces(self, face_boxes, top_n=1):
        # P1 co_varnames=['self','face_boxes','top_n','widths','heights','areas',
        #                 'top_indices','top_faces']
        # W6-4 probe_w64b:oracle 返回 int32(分数 2.5→2 截断,坐标 50.7→50)
        widths = face_boxes[:, 2] - face_boxes[:, 0]
        heights = face_boxes[:, 3] - face_boxes[:, 1]
        areas = widths * heights
        # W6-4 probe_w64b top3:top_n=len 时 oracle 返回原行序([0,1,2]),
        # argpartition 的分块序需按索引排序(argpartition+np.sort 与 1/2/3 全例吻合)
        top_indices = np.sort(np.argpartition(areas, -top_n)[-top_n:])
        top_faces = face_boxes[top_indices]
        return top_faces.astype(np.int32)

    def get_order_face(self, face_boxes, top_n=2):
        # P1 co_varnames=['self','face_boxes','top_n','top2_faces','box1','box2',
        #                 'center_x1','center_x2'];W6-4:输出 int32、左→右排序
        top2_faces = self.get_top_faces(face_boxes, top_n)
        box1 = top2_faces[0]
        box2 = top2_faces[1]
        center_x1 = (box1[0] + box1[2]) / 2
        center_x2 = (box2[0] + box2[2]) / 2
        if center_x1 > center_x2:
            return np.stack([box2, box1])
        return np.stack([box1, box2])

    def get_bounding_box(self, landmarks, w, h):
        # P1 co_varnames=['self','landmarks','w','h','points','xmin','ymin','w_rect',
        #                 'h_rect','Xmin','Xmax','Ymin','Ymax','bounding_box','crop_w',
        #                 'crop_h','scale_x','scale_y','crop_lm']
        # W6-4 反汇编+41/41 网格(boundingRect 的 int32 截断解释了"同 min/max 异输出"):
        # [I007 E2E 定谳 i007-fix-9] 终轮变体裁决(i007_bboxf 探针,clip25 320×240
        #   真帧 ×3):oracle 边界组合 = **全浮点运算后逐项 int 截断**——
        #   Xmin/Xmax = int(x_c ∓ hw)、Ymin/Ymax = int(ymin+hf ∓ hw)
        #   (三帧 X_b/Y_b 全中);逐项 int 再加减的 fix-3 形在真帧上差 ±1 被证伪
        #   (该形仅与 face.png 单点数值巧合)。不委托 fast_crop_bounds(模块级
        #   函数保持 W6-4 原式,12 例 golden 全过)。oracle 真值:flow bb=
        #   [29,121,95,187],浮点中间量 xmin=112/ymin=39/w=59/h=46。
        points = np.ascontiguousarray(landmarks)
        xmin, ymin, w_rect, h_rect = cv2.boundingRect(points.astype(np.int32))
        x_c = xmin + w_rect / 2
        half_width = w_rect * 0.75 * self.wh_ratio_factor
        height_factor = w_rect * 0.6 * self.wh_ratio_factor
        Xmin = max(0, int(x_c - half_width))
        Xmax = min(w, int(x_c + half_width))
        Ymin = max(0, int(ymin + height_factor - half_width))
        Ymax = min(h, int(ymin + height_factor + half_width))
        Xmin, Xmax = np.clip([Xmin, Xmax], 0, w)
        Ymin, Ymax = np.clip([Ymin, Ymax], 0, h)
        bounding_box = np.array([Ymin, Ymax, Xmin, Xmax], dtype=np.int32)
        crop_w = Xmax - Xmin
        crop_h = Ymax - Ymin
        scale_x = self.target_size_float / crop_w
        scale_y = self.target_size_float / crop_h
        crop_lm = fast_landmark_transform(points, Xmin, Ymin, scale_x, scale_y)
        return bounding_box, crop_lm

    def loc_detect_face(self, idx_batch):
        # P1 co_varnames=['self','idx_batch','results','idx','loc_dict','img','h','w',
        #                 'face_boxes','_','i','face_box','x1','y1','x2','y2','score',
        #                 'is_valid_bounds','face_img','pots','landmarks','bounding_box',
        #                 'crop_lm','xmin','ymin','w_rect','h_rect','Xmin_3dmm','Xmax_3dmm',
        #                 'Ymin_3dmm','Ymax_3dmm','head_poses']
        # W6-4 probe_w64b/c:line182-184 = data_dict[idx] → ['imgs_data'] → img.shape;
        #   缺 'imgs_data' 键 → KeyError(probe_w64c loc_after_flow 实测)
        # [I007 E2E 定谳 i007-fix-1] 带脸条目为"原地增补"结构(oracle 真值探针
        #   #3/#4/#5/#6,evidence/integration/i007_drivers/i007_probe_truth{3,4,5,6}.py,
        #   共享目录 i007_truth{3,4,5,6}_console.log):
        #   * face_boxes = get_top_faces(全部检出,1)(int32 (1,5),面积最大脸);
        #   * 送 pfpld 的裁剪 = fast_bbox_expansion(top1, 0.2)(new_y1 不上扩);
        #   * 关键点平移回全图坐标后**以 int32** 进 get_bounding_box:
        #     bounding_box/crop_lm 双 sha256 逐字节命中 ee340245…/e62290ec…;
        #   * head_poses 在人脸裁剪上计算、经 fast_pose_check 门控且**不落条目**
        #     (oracle pose_check=True/False 条目全等;全图姿态 -88.59° 越界而
        #     条目仍落盘 → 计算面必为裁剪图);
        #   * 无脸条目加 'no_face' 键且 idx 不进 results(oracle loc_noface 实测);
        #   * results 为新 dict,条目对象与 data_dict 共享(原地增补)。
        results = {}
        for idx in idx_batch:
            loc_dict = self.data_dict[idx]
            img = loc_dict['imgs_data']
            h, w = img.shape[:2]
            face_boxes, _ = self.scrfd_detector.get_bboxes(img)
            if face_boxes is None or len(face_boxes) == 0:
                loc_dict['no_face'] = True
                continue
            top = self.get_top_faces(face_boxes, 1)
            x1, y1, x2, y2 = [int(v) for v in top[0][:4]]
            ex1, ey1, ex2, ey2 = fast_bbox_expansion(x1, y1, x2, y2, w, h, 0.2)
            face_img = img[ey1:ey2, ex1:ex2]
            if face_img.size == 0:
                continue
            pots = self.scrfd_predictor.forward(face_img)
            points = (pots[0] + np.array([float(ex1), float(ey1)])).astype(np.int32)
            bounding_box, crop_lm = self.get_bounding_box(points, w, h)
            if self.pose_check:
                head_poses = self.hp.get_head_pose(face_img)
                if not fast_pose_check(head_poses, self.pose_threshold):
                    continue
            loc_dict['face_boxes'] = top
            loc_dict['bounding_box'] = bounding_box
            loc_dict['crop_lm'] = crop_lm
            results[idx] = loc_dict
        return results

    def flow_optimized(self):
        # P1 co_varnames=['self','keys','detection_results']
        # W6-4 probe_w64c:line253 读 self.data_dict(缺属性 → AttributeError),
        #   line256 逐键调 self.loc_detect_face(键缺 'imgs_data' → KeyError)
        detection_results = self.data_dict
        keys = list(detection_results.keys())
        for idx in keys:
            self.loc_detect_face([idx])
        return detection_results

    def flow(self, caped_img):
        # P1 co_varnames=['self','caped_img','no_face','idx','item']
        # W6-4 probe_w64c:flow([img]) → (data_dict, no_face);无人脸条目 =
        #   {'idx': idx, 'no_face': True};data_dict 挂到 self(后续
        #   loc_detect_face/flow_optimized 从 data_dict 取 'imgs_data')
        # [I007 E2E 定谳 i007-fix-1] 带脸条目为完整检测结果(oracle 真值探针 #3/#5):
        #   {'idx', 'face_boxes'(int32 (1,5)), 'bounding_box'(int32 (4)),
        #    'crop_lm'(int32 (68,2))} —— **不含 imgs_data**(oracle 实测键集,
        #    后续 loc_detect_face 对 flow 产物取 'imgs_data' 即 KeyError);
        #   数值管线与 loc_detect_face 同(扩张框 + pfpld + 平移 int32 +
        #   get_bounding_box),且 flow 无 pose 门控(co_varnames 无 head_poses,
        #   oracle pose_check=True 的 flow 条目同样不含量化姿态)。
        self.data_dict = {}
        no_face = []
        for idx, item in enumerate(caped_img):
            face_boxes, _ = self.scrfd_detector.get_bboxes(item)
            if face_boxes is None or len(face_boxes) == 0:
                self.data_dict[idx] = {'idx': idx, 'no_face': True}
                no_face.append(idx)
                continue
            h, w = item.shape[:2]
            top = self.get_top_faces(face_boxes, 1)
            x1, y1, x2, y2 = [int(v) for v in top[0][:4]]
            ex1, ey1, ex2, ey2 = fast_bbox_expansion(x1, y1, x2, y2, w, h, 0.2)
            face_img = item[ey1:ey2, ex1:ex2]
            if face_img.size == 0:
                self.data_dict[idx] = {'idx': idx, 'no_face': True}
                no_face.append(idx)
                continue
            pots = self.scrfd_predictor.forward(face_img)
            points = (pots[0] + np.array([float(ex1), float(ey1)])).astype(np.int32)
            bounding_box, crop_lm = self.get_bounding_box(points, w, h)
            self.data_dict[idx] = {'idx': idx, 'face_boxes': top,
                                   'bounding_box': bounding_box, 'crop_lm': crop_lm}
        return self.data_dict, no_face
