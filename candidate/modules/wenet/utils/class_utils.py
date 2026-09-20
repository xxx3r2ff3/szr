#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-11-28] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>
import torch
from torch.nn import BatchNorm1d, LayerNorm

from wenet.transformer.attention import (MultiHeadedAttention,
                                         MultiHeadedCrossAttention,
                                         RelPositionMultiHeadedAttention,
                                         RopeMultiHeadedAttention,
                                         ShawRelPositionMultiHeadedAttention)
from wenet.transformer.embedding import (
    LearnablePositionalEncoding, NoPositionalEncoding, PositionalEncoding,
    RelPositionalEncoding, RopePositionalEncoding, WhisperPositionalEncoding)
from wenet.transformer.norm import RMSNorm
from wenet.transformer.positionwise_feed_forward import (
    GatedVariantsMLP, MoEFFNLayer, PositionwiseFeedForward)
from wenet.transformer.subsampling import (
    Conv1dSubsampling2, Conv2dSubsampling4, Conv2dSubsampling6,
    Conv2dSubsampling8, EmbedinigNoSubsampling, LinearNoSubsampling,
    StackNFramesSubsampling)
from wenet.transformer.swish import Swish

WENET_ACTIVATION_CLASSES = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "swish": getattr(torch.nn, "SiLU", Swish),
    "gelu": torch.nn.GELU,
}

WENET_RNN_CLASSES = {
    "rnn": torch.nn.RNN,
    "lstm": torch.nn.LSTM,
    "gru": torch.nn.GRU,
}

WENET_SUBSAMPLE_CLASSES = {
    "linear": LinearNoSubsampling,
    "embed": EmbedinigNoSubsampling,
    "conv1d2": Conv1dSubsampling2,
    "conv2d": Conv2dSubsampling4,
    "conv2d6": Conv2dSubsampling6,
    "conv2d8": Conv2dSubsampling8,
    'paraformer_dummy': torch.nn.Identity,
    'stack_n_frames': StackNFramesSubsampling
}

WENET_EMB_CLASSES = {
    "embed": PositionalEncoding,
    "abs_pos": PositionalEncoding,
    "rel_pos": RelPositionalEncoding,
    "no_pos": NoPositionalEncoding,
    "abs_pos_whisper": WhisperPositionalEncoding,
    "embed_learnable_pe": LearnablePositionalEncoding,
    'rope_pos': RopePositionalEncoding
}

WENET_ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
    "crossattn": MultiHeadedCrossAttention,
    'shaw_rel_selfattn': ShawRelPositionMultiHeadedAttention,
    'rope_abs_selfattn': RopeMultiHeadedAttention
}

WENET_MLP_CLASSES = {
    'position_wise_feed_forward': PositionwiseFeedForward,
    'moe': MoEFFNLayer,
    'gated': GatedVariantsMLP
}

WENET_NORM_CLASSES = {
    'layer_norm': LayerNorm,
    'batch_norm': BatchNorm1d,
    'rms_norm': RMSNorm
}

# ---------------------------------------------------------------------------
# 重建出处与证据(R066)
#   目标 pyd : modules/wenet/utils/class_utils.cp310-win_amd64.pyd
#              sha256 = f21c8b1f3dc02124dc551321f5d9a01ed38526daf0aa51667bff4ce8e1457d9d
#   上游对齐 : wenet-e2e/wenet @ 6480b8fffe372a170989fd65f9b9bd97ff368df3
#              https://github.com/wenet-e2e/wenet/blob/6480b8fffe372a170989fd65f9b9bd97ff368df3/wenet/utils/class_utils.py
#              (结构/风格对齐;键集按 oracle E1 实测裁剪,见下"版本分歧")
#   E1 证据  : 共享目录 ns_wenet2/probe_surface.json、probe_detail.json
#              (探针脚本 probes/*.py,已抄入 evidence/modules/wenet__utils__class_utils/runtime/)
#              * dir(module) 实测 43 个公开名:7 个 WENET_*_CLASSES 字典 + 25 个类 +
#                LayerNorm/BatchNorm1d(torch)+Swish+RMSNorm+torch;
#              * 七个字典的"键序 + 值来源(module/qualname)"逐项实测:
#                ACT{hardtanh,tanh,relu,selu,swish→SiLU,gelu}
#                RNN{rnn,lstm,gru}
#                SUBSAMPLE{linear,embed,conv1d2,conv2d,conv2d6,conv2d8,paraformer_dummy→Identity,stack_n_frames}
#                EMB{embed,abs_pos,rel_pos,no_pos,abs_pos_whisper,embed_learnable_pe,rope_pos}
#                ATTENTION{selfattn,rel_selfattn,crossattn,shaw_rel_selfattn,rope_abs_selfattn}
#                MLP{position_wise_feed_forward,moe,gated}
#                NORM{layer_norm→torch.nn.LayerNorm,batch_norm→torch.nn.BatchNorm1d,rms_norm→RMSNorm}
#   E2 证据  : evidence/modules/wenet__utils__class_utils/static/{api_surface.md,imports.txt}
#              strings: layer_norm/batch_norm/rms_norm/selfattn/rel_selfattn/crossattn/
#              shaw_rel_selfattn/rope_abs_selfattn/rope_pos/paraformer_dummy/stack_n_frames/
#              abs_pos_whisper/embed_learnable_pe/position_wise_feed_forward/gated …
#  版本分歧 : oracle 的 class_utils 是"裁剪版"仓库快照 —— 上游 6480b8f 同一文件还含
#              efficient_conformer/squeezeformer/paraformer-embedding/firered 条目
#              (conv2d2、dwconv2d4、abs_pos_paraformer、grouped_rel_selfattn、firered_*),
#              但 oracle 串表与 dir() 均无这些符号(双源一致),故候选按实测键集裁剪,
#              不引入 oracle 不存在的导入(否则候选会 import 未随包发布的模块)。
#   约束     : 只依赖 torch 与同树候选源码(wenet.transformer.* 属其它 R 任务候选,
#              当前未落盘 → 本候选暂不可独立导入;用例采用门控导入路径,详见
#              reports/modules/wenet__utils__class_utils-impl.md)。
# ---------------------------------------------------------------------------
