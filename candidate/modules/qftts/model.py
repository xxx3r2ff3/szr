# -*- coding: utf-8 -*-
"""model —— R047(T3:GPT-SoVITS/qftts 谱系 ZipVoice 模型装配)。

源件:modules/qftts/model.cp310-win_amd64.pyd
      sha256 0f37625eda4b24ad...(见 contracts/modules/qftts__model.json)
对账材料:
  * 上游 zipvoice models/zipvoice.py + models/zipvoice_distill.py
    (szr2026_out/bindiff/zipvoice_upstream)——本模块同源;
  * E1 探针 probe_e2 "surface" 节点(VM oracle 实测的模块面与符号来源)。

二进制/E1 出处标注:
  [E1-e2] 修正后的 app_sys_path 下 `import modules.qftts.model` **成功**;
      模块名表实测:{DDP, DistillEulerSolver, EulerSolver, List, Optional,
      TTSZipformer, ZipVoice, ZipVoiceDistill, condition_time_mask,
      get_tokens_index, make_pad_mask, nn, pad_labels,
      prepare_avg_tokens_durations, torch}。
  [E1-e2] 导入期新增 sys.modules = {model, modules.qftts.model, scaling,
      zipformer} → 证明本模块导入 `zipformer`(其内部再导入 `scaling`)与
      `solver`;发现期"ImportError: cannot import name condition_time_mask"
      属于当时的 sys.path 形态问题(缺 modules/qftts 入路径)。
  [E1-e2] 符号来源实测:DDP→torch.nn.parallel.distributed、
      TTSZipformer→zipformer、EulerSolver/DistillEulerSolver→solver、
      condition_time_mask/get_tokens_index/make_pad_mask/pad_labels/
      prepare_avg_tokens_durations→utils(common)。
      注意 R047 契约的覆盖单元仅为 `<module>::B1`(模块级导入块),
      函数/类体不在本任务覆盖分母内。
"""
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# [E1-e2] 兄弟包绝对导入(oracle 导入期行为):solver 用同树候选,
# zipformer/scaling 同;utils 五函数由 R053 交付的同树候选 utils.py 顶层提供
# (E1 定谳:安装名 utils,__module__=="utils",无 utils.common 子模块)。
from solver import DistillEulerSolver, EulerSolver
from zipformer import TTSZipformer
from utils import (
    condition_time_mask,
    get_tokens_index,
    make_pad_mask,
    pad_labels,
    prepare_avg_tokens_durations,
)


class ZipVoice(nn.Module):
    """The ZipVoice model."""

    def __init__(
        self,
        fm_decoder_downsampling_factor: List[int] = [1, 2, 4, 2, 1],
        fm_decoder_num_layers: int = 1,
        fm_decoder_cnn_module_kernel: int = 31,
        fm_decoder_dim: int = 512,
        fm_decoder_feedforward_dim: int = 1024,
        fm_decoder_num_heads: int = 4,
        text_encoder_dim: int = 192,
        text_encoder_num_layers: int = 4,
        text_encoder_num_heads: int = 4,
        text_encoder_feedforward_dim: int = 512,
        text_encoder_cnn_module_kernel: int = 31,
        text_encoder_downsampling_factor: int = 2,
        text_encoder_num_mels: int = 100,
        feat_dim: int = 100,
        vocab_size: int = 400,
        use_half: bool = False,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.text_encoder = TTSZipformer(
            output_downsampling_factor=text_encoder_downsampling_factor,
            downsampling_factor=(1,) * text_encoder_num_layers,
            num_encoder_layers=text_encoder_num_layers,
            encoder_dim=text_encoder_dim,
            num_heads=text_encoder_num_heads,
            feedforward_dim=text_encoder_feedforward_dim,
            cnn_module_kernel=text_encoder_cnn_module_kernel,
            num_mels=text_encoder_num_mels,
            vocab_size=vocab_size,
            use_half=use_half,
        )
        self.fm_decoder = TTSZipformer(
            output_downsampling_factor=1,
            downsampling_factor=tuple(fm_decoder_downsampling_factor),
            num_encoder_layers=fm_decoder_num_layers,
            encoder_dim=fm_decoder_dim,
            num_heads=fm_decoder_num_heads,
            feedforward_dim=fm_decoder_feedforward_dim,
            cnn_module_kernel=fm_decoder_cnn_module_kernel,
            num_mels=feat_dim,
            vocab_size=vocab_size,
            use_half=use_half,
        )

    def forward_text_encoder(
        self,
        tokens: torch.Tensor,
        tokens_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Forward the text encoder."""
        text_encoder_out, _ = self.text_encoder(
            tokens, tokens_lens, return_before_output_layer=True
        )
        return text_encoder_out

    def forward_fm_decoder(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        """Forward the flow-matching decoder."""
        x = torch.cat([xt, speech_condition], dim=-1)
        x, _ = self.fm_decoder(x, padding_mask=padding_mask)

        return x


class ZipVoiceDistill(ZipVoice):
    """ZipVoice-Distill model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        required_params = {
            "feat_dim",
            "fm_decoder_downsampling_factor",
            "fm_decoder_num_layers",
            "fm_decoder_cnn_module_kernel",
            "fm_decoder_dim",
            "fm_decoder_feedforward_dim",
            "fm_decoder_num_heads",
            "text_encoder_dim",
        }
        missing = required_params - set(kwargs)
        if missing:
            raise ValueError("Missing required parameters: %s" % sorted(missing))
