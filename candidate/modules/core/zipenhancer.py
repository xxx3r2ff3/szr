# -*- coding: utf-8 -*-
"""zipenhancer —— R012 重建(源 modules/core/zipenhancer.cp310-win_amd64.pyd,T4R)。

证据标签: [pyd 0x…] Ghidra 反编译函数入口(evidence/modules/core__zipenhancer/
static/decomp/out/)、[pyd 常量] 串表字面量(…/constants.txt)、
[E1 zip-e1] VM 内 oracle 探针(runtime/probe_zip_e1.py + _out.json)。

E1 结论(runtime/probe_zip_e1_out.json):
* 模块命名空间 9 名:Optional / Union / pipeline / Tasks / ZipEnhancer / os /
  tempfile / torch / torchaudio(无其它公开名);
* `Optional` = typing.Optional,`Union` = typing.Union(调用即
  `TypeError: Cannot instantiate typing.Optional|Union`;下标可用:
  `Union[int, str]` → typing.Union[int, str]);
* `pipeline` = modelscope.pipelines.builder.pipeline;无参调用 →
  `ValueError: task or pipeline_name is required`(本地判定,不联网);
  未知 task → KeyError(取 modelscope 默认流水线表);
* `Tasks` = modelscope.utils.constant.Tasks(bases = CVTasks/NLTasks/AudioTasks/
  MultiModalTasks/ScienceTasks/Other);
* ZipEnhancer 三方法签名(动态探测):
  __init__(self, model_path='iic/speech_zipenhancer_ans_multiloss_16k_base')、
  _normalize_loudness(self, wav_path)、enhance(self, input_path,
  output_path=None, normalize_loudness=True) -> str。

覆盖单元(契约):Optional / Union / pipeline 三个模块级 import 名(各 B0)。
ZipEnhancer 类方法体按 [pyd 0x180001080/0x180001750/0x1800027a0] 反编译 +
串表注释还原;本环境不做在线模型加载,故不构造实例(enhance 需要 modelscope
在线/本地模型,登记为未定谳项,见报告)。
"""
import os
import tempfile
from typing import Optional, Union

import torch
import torchaudio
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class ZipEnhancer:
    """ZipEnhancer Audio Denoising Enhancer [pyd 常量:类 docstring 原文]"""

    def __init__(self, model_path: str = 'iic/speech_zipenhancer_ans_multiloss_16k_base'):
        """Initialize ZipEnhancer

        Args:
            model_path: ModelScope model path or local path
        """
        # [pyd 0x180001080] 唯一实参 model_path(默认值取自串表并作为函数默认);
        # 构造函数内建 modelscope 流水线实例,存于 self._pipeline(串表名)。
        self._pipeline = pipeline(Tasks.acoustic_noise_suppression, model=model_path)

    def _normalize_loudness(self, wav_path: str):
        """Audio loudness normalization

        Args:
            wav_path: Audio file path
        """
        # [pyd 0x180001750] torchaudio.load → 峰值归一(串表: audio/gain/audio/
        # normalized_audio/loudness/save 顺序)→ torchaudio.save 回写同一路径。
        audio, sample_rate = torchaudio.load(wav_path)
        gain = 1.0 / (audio.abs().max() + 1e-8)
        normalized_audio = audio * gain
        torchaudio.save(wav_path, normalized_audio, sample_rate)
        return normalized_audio

    def enhance(self, input_path: str, output_path: Optional[str] = None,
                normalize_loudness: bool = True) -> str:
        """Audio denoising enhancement

        Args:
            input_path: Input audio file path
            output_path: Output audio file path (optional, creates temp file by default)
            normalize_loudness: Whether to perform loudness normalization

        Returns:
            str: Output audio file path

        Raises:
            RuntimeError: If pipeline is not initialized or processing fails
        """
        # [pyd 0x1800027a0] 串表顺序: exists → FileNotFoundError('Input audio file
        # does not exist: ' + path) → tempfile.NamedTemporaryFile(suffix='.wav') →
        # self._pipeline(...) → normalize_loudness 分支 → return output_path;
        # 异常统一包成 RuntimeError('Audio denoising processing failed: ' + e)。
        if not os.path.exists(input_path):
            raise FileNotFoundError('Input audio file does not exist: ' + input_path)
        try:
            if output_path is None:
                tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                output_path = tmp_file.name
            if self._pipeline is None:
                raise RuntimeError('pipeline is not initialized')
            self._pipeline(input_path, output_path)
            if normalize_loudness:
                self._normalize_loudness(output_path)
            return output_path
        except Exception as exc:
            raise RuntimeError('Audio denoising processing failed: ' + str(exc))
