# -*- coding: utf-8 -*-
"""feature —— R045(T3:GPT-SoVITS/qftts 谱系)。

源件:modules/qftts/feature.cp310-win_amd64.pyd
      sha256 a2b49effa853c49a...(见 contracts/modules/qftts__feature.json)
对账材料:
  * 上游 zipvoice utils/feature.py(szr2026_out/bindiff/zipvoice_upstream/
    zipvoice/utils/feature.py)——本模块逐行同源(含 VocosFbank/VocosFbankConfig);
  * E1 探针 probe_e1/e2/e5/e6/e7/e8(VM oracle 实测)。

二进制/E1 出处标注:
  [E1-e2] 模块名表 {FeatureExtractor, Seconds, Union, VocosFbank,
      VocosFbankConfig, compute_num_frames, dataclass, np, register_extractor,
      torch, torchaudio};Union is typing.Union = True;dataclass is
      dataclasses.dataclass = True;Seconds 是 float 的子类。
  [E1-e2] register_extractor 签名 "(cls)",无默认值;__doc__ 与上游 lhotse
      装饰器逐字一致;register_extractor() → TypeError("missing 1 required
      positional argument: 'cls'");register_extractor(x, 1) → TypeError
      ("takes 1 positional argument but 2 were given");register_extractor(cls=x)
      可用;未知关键字 → TypeError;对没有 name 的类 → AttributeError
      ("type object '_X' has no attribute 'name'")。
  [E1-e6/e8] compute_num_frames 只依赖 duration/frame_shift(与 lhotse 的
      compute_num_frames 语义一致:采样率不参与运算!):
        duration=0 → 0(提前返回);
        frame_shift=0 → ZeroDivisionError("integer division or modulo by zero");
        int(duration*sr/fs) 的旧猜测被实测否证(1.0/0.01/2**40 得 100 而非天文数字);
        实测表:{1.0,0.01,16000}→100、{1.0,0.02,16000}→50、{3.0,0.008,16000}→375、
        {0.5,0.008,16000}→63(round-half-even)、{1.0,256,22050}→0、
        {1.0,3.0,16000}→0、{1e-9,0.01,16000}→0、{1.0,1e-7,1}→ZeroDivisionError。
  [E1-e5/e6] VocosFbankConfig 是 @dataclass:类属性 sampling_rate=24000、
      n_mels=100、n_fft=1024、hop_length=256;签名 (self) -> None;
      位置参数被拒("takes 1 positional argument but 3 were given");
      __dataclass_params__ = _DataclassParams(init=True,repr=True,eq=True,
      order=False,unsafe_hash=False,frozen=False)。
  [E1-e5] VocosFbank:name="VocosFbank"、config_type=VocosFbankConfig、
      __init__ 构造 config=VocoosFbankConfig, num_channels=1,
      fbank=torchaudio.transforms.MelSpectrogram(sample_rate=24000,n_fft=1024,
      hop_length=256,n_mels=100,center=True,power=1);feature_dim(sr)=n_mels=100;
      frame_shift=hop_length/sampling_rate=0.010666666666666666;
      extract 采样率不符 → AssertionError("Mismatched sampling rate: extractor
      expects 24000, got 32000");24000 样本 → (94,100)。
"""
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
import torchaudio

try:  # lhotse 与 oracle 运行时同源;缺失时退化为本文件内的等价实现
    from lhotse.features.base import FeatureExtractor, register_extractor
    from lhotse.utils import Seconds, compute_num_frames
except ImportError:  # pragma: no cover - 仅在无 lhotse 的环境触发
    Seconds = float

    def compute_num_frames(duration, frame_shift, sampling_rate) -> int:
        # [E1-e6/e8] 与 lhotse.utils.compute_num_frames 同语义。
        if duration == 0:
            return 0
        return round(duration / frame_shift)

    class FeatureExtractor:
        pass

    def register_extractor(cls):
        # [E1-e2] lhotse 装饰器行为:要求 cls.name,并回填 cls.__module__。
        _ = cls.name
        return cls


@dataclass
class VocosFbankConfig:
    # [E1-e5/e6] 类属性即默认值(24000/100/1024/256),逐值实测。
    sampling_rate: int = 24000
    n_mels: int = 100
    n_fft: int = 1024
    hop_length: int = 256


@register_extractor
class VocosFbank(FeatureExtractor):

    name = "VocosFbank"
    config_type = VocosFbankConfig

    def __init__(self, num_channels: int = 1):
        config = VocosFbankConfig
        super().__init__(config=config)
        assert num_channels in (1, 2)
        self.num_channels = num_channels
        self.fbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sampling_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            center=True,
            power=1,
        )

    def _feature_fn(self, sample):
        mel = self.fbank(sample)
        logmel = mel.clamp(min=1e-7).log()

        return logmel

    @property
    def device(self) -> Union[str, torch.device]:
        return self.config.device

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.n_mels

    def extract(
        self,
        samples: Union[np.ndarray, torch.Tensor],
        sampling_rate: int,
    ) -> Union[np.ndarray, torch.Tensor]:
        # Check for sampling rate compatibility.
        expected_sr = self.config.sampling_rate
        assert sampling_rate == expected_sr, (
            f"Mismatched sampling rate: extractor expects {expected_sr}, "
            f"got {sampling_rate}"
        )
        is_numpy = False
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
            is_numpy = True

        if len(samples.shape) == 1:
            samples = samples.unsqueeze(0)
        else:
            assert samples.ndim == 2, samples.shape

        if self.num_channels == 1:
            if samples.shape[0] == 2:
                samples = samples.mean(dim=0, keepdims=True)
        else:
            assert samples.shape[0] == 2, samples.shape

        mel = self._feature_fn(samples)
        # (1, n_mels, time) or (2, n_mels, time)
        mel = mel.reshape(-1, mel.shape[-1]).t()
        # (time, n_mels) or (time, 2 * n_mels)

        num_frames = compute_num_frames(
            samples.shape[1] / sampling_rate, self.frame_shift, sampling_rate
        )

        if mel.shape[0] > num_frames:
            mel = mel[:num_frames]
        elif mel.shape[0] < num_frames:
            mel = mel.unsqueeze(0)
            mel = torch.nn.functional.pad(
                mel, (0, 0, 0, num_frames - mel.shape[1]), mode="replicate"
            ).squeeze(0)

        if is_numpy:
            return mel.cpu().numpy()
        else:
            return mel

    @property
    def frame_shift(self) -> Seconds:
        return self.config.hop_length / self.config.sampling_rate
