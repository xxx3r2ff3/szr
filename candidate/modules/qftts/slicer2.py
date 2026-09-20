# -*- coding: utf-8 -*-
"""slicer2 —— R050(T3:GPT-SoVITS slicer2 谱系,含本地改动)。

源件:modules/qftts/slicer2.cp310-win_amd64.pyd
      sha256 d673aac863ff63d0...(见 contracts/modules/qftts__slicer2.json)
对账材料:
  * 参考明文 `/Volumes/A/数字人/szr2026/modules/qftts/slicer2.py`(同版本明文);
  * 上游 GPT-SoVITS tools/slicer2.py(szr2026_out/bindiff/gptsovits_upstream);
  * E1 探针 probe_e1/e2/e3/e9/e11/e12/e13/e14/e15(VM oracle 实测)。

二进制/E1 出处标注:
  [E1-e2]  get_rms 默认值 (2048, 512, 'constant');签名
           "(y, frame_length=2048, hop_length=512, pad_mode='constant')";
           pad_mode='nope' → ValueError("mode 'nope' is not supported");
           无 axis/center 关键字 → TypeError;
           const 0.5 输入首帧 0.35355338…(与明文 librosa 实现逐位一致)。
  [E1-e3]  monkeypatch 模块级 get_rms 注入已知 rms 序列后观测 slice 分段决策。
  [E1-e15] 尾静音/首静音/多静音逐帧明细。
  [E1-e14] 反推实现与 oracle 对照:合成 rms 序列 33/34、真实波形 8/8 一致。

本地改动(相对上游明文,证据见上):
  1. `get_rms` 与上游逐位一致(未改);
  2. `slice` 的静音切点改为"区间端点"而非"区间内 argmin":
     上游 mid/long 分支在静音窗口内取 argmin 决定保留位置,本地改为
     pos = max(clip_start, silence_start, i - max_sil_kept) 的端点式切点;
  3. 返回三元组 [chunk_waveform, begin_ms, end_ms](上游同形),但
     "太短直接返回" 分支返回 [waveform, 0, int(samples/hop)*hop];
  4. 首段长静音(d > 2*max_sil_kept 且 silence_start == 0)整段丢弃;
  5. 无切点时返回单段 [waveform, 0, total_frames*hop]。
"""
import numpy as np


# This function is obtained from librosa.
def get_rms(
    y,
    frame_length=2048,
    hop_length=512,
    pad_mode="constant",
):
    # [E1-e2] 与上游明文逐行一致;默认值与关键字名由 oracle signature 复核。
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    # put our new within-frame axis at the end for now
    out_strides = y.strides + tuple([y.strides[axis]])
    # Reduce the shape on the framing axis
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    # Calculate power
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)

    return np.sqrt(power)


class Slicer:
    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 5000,
        min_interval: int = 300,
        hop_size: int = 20,
        max_sil_kept: int = 5000,
    ):
        # [E1-e2] 错误文案逐字一致(两条 ValueError)。
        if not min_length >= min_interval >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: min_length >= min_interval >= hop_size"
            )
        if not max_sil_kept >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: max_sil_kept >= hop_size"
            )
        # [E1-e2] sr=32000 默认:threshold=0.01 hop_size=640 win_size=2560
        #         min_length=250 min_interval=15 max_sil_kept=250(逐值实测)。
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size : min(waveform.shape[1], end * self.hop_size)]
        else:
            return waveform[begin * self.hop_size : min(waveform.shape[0], end * self.hop_size)]

    # @timeit
    def slice(self, waveform):
        # [E1-e13] 输入 <= min_length 样本时直接整段返回(带帧跨度)。
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if samples.shape[0] <= self.min_length:
            return [[waveform, 0, int(samples.shape[0] / self.hop_size) * self.hop_size]]
        rms_list = get_rms(
            y=samples, frame_length=self.win_size, hop_length=self.hop_size
        ).squeeze(0)
        # [E1-e3/e9/e12/e14/e15] 本地切点:sil_tags 记录 (remove_begin, remove_end)。
        sil_tags = []
        silence_start = None
        clip_start = 0
        drop_leading = False
        trailing_start = -1
        for i, rms in enumerate(rms_list):
            # Keep looping while frame is silent.
            if rms < self.threshold:
                # Record start of silent frames.
                if silence_start is None:
                    silence_start = i
                continue
            # Keep looping while frame is not silent and silence start has not been recorded.
            if silence_start is None:
                continue
            # Clear recorded silence start if interval is not enough or clip is too short
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (
                i - silence_start >= self.min_interval
                and i - clip_start >= self.min_length
            )
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            # [E1-e14] 本地改动 2:端点式切点(非 argmin)。
            pos = max(clip_start, silence_start, i - self.max_sil_kept)
            if silence_start == 0 and i - silence_start > self.max_sil_kept * 2:
                # [E1-e15] 本地改动 4:首段长静音整段丢弃,不产生切点。
                drop_leading = True
            else:
                sil_tags.append((silence_start, pos))
            clip_start = pos
            silence_start = None
        # Deal with trailing silence.
        total_frames = rms_list.shape[0]
        trailing = False
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            sil_tags.append(
                (silence_start, min(total_frames, silence_start + self.max_sil_kept))
            )
            trailing = True
            trailing_start = silence_start
        # Apply and return slices.
        ####音频+起始时间+终止时间
        chunks = []
        if not sil_tags:
            # [E1-e2/e13] 无切点:整段(帧跨度向上取整到 hop 的整数倍)。
            chunks.append(
                [waveform, int(clip_start * self.hop_size), int(total_frames * self.hop_size)]
            )
        else:
            if not drop_leading and sil_tags[0][0] > 0:
                chunks.append(
                    [
                        self._apply_slice(waveform, 0, sil_tags[0][0]),
                        0,
                        int(sil_tags[0][0] * self.hop_size),
                    ]
                )
            for i in range(len(sil_tags) - 1):
                chunks.append(
                    [
                        self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]),
                        int(sil_tags[i][1] * self.hop_size),
                        int(sil_tags[i + 1][0] * self.hop_size),
                    ]
                )
            _trailing_capped = (
                trailing
                and sil_tags
                and sil_tags[-1][1] == min(total_frames, trailing_start + self.max_sil_kept)
            )
            if sil_tags[-1][1] < total_frames and not _trailing_capped:
                chunks.append(
                    [
                        self._apply_slice(waveform, sil_tags[-1][1], total_frames),
                        int(sil_tags[-1][1] * self.hop_size),
                        int(total_frames * self.hop_size),
                    ]
                )
        # [E1-e15] 空区间(长度 <= 0)不产出;全静音 → []。
        return [chunk for chunk in chunks if chunk[2] > chunk[1]]


def main():
    import os.path
    from argparse import ArgumentParser

    import librosa
    import soundfile

    parser = ArgumentParser()
    parser.add_argument("audio", type=str, help="The audio to be sliced")
    parser.add_argument("--out", type=str, help="Output directory of the sliced audio clips")
    parser.add_argument(
        "--db_thresh",
        type=float,
        required=False,
        default=-40,
        help="The dB threshold for silence detection",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        required=False,
        default=5000,
        help="The minimum milliseconds required for each sliced audio clip",
    )
    parser.add_argument(
        "--min_interval",
        type=int,
        required=False,
        default=300,
        help="The minimum milliseconds for a silence part to be sliced",
    )
    parser.add_argument(
        "--hop_size",
        type=int,
        required=False,
        default=10,
        help="Frame length in milliseconds",
    )
    parser.add_argument(
        "--max_sil_kept",
        type=int,
        required=False,
        default=500,
        help="The maximum silence length kept around the sliced clip, presented in milliseconds",
    )
    args = parser.parse_args()
    out = args.out
    if out is None:
        out = os.path.dirname(os.path.abspath(args.audio))
    audio, sr = librosa.load(args.audio, sr=None, mono=False)
    slicer = Slicer(
        sr=sr,
        threshold=args.db_thresh,
        min_length=args.min_length,
        min_interval=args.min_interval,
        hop_size=args.hop_size,
        max_sil_kept=args.max_sil_kept,
    )
    chunks = slicer.slice(audio)
    if not os.path.exists(out):
        os.makedirs(out)
    for i, chunk in enumerate(chunks):
        if len(chunk.shape) > 1:
            chunk = chunk.T
        soundfile.write(
            os.path.join(
                out,
                "%s_%d.wav" % (os.path.basename(args.audio).rsplit(".", maxsplit=1)[0], i),
            ),
            chunk,
            sr,
        )


if __name__ == "__main__":
    main()
