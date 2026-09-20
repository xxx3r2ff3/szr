# -*- coding: utf-8 -*-
"""w2l.audio —— R017(语义重建自 modules/w2l/audio.cp310-win_amd64.pyd)。

源 pyd:modules/w2l/audio.cp310-win_amd64.pyd
        sha256 2a3c16b8e87fbfd301ee32e311c6712375beea38f1de9edd49e41bcbe62a132d
反编译证据:evidence/modules/w2l__audio/static/pseudocode/
  * w2l__audio__PyInit_audio__180010440.c / FUN_1800106f0(模块 exec):模块级
    语句顺序 = import librosa / import librosa.filters / import numpy as np /
    from scipy import signal / from scipy.io import wavfile /
    from modules.w2l.hparams import hparams as hp,随后按源序定义 19 个函数,
    并在 librosa_pad_lr 与 _linear_to_mel 之间有 `_mel_basis = None`
    (串表槽位 0x18001a218;`__name__`/`__main__` 只是 Cython pymod_exec 样板);
  * FUN_18000f590:Cython 为每函数登记的局部变量名表(源语句级顺序):
    load_wav(path,sr) save_wav(wav,path,sr) save_wavenet_wav(wav,path,sr)
    preemphasis(wav,k,preemphasize) inv_preemphasis(wav,k,inv_preemphasize)
    get_hop_size(hop_size) linearspectrogram(wav,D,S) melspectrogram(wav,D,S)
    _lws_processor(lws) _stft(y) num_frames(length,fsize,fshift,pad,M)
    pad_lr(M,x,fsize,fshift,pad,T,r) librosa_pad_lr(x,fsize,fshift)
    _linear_to_mel(spectogram) _build_mel_basis() _amp_to_db(x,min_level)
    _db_to_amp(x) _normalize(S) _denormalize(D);
  * 每个函数体一个 .c(文件名见各行内注释),内含 DAT_ → STR("…") 槽位解析。

E1 证据(VM oracle 侧 probe_e1_audio{,2,3,4}.py + _out.json,
归档 evidence/modules/w2l__audio/runtime/):
  * 命名空间 = {load_wav, save_wav, save_wavenet_wav, preemphasis, inv_preemphasis,
    get_hop_size, linearspectrogram, melspectrogram, _lws_processor, _stft,
    num_frames, pad_lr, librosa_pad_lr, _linear_to_mel, _build_mel_basis,
    _amp_to_db, _db_to_amp, _normalize, _denormalize} + {np, librosa, signal,
    wavfile, hp, _mel_basis};hp 是 hparams.HParams 实例(→ from … import hparams as hp);
    _mel_basis 初值 None;
  * save_wav 归一化基线实测带 0.01 下限(|max|=0.002 → 3276/6553;0.001 → 327);
  * _normalize/_denormalize 的四开关组合 × 越界输入网格实测:
    clip 分支 = np.clip(expr, ±max_abs_value)/(0…max_abs_value),
    非 clip 分支 = 裸表达式;min_level_db=-80/max_abs_value=2 复测锁定
    _denormalize 的 clip 边界为 [hp.min_level_db, 0];
  * save_wavenet_wav 在本环境恒 AttributeError("No librosa attribute output",
    librosa 0.11 移除 librosa.output;该文案由 librosa 自身 __getattr__ 产生,
    纯 Python 候选同样逐字一致);
  * _stft(use_lws=True) 走 `_lws_processor(hp)`(实测 TypeError,本构建的既有缺陷);
  * num_frames/pad_lr 的整数与浮点、负长度、2-D 输入实测(见报告 §3)。

覆盖单元:19 个 block(每函数 B0);无分支边、无状态机。
"""
import librosa  # [pyd pymod_exec:PyImport librosa]
import librosa.filters  # [pyd pymod_exec:PyImport librosa.filters]
import numpy as np  # [pyd pymod_exec:import numpy as np]
from scipy import signal  # [pyd pymod_exec:from scipy import signal]
from scipy.io import wavfile  # [pyd pymod_exec:from scipy.io import wavfile]

# [pyd pymod_exec:STR("modules.w2l.hparams") → STR("hparams") → 绑定名 hp]
from modules.w2l.hparams import hparams as hp


def load_wav(path, sr):  # [pyd FUN_180001000 包装 + FUN_180001200 体]
    # [pyd FUN_180001200:STR("librosa")→core→load→STR("sr");E1 返回纯 ndarray]
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr):  # [pyd FUN_1800014f0 包装 + FUN_1800017a0 体]
    # [pyd FUN_1800017a0 py=12] np.max(np.abs(wav)) 先求、再就地 *=;
    # E1:save_wav([0.001,0.002]) → [3276,6553] ⇒ 基线 = 32767/max(0.01, |max|)
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # [pyd FUN_1800017a0 py=14:STR("wavfile")→STR("write"),STR("astype"),STR("int16")]
    wavfile.write(path, sr, wav.astype(np.int16))


def save_wavenet_wav(wav, path, sr):  # [pyd FUN_180002270 包装 + FUN_180002520 体]
    # [pyd FUN_180002520:STR("librosa")→STR("output")→STR("write_wav")→STR("sr")]
    # 本环境 librosa 0.11 无 output 属性 → AttributeError(实测文案见模块 docstring)
    librosa.output.write_wav(path, wav, sr=sr)


def preemphasis(wav, k, preemphasize=True):  # [pyd FUN_180002800 + FUN_180002a90]
    if preemphasize:
        # [pyd FUN_180002a90:STR("signal")→STR("lfilter");E1 [1,2,3],k=.97→[1,1.03,1.06]]
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):  # [pyd FUN_180002e70 + FUN_180003100]
    if inv_preemphasize:
        # [pyd FUN_180003100:STR("signal")→STR("lfilter")]
        return signal.lfilter([1], [1, -k], wav)
    return wav


def get_hop_size():  # [pyd FUN_1800034e0:hop_size/frame_shift_ms/sample_rate 槽位]
    hop_size = hp.hop_size
    if hop_size is None:
        assert hp.frame_shift_ms is not None
        # E1:frame_shift_ms=10 → 160(实测)
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    return hop_size


def linearspectrogram(wav):  # [pyd FUN_180003be0:wav,D,S 局部名]
    # [pyd FUN_180003be0:STR("_stft")→STR("preemphasis")→hp.preemphasis/preemphasize]
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    # [同函数:STR("_amp_to_db")→STR("np")→STR("abs")→hp.ref_level_db]
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db
    # [同函数:hp.signal_normalization → STR("_normalize");E1 关归一化返回裸 S(float64)]
    if hp.signal_normalization:
        return _normalize(S)
    return S


def melspectrogram(wav):  # [pyd FUN_180004bd0:wav,D,S 局部名]
    # [pyd FUN_180004bd0:STR("_stft")→STR("preemphasis")→hp.preemphasis/preemphasize]
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    # [同函数:STR("_amp_to_db")→STR("_linear_to_mel")→STR("np")→STR("abs")→hp.ref_level_db]
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db
    if hp.signal_normalization:
        return _normalize(S)
    return S


def _lws_processor():  # [pyd FUN_180005df0:局部名 lws;hp.n_fft/get_hop_size/win_size]
    # [pyd FUN_180005df0:STR("lws")→STR("lws")→hp.n_fft→get_hop_size→
    #  STR("fftsize")→hp.win_size→STR("mode")→STR("speech")]
    import lws
    return lws.lws(hp.n_fft, get_hop_size(), fftsize=hp.win_size, mode="speech")


def _stft(y):  # [pyd FUN_180006470:局部名 y]
    if hp.use_lws:
        # [pyd FUN_180006470:STR("_lws_processor")→hp→STR("stft")→STR("T")]
        # 注意:本构建把 hp 当位置实参传给零参 `_lws_processor`(E1 实测 TypeError),
        # 属既有缺陷,按原样复刻,不得"顺手修好"。
        return _lws_processor(hp).stft(y).T
    else:
        # [pyd FUN_180006470:STR("librosa")→STR("stft")→STR("y")→hp.n_fft→
        #  get_hop_size→STR("hop_length")→hp.win_size→STR("win_length")]
        return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=get_hop_size(),
                            win_length=hp.win_size)


def num_frames(length, fsize, fshift):  # [pyd FUN_180007090:length,fsize,fshift,pad,M]
    """Compute number of time frames of spectrogram
    """
    # [pyd FUN_180007090 py 槽位 length/fsize/fshift;E1 0→3, 10.5→7.0 走 `//` 与 `%`]
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):  # [pyd FUN_180007710 包装 + FUN_1800079c0 体]
    """Compute left and right padding
    """
    # [pyd FUN_1800079c0:STR("num_frames");E1 len() 语义(list/2-D 均按 len(x))]
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


def librosa_pad_lr(x, fsize, fshift):  # [pyd FUN_1800080b0 包装 + FUN_180008350 体]
    # [pyd FUN_180008350:STR("shape")×2;E1 (2,10)/fsize800/fshift200 → 198 用 fshift]
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]


# Conversions
# [pyd pymod_exec:STR("_mel_basis") 出现在 librosa_pad_lr 与 _linear_to_mel 之间]
_mel_basis = None


def _linear_to_mel(spectogram):  # [pyd FUN_1800085b0:参数名 spectogram(原文拼写)]
    global _mel_basis
    # [pyd FUN_1800085b0:STR("_build_mel_basis")→STR("_mel_basis")→
    #  STR("np")→STR("dot")→STR("_mel_basis")]
    _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis():  # [pyd FUN_180008c60:无局部名]
    # [pyd FUN_180008c60:E1 fmax=9000 → AssertionError ⇒ assert 在 mel() 之前]
    assert hp.fmax <= hp.sample_rate // 2
    # [同函数:STR("librosa")→STR("filters")→STR("mel")→hp.sample_rate→STR("sr")→
    #  hp.n_fft→STR("n_fft")→hp.num_mels→STR("n_mels")→hp.fmin→STR("fmin")→
    #  hp.fmax→STR("fmax")(全部关键字实参)]
    return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,
                               fmin=hp.fmin, fmax=hp.fmax)


def _amp_to_db(x):  # [pyd FUN_180009a80:局部名 x,min_level]
    # [pyd FUN_180009a80:STR("np")→STR("exp")→hp.min_level_db→STR("np")→STR("log")→
    #  STR("np")→STR("log10")→STR("np")→STR("maximum")]
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):  # [pyd FUN_18000a580:STR("np")→STR("power")]
    return np.power(10.0, (x) * 0.05)


def _normalize(S):  # [pyd FUN_18000a850:py=110/111/112/115/117/118/119/121]
    if hp.allow_clipping_in_normalization:  # [py=110]
        if hp.symmetric_mels:  # [py=111]
            # [py=112;E1 clip 边界 = ±max_abs_value 实测]
            return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                           -hp.max_abs_value, hp.max_abs_value)
        else:
            # [py=115;E1 clip 边界 = 0…max_abs_value 实测]
            return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)

    # [py=117;E1 触发条件实测:S=1.0 或 S.min()-min_level_db<0 → AssertionError]
    assert S.max() <= 0 and S.min() - hp.min_level_db >= 0
    if hp.symmetric_mels:  # [py=118]
        return (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value  # [py=119]
    else:
        return hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))  # [py=121]


def _denormalize(D):  # [pyd FUN_18000d250:py=124/125/126/130/132/133/135]
    if hp.allow_clipping_in_normalization:  # [py=124]
        if hp.symmetric_mels:  # [py=125]
            # [py=126;E1 min_level_db=-80/max_abs_value=2 复测:边界=[min_level_db,0]]
            return np.clip(((D + hp.max_abs_value) * (-hp.min_level_db) / (2 * hp.max_abs_value)) + hp.min_level_db,
                           hp.min_level_db, 0)
        else:
            # [py=130;同上边界实测]
            return np.clip(((D * (-hp.min_level_db) / hp.max_abs_value) + hp.min_level_db), hp.min_level_db, 0)

    if hp.symmetric_mels:  # [py=132]
        return (((D + hp.max_abs_value) * (-hp.min_level_db) / (2 * hp.max_abs_value)) + hp.min_level_db)  # [py=133]
    else:
        return ((D * (-hp.min_level_db) / hp.max_abs_value) + hp.min_level_db)  # [py=135]
