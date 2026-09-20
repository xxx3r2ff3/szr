# -*- coding: utf-8 -*-
"""api_v2 —— R038(T4R:qftts TTS HTTP 服务模块,GPT-SoVITS api_v2 形态 + ZipVoice ONNX 推理)。

源件:modules/qftts/api_v2.cp310-win_amd64.pyd
      sha256 4051a1489e0ff9a983b0b90f29f98cab73372c901537283b496b60e7ffd9006e
      (contracts/modules/qftts__api_v2.json)

对账材料与出处标注体系
----------------------
* [上游] k2-fsa/ZipVoice(git 2f7326fbfe999a3ad179e3f1af82a424d4a62819)
  ``zipvoice/bin/infer_zipvoice_onnx.py``(OnnxModel/sample/get_vocoder/get_parser/initialize 语义)
  与 ``zipvoice/bin/infer_zipvoice.py``(generate_sentence 语义);
  快照:szr2026_out/bindiff/zipvoice_upstream。
* [E1-surface] probe_api2_out.json:app_sys_path + cwd 夹具(polyphonic.rep、
  polyphonic-fix.rep)下 oracle 导入成功;模块公开名 60 个逐名列出(本文件
  模块级名字集合与之逐一对应,导入面二者一致)。
* [E1-sig]   probe_api3_out.json "signatures":generate_sentence /
  get_prompt_features / sample / OnnxModel.__init__ / run_text_encoder /
  run_fm_decoder / get_vocoder 的 inspect.signature 实测;tts_endpoint 非
  coroutine 函数形态(Cython async)、qualname=="tts_endpoint"。
* [E1-dict]  probe_api3/4/5_out.json:read_dict() 零参、按行
  ``word, pinyin_str = line.strip().split(":")`` 严格二元解包
  (probe5 无冒号行 → ``ValueError: need more than 1 value to unpack``,
  probe4 同);cache_dict(dict, path) 落 pickle(probe3:1322632 字节,与
  树上 polyphonic.pickle 逐字节同大小);build_pinyin_dict 返回
  (dict[word]="<p1><p2>", set(words))(probe3 shapes);stale-pickle 与
  合并/重复键语义见 probe5;replace_words_with_jieba 逐词替换实测:
  "中标的湖泊"→"中<biao1><di4><hu2><po1>"、"差不多吧"→"<cha4><bu5><duo1>吧"。
* [E1-args]  probe_api3_out.json "args_defaults":模块级 parser 解析出
  host='0.0.0.0'/port=9885(与明文 start_api.py 的 -a/-p 定义同形);
  zip_parser_defaults:get_zip_parser() 的 21 个参数默认值逐项实测。
* [E1-fail]  contracts evidence ev-qftts__api_v2-*:cwd 缺 polyphonic.rep 时
  导入期 FileNotFoundError(probe_api1 phaseA 复现同文案)。
* [str]      二进制串表(evidence/modules/qftts__api_v2/static/):0.0.0.0、
  HUGGINGFACE_REPO、MODEL_DIR、CACHE_PATH、PP_DICT_PATH、PP_FIX_DICT_PATH、
  DIY_PP_DICT_PATH、json_post_raw、tts_endpoint、result.wav、temp 等字符串槽位。

差分口径(见 reports/modules/qftts__api_v2-impl.md §7):
 词典子系统/解析器/导入门控为双侧差分已定谳单元;模型装载链
 (get_sentence_params/get_prompt_features/generate_sentence/tts_endpoint)
 为 [E1-sig]+[上游] 语义重建,数值路径未差分(需真模型,登记未定谳)。
"""
import argparse
import datetime as dt
import json
import logging
import os
import pickle
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

if sys.platform == "win32":
    for _p in [
        os.path.join(sys.prefix, "cuda", "v11.8", "bin"),
        os.path.join(sys.prefix, "bin"),
        os.path.join(sys.prefix, "lib", "site-packages", "onnxruntime", "capi"),
    ]:
        if os.path.isdir(_p):
            try:
                os.add_dll_directory(_p)
            except Exception:
                pass
            os.environ["PATH"] = _p + ";" + os.environ.get("PATH", "")

import jieba
import onnxruntime
import torch
import torch.nn as nn
import torchaudio
from huggingface_hub import hf_hub_download
from lhotse.utils import fix_random_seed
from torch import Tensor

# [E1-surface] 兄弟包绝对导入(oracle 导入期新增 sys.modules:infer/utils/
# feature/tokenizer/solver/scaling/zipformer),六个 infer 名与目录面逐一对齐。
from infer import (
    add_punctuation,
    chunk_tokens_punctuation,
    cross_fade_concat,
    load_prompt_wav,
    remove_silence,
    rms_norm,
)
from feature import VocosFbank
from solver import get_time_steps
from tokenizer import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from utils import AttributeDict, str2bool
from vocos import Vocos

# ---------------------------------------------------------------------------
# 模块级常量([E1-surface] dir() 实测值;[str] 串表槽位)
# ---------------------------------------------------------------------------
PP_DICT_PATH = "polyphonic.rep"        # [E1-surface] 常量值实测
PP_FIX_DICT_PATH = "polyphonic-fix.rep"
CACHE_PATH = "polyphonic.pickle"
HUGGINGFACE_REPO = "k2-fsa/ZipVoice"
MODEL_DIR = {
    "zipvoice": "zipvoice",
    "zipvoice_distill": "zipvoice_distill",
}

# [E1-dict] probe_api3 "paths":now_dir=__file__ 所在目录、
# root_dir=now_dir 上两级、DIY_PP_DICT_PATH=root_dir/polyphonic.rep。
now_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(now_dir))
DIY_PP_DICT_PATH = os.path.join(root_dir, "polyphonic.rep")


def read_dict():
    # [E1-dict] 零参(inspect.signature "()"、call0 "takes no arguments");
    # 逐行严格二元解包(probe5/probe8:无冒号行、空行 → ValueError
    # "need more than 1 value to unpack");读 PP_DICT_PATH 后再读
    # PP_FIX_DICT_PATH(probe_api1:按缺失顺序依次报错),后文件覆盖先文件
    # (probe_api8 tiny_rep_tiny_fix_dups:fix 的 湖泊=['yy2'] 覆盖 rep)。
    # [E1-dict] probe_api10:oracle 对**音节串做对象共享**(解析期 intern 表;
    # Mac 端逐字节复现:仅"音节 setdefault 共享"变体 dump(protocol=4) 与树上
    # polyphonic.pickle sha 一致 c49a8a3b…,plain/整值缓存变体均不等),
    # 因此本实现在解析期维护音节共享表 —— cache_dict 落盘字节才能与
    # oracle 逐字节一致(1322632)。
    polyphonic_dict = {}
    seen = {}
    for rep_path in (PP_DICT_PATH, PP_FIX_DICT_PATH):
        with open(rep_path, "r", encoding="utf-8") as f:
            for line in f:
                word, pinyin_str = line.strip().split(":")
                word = word.strip()
                inner = pinyin_str.strip()
                if inner.startswith("[") and inner.endswith("]"):
                    inner = inner[1:-1]
                pinyins = []
                for p in inner.split(","):
                    p = p.strip().strip("'\"")
                    if p:
                        p = seen.setdefault(p, p)
                        pinyins.append(p)
                polyphonic_dict[word] = pinyins
    return polyphonic_dict


def cache_dict(polyphonic_dict, file_path):
    # [E1-dict] probe_api3 "cache_dict":返回 None,落盘 pickle 与树上
    # polyphonic.pickle 逐字节同大小(1322632);树上件头字节 80 04 95 =
    # **protocol 4**(3.10 默认为 5,故为显式协议;首轮捕获 sha 差异根因)。
    with open(file_path, "wb") as f:
        pickle.dump(polyphonic_dict, f, protocol=4)


def get_dict():
    # [E1-dict] 零参;probe_api9 "oracle_stale_pickle":cwd 预置过期
    # polyphonic.pickle 时 word_data 仍为 rep 全量重读(get_eq_stale=false、
    # get_eq_read=true)→ get_dict 不消费 pickle 内容,只在其缺失时落盘缓存
    # (probe_api5/6 "import_only_fullpath":纯 import 即写 polyphonic.pickle)。
    polyphonic_dict = read_dict()
    if not os.path.exists(CACHE_PATH):
        cache_dict(polyphonic_dict, CACHE_PATH)
    return polyphonic_dict


def build_pinyin_dict(word_data):
    # [E1-dict] probe_api3 "build_pinyin_dict_shapes":
    #   {}                  -> ({}, set())
    #   {"湖泊": [...]}     -> ({"湖泊": "<hu2><po1>"}, {"湖泊"})
    #   {"差不多": [...]}   -> ({"差不多": "<cha4><bu5><duo1>"}, {...})
    # 返回 (词→<音1><音2> 拼接串, 词集合) 二元组;模块级
    # pinyin_dict/custom_dict 即其两个分量(probe3 长度均为 45047)。
    pinyin_dict = {}
    for word, pinyins in word_data.items():
        pinyin_dict[word] = "".join("<%s>" % p for p in pinyins)
    return pinyin_dict, set(word_data.keys())


def replace_words_with_jieba(sentence, pinyin_dict, custom_dict):
    # [E1-dict] probe_api3 "rwwj":jieba 分词后逐词查词典,命中替换为
    # <拼>串、未命中保留原词,拼接返回;空串 → ""(实测 "''")。
    words = jieba.lcut(sentence)
    result = ""
    for word in words:
        if word in custom_dict:
            result += pinyin_dict[word]
        else:
            result += word
    return result


def get_zip_parser():
    # [E1-args] probe_api2 "zip_parser_defaults":21 个参数与默认值逐项实测
    # (与上游 infer_zipvoice_onnx.get_parser 相比:去掉 --raw-evaluation;
    # model-dir 默认 "pretrained_models"、vocoder-path 默认
    # "pengzhendong/vocos-mel-24khz"、remove-long-sil 默认 True —— 三处
    # 本地化改动均以实测为准)。formatter_class 实测
    # ArgumentDefaultsHelpFormatter。
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--onnx-int8",
        type=str2bool,
        default=False,
        help="Whether to use the int8 model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="zipvoice",
        choices=["zipvoice", "zipvoice_distill"],
        help="The model used for inference",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="pretrained_models",
        help="The path to the local onnx model.",
    )
    parser.add_argument(
        "--vocoder-path",
        type=str,
        default="pengzhendong/vocos-mel-24khz",
        help="The vocoder checkpoint.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="emilia",
        choices=["emilia", "libritts", "espeak", "simple"],
        help="Tokenizer type.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en-us",
        help="Language identifier, used when tokenizer type is espeak. see"
        "https://github.com/rhasspy/espeak-ng/blob/master/docs/languages.md",
    )
    parser.add_argument(
        "--test-list",
        type=str,
        default=None,
        help="The list of prompt speech, prompt_transcription, "
        "and text to synthesize in the format of "
        "'{wav_name}\\t{prompt_transcription}\\t{prompt_wav}\\t{text}'.",
    )
    parser.add_argument(
        "--prompt-wav",
        type=str,
        default=None,
        help="The prompt wav to mimic",
    )
    parser.add_argument(
        "--prompt-text",
        type=str,
        default=None,
        help="The transcription of the prompt wav",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="The text to synthesize",
    )
    parser.add_argument(
        "--res-dir",
        type=str,
        default="results",
        help="Path name of the generated wavs dir, used when test-list is not None",
    )
    parser.add_argument(
        "--res-wav-path",
        type=str,
        default="result.wav",
        help="Path name of the generated wav path, used when test-list is None",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="The scale of classifier-free guidance during inference.",
    )
    parser.add_argument(
        "--num-step",
        type=int,
        default=None,
        help="The number of sampling steps.",
    )
    parser.add_argument(
        "--feat-scale",
        type=float,
        default=0.1,
        help="The scale factor of fbank feature",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Control speech speed, 1.0 means normal, >1.0 means speed up",
    )
    parser.add_argument(
        "--t-shift",
        type=float,
        default=0.5,
        help="Shift t to smaller ones if t_shift < 1.0",
    )
    parser.add_argument(
        "--target-rms",
        type=float,
        default=0.1,
        help="Target speech normalization rms value, set to 0 to disable normalization",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=666,
        help="Random seed",
    )
    parser.add_argument(
        "--num-thread",
        type=int,
        default=1,
        help="Number of threads to use for ONNX Runtime and PyTorch.",
    )
    parser.add_argument(
        "--remove-long-sil",
        type=str2bool,
        default=True,
        help="Whether to remove long silences in the middle of the generated "
        "speech (edge silences will be removed by default).",
    )
    return parser


class OnnxModel:
    """ONNX 推理壳([上游] infer_zipvoice_onnx.py OnnxModel 逐行还原;
    [E1-sig] __init__/run_text_encoder/run_fm_decoder 签名实测;
    onnxruntime 以模块名直引(dir() 名为 onnxruntime,非 ort)。
    [E1-dict] probe_api3 "onnxmodel_failpath":不存在路径构造 →
    onnxruntime NoSuchFile("Load model from no1.onnx failed")。"""

    def __init__(
        self,
        text_encoder_path: str,
        fm_decoder_path: str,
        num_thread: int = 1,
    ):
        session_opts = onnxruntime.SessionOptions()
        session_opts.inter_op_num_threads = num_thread
        session_opts.intra_op_num_threads = num_thread

        self.session_opts = session_opts

        self.init_text_encoder(text_encoder_path)
        self.init_fm_decoder(fm_decoder_path)

    def init_text_encoder(self, model_path: str):
        self.text_encoder = onnxruntime.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            # [修复 2026-09-20 短视频接入] 原写死 CPUExecutionProvider(RTF≈10,
            # 实时渲染链吃不消);本机 onnxruntime-gpu 具备 Tensorrt/CUDA EP,
            # 改为 CUDA 优先、CPU 兜底(EP 初始化失败时 ORT 自动回退次序)。
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def init_fm_decoder(self, model_path: str):
        self.fm_decoder = onnxruntime.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        meta = self.fm_decoder.get_modelmeta().custom_metadata_map
        self.feat_dim = int(meta["feat_dim"])

    def run_text_encoder(
        self,
        tokens: Tensor,
        prompt_tokens: Tensor,
        prompt_features_len: Tensor,
        speed: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        out = self.text_encoder.run(
            [
                self.text_encoder.get_outputs()[0].name,
            ],
            {
                self.text_encoder.get_inputs()[0].name: tokens.numpy(),
                self.text_encoder.get_inputs()[1].name: prompt_tokens.numpy(),
                self.text_encoder.get_inputs()[2].name: prompt_features_len.numpy(),
                self.text_encoder.get_inputs()[3].name: speed.numpy(),
            },
        )
        return torch.from_numpy(out[0])

    def run_fm_decoder(
        self,
        t: Tensor,
        x: Tensor,
        text_condition: Tensor,
        speech_condition: Tensor,
        guidance_scale: Tensor,
    ) -> Tensor:
        out = self.fm_decoder.run(
            [
                self.fm_decoder.get_outputs()[0].name,
            ],
            {
                self.fm_decoder.get_inputs()[0].name: t.numpy(),
                self.fm_decoder.get_inputs()[1].name: x.numpy(),
                self.fm_decoder.get_inputs()[2].name: text_condition.numpy(),
                self.fm_decoder.get_inputs()[3].name: speech_condition.numpy(),
                self.fm_decoder.get_inputs()[4].name: guidance_scale.numpy(),
            },
        )
        return torch.from_numpy(out[0])


def sample(
    model: "OnnxModel",
    tokens: List[List[int]],
    prompt_tokens: List[List[int]],
    prompt_features: Tensor,
    speed: float = 1.0,
    t_shift: float = 0.5,
    guidance_scale: float = 1.0,
    num_step: int = 16,
) -> Tensor:
    # [上游] infer_zipvoice_onnx.sample 逐行还原;[E1-sig] 签名实测;
    # [E1-dict] probe_api3 "sample_dummy":哑模型(num_frames=5、prompt 3 帧、
    # feat_dim=8、num_step=3、fix_random_seed(0))输出 shape (1,2,8)
    # = x[:, prompt_features_len:, :] 截断语义实测一致。
    assert len(tokens) == len(prompt_tokens) == 1
    tokens = torch.tensor(tokens, dtype=torch.int64)
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.int64)
    prompt_features_len = torch.tensor(prompt_features.size(1), dtype=torch.int64)
    speed = torch.tensor(speed, dtype=torch.float32)

    text_condition = model.run_text_encoder(
        tokens, prompt_tokens, prompt_features_len, speed
    )

    batch_size, num_frames, _ = text_condition.shape
    assert batch_size == 1
    feat_dim = model.feat_dim

    timesteps = get_time_steps(
        t_start=0.0,
        t_end=1.0,
        num_step=num_step,
        t_shift=t_shift,
    )
    x = torch.randn(batch_size, num_frames, feat_dim)
    speech_condition = torch.nn.functional.pad(
        prompt_features, (0, 0, 0, num_frames - prompt_features.shape[1])
    )
    guidance_scale = torch.tensor(guidance_scale, dtype=torch.float32)

    for step in range(num_step):
        v = model.run_fm_decoder(
            t=timesteps[step],
            x=x,
            text_condition=text_condition,
            speech_condition=speech_condition,
            guidance_scale=guidance_scale,
        )
        x = x + v * (timesteps[step + 1] - timesteps[step])

    x = x[:, prompt_features_len.item():, :]
    return x


@lru_cache(maxsize=None)
def get_vocoder(vocos_local_path: Optional[str] = None):
    # [上游] infer_zipvoice.get_vocoder 逐行还原;[E1-sig] 签名实测;
    # [E1-surface] get_vocoder 为 _lru_cache_wrapper(上游未缓存,此处
    # lru_cache 包装属实测口径)。Vocos 来自已安装的 vocos 包。
    if vocos_local_path:
        vocoder = Vocos.from_hparams(f"{vocos_local_path}/config.yaml")
        state_dict = torch.load(
            f"{vocos_local_path}/pytorch_model.bin",
            weights_only=True,
            map_location="cpu",
        )
        vocoder.load_state_dict(state_dict)
    else:
        vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    return vocoder


@lru_cache(maxsize=None)
def get_prompt_features(
    tokenizer,
    feature_extractor,
    target_rms,
    feat_scale,
    sampling_rate,
    prompt_wav,
    prompt_text,
    speed,
):
    # [E1-sig] 签名 8 参实测(_lru_cache_wrapper);函数体按上游
    # generate_sentence 的 prompt 预处理段语义重建(load_prompt_wav →
    # rms_norm → fbank 提取 → 尺度缩放),返回 (features, rms, duration,
    # prompt_tokens_str) —— 返回结构为串表名(prompt_features/prompt_rms/
    # prompt_duration/prompt_tokens_str)推定,未做数值差分(见报告 §7)。
    prompt_wav = load_prompt_wav(prompt_wav, sampling_rate=sampling_rate)
    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)
    prompt_duration = prompt_wav.shape[-1] / sampling_rate
    prompt_features = feature_extractor.extract(
        prompt_wav, sampling_rate=sampling_rate
    )
    prompt_features = prompt_features.unsqueeze(0) * feat_scale
    prompt_tokens_str = tokenizer.texts_to_tokens([prompt_text])[0]
    return prompt_features, prompt_rms, prompt_duration, prompt_tokens_str


@lru_cache(maxsize=None)
def get_sentence_params():
    # [E1-dict] 零参(_lru_cache_wrapper);懒加载单例:cwd 缺
    # pretrained_models 目录时 FileNotFoundError("pretrained_models does
    # not exist")(probe_api3 "get_sentence_params" 首步实测),缺
    # pretrained_models\text_encoder.onnx 时同型文案(probe_api3 steps[0])
    # —— 检查顺序按上游 initialize 本地目录分支(text_encoder/fm_decoder/
    # model.json/tokens.txt)。函数体为 [上游]+[E1-args] 重建,未数值差分。
    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"{model_dir} does not exist")
    text_encoder_name = "text_encoder_int8.onnx" if args.onnx_int8 else "text_encoder.onnx"
    fm_decoder_name = "fm_decoder_int8.onnx" if args.onnx_int8 else "fm_decoder.onnx"
    for filename in (text_encoder_name, fm_decoder_name, "model.json", "tokens.txt"):
        path = os.path.join(model_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} does not exist")
    token_file = os.path.join(model_dir, "tokens.txt")
    if args.tokenizer == "emilia":
        tokenizer = EmiliaTokenizer(token_file=token_file)
    elif args.tokenizer == "libritts":
        tokenizer = LibriTTSTokenizer(token_file=token_file)
    elif args.tokenizer == "espeak":
        tokenizer = EspeakTokenizer(token_file=token_file, lang=args.lang)
    else:
        assert args.tokenizer == "simple"
        tokenizer = SimpleTokenizer(token_file=token_file)
    with open(os.path.join(model_dir, "model.json"), "r", encoding="utf-8") as f:
        model_config = json.load(f)
    model = OnnxModel(
        os.path.join(model_dir, text_encoder_name),
        os.path.join(model_dir, fm_decoder_name),
        num_thread=args.num_thread,
    )
    vocoder = get_vocoder(args.vocoder_path)
    vocoder.eval()
    if model_config["feature"]["type"] == "vocos":
        feature_extractor = VocosFbank()
    else:
        raise NotImplementedError(
            f"Unsupported feature type: {model_config['feature']['type']}"
        )
    params = AttributeDict()
    params.update(vars(args))
    model_defaults = {
        "zipvoice": {
            "num_step": 16,
            "guidance_scale": 1.0,
        },
        "zipvoice_distill": {
            "num_step": 8,
            "guidance_scale": 3.0,
        },
    }
    for param, value in model_defaults.get(params.model_name, {}).items():
        if getattr(params, param) is None:
            setattr(params, param, value)
            logging.info(f"Setting {param} to default value: {value}")
    params.sampling_rate = model_config["feature"]["sampling_rate"]
    return model, vocoder, tokenizer, feature_extractor, params


@torch.inference_mode()
def generate_sentence(
    save_path: str,
    prompt_text: str,
    prompt_wav: str,
    text: str,
    model,
    vocoder,
    tokenizer,
    feature_extractor,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    remove_long_sil: bool = False,
):
    # [E1-sig] 签名 16 参逐字实测(与上游 generate_sentence 相比无 device 形参;
    # 尾部 remove_long_sil: bool = False)。函数体按上游 onnx 变体语义重建:
    # get_prompt_features 预处理 → add_punctuation → texts_to_tokens →
    # token_duration/chunk_tokens_punctuation 切块 → 逐块 sample → vocoder →
    # cross_fade_concat → remove_silence → metrics(rtf 三项)→ torchaudio.save。
    # 数值路径未差分(需真模型,报告 §7 登记)。
    (
        prompt_features,
        prompt_rms,
        prompt_duration,
        prompt_tokens_str,
    ) = get_prompt_features(
        tokenizer, feature_extractor, target_rms, feat_scale, sampling_rate,
        prompt_wav, prompt_text, speed,
    )

    if prompt_duration > 20:
        logging.warning(
            f"Given prompt wav is too long ({prompt_duration}s). "
            f"Please provide a shorter one (1-3 seconds is recommended)."
        )
    elif prompt_duration > 10:
        logging.warning(
            f"Given prompt wav is long ({prompt_duration}s). "
            f"It will lead to slower inference speed and possibly worse speech quality."
        )

    text = add_punctuation(text)
    prompt_text = add_punctuation(prompt_text)

    tokens_str = tokenizer.texts_to_tokens([text])[0]

    token_duration = prompt_duration / (len(prompt_tokens_str) * speed)
    max_tokens = int((25 - prompt_duration) / token_duration)
    chunked_tokens_str = chunk_tokens_punctuation(tokens_str, max_tokens=max_tokens)
    chunked_tokens = tokenizer.tokens_to_token_ids(chunked_tokens_str)
    prompt_tokens = tokenizer.tokens_to_token_ids([prompt_tokens_str])

    chunked_features = []
    start_t = dt.datetime.now()
    for tokens in chunked_tokens:
        pred_features = sample(
            model=model,
            tokens=[tokens],
            prompt_tokens=prompt_tokens,
            prompt_features=prompt_features,
            speed=speed,
            t_shift=t_shift,
            guidance_scale=guidance_scale,
            num_step=num_step,
        )
        pred_features = pred_features.permute(0, 2, 1) / feat_scale
        chunked_features.append(pred_features)

    chunked_wavs = []
    start_vocoder_t = dt.datetime.now()
    for pred_features in chunked_features:
        wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)
        if prompt_rms < target_rms:
            wav = wav * prompt_rms / target_rms
        chunked_wavs.append(wav)

    t = (dt.datetime.now() - start_t).total_seconds()

    final_wav = cross_fade_concat(
        chunked_wavs, fade_duration=0.1, sample_rate=sampling_rate
    )
    final_wav = remove_silence(
        final_wav, sampling_rate, only_edge=(not remove_long_sil), trail_sil=0
    )

    t_no_vocoder = (start_vocoder_t - start_t).total_seconds()
    t_vocoder = (dt.datetime.now() - start_vocoder_t).total_seconds()
    wav_seconds = final_wav.shape[-1] / sampling_rate
    rtf = t / wav_seconds
    rtf_no_vocoder = t_no_vocoder / wav_seconds
    rtf_vocoder = t_vocoder / wav_seconds
    metrics = {
        "t": t,
        "t_no_vocoder": t_no_vocoder,
        "t_vocoder": t_vocoder,
        "wav_seconds": wav_seconds,
        "rtf": rtf,
        "rtf_no_vocoder": rtf_no_vocoder,
        "rtf_vocoder": rtf_vocoder,
    }

    torchaudio.save(save_path, final_wav.cpu(), sample_rate=sampling_rate)
    return metrics


async def tts_endpoint(request):
    # [str] 串表槽位 json_post_raw/prompt_text/prompt_wav/target_text/
    # speed/save_path/result.wav/temp;明文 start_api.py 以
    # ``metrics = await tts_endpoint(request)`` 消费并 print(metrics) →
    # 返回 metrics 字典。请求字段名与回退值为串表级推定,未差分(§7)。
    json_post_raw = await request.json()
    prompt_text = json_post_raw.get("prompt_text", "")
    prompt_wav = json_post_raw.get("prompt_wav", "")
    target_text = json_post_raw.get("target_text", "")
    speed = float(json_post_raw.get("speed", args.speed) or args.speed)
    guidance_scale = json_post_raw.get("guidance_scale") or args.guidance_scale
    num_step = json_post_raw.get("num_step") or args.num_step
    text = replace_words_with_jieba(target_text, pinyin_dict, custom_dict)
    # [修复 2026-09-20 短视频接入] 重建版把落盘路径写死为 now_dir/temp/
    # result.wav,忽略了请求的 output_path——而调用方 app_util.qfttsClone 的
    # 契约就是"服务写到请求给的 output_path"(E1 probe2 载荷含 output_path,
    # oracle 侧该文件真实生成后被 _finalize_16k 消费)。不修则 qfttsClone
    # 恒 FileNotFoundError。请求未给 output_path 时回退模块 temp(原行为)。
    save_path = json_post_raw.get("output_path") or os.path.join(
        now_dir, "temp", "result.wav")
    save_path = os.path.abspath(save_path)
    # [修复 2026-09-20 短视频接入] temp/ 目录缺失时 torchaudio.save 报
    # "Error opening ...result.wav: System error"(start_api 捕获后 200 null)。
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model, vocoder, tokenizer, feature_extractor, params = get_sentence_params()
    metrics = generate_sentence(
        save_path,
        prompt_text,
        prompt_wav,
        text,
        model,
        vocoder,
        tokenizer,
        feature_extractor,
        num_step=num_step,
        guidance_scale=guidance_scale,
        speed=speed,
        t_shift=args.t_shift,
        target_rms=args.target_rms,
        feat_scale=args.feat_scale,
        sampling_rate=params.sampling_rate,
        remove_long_sil=args.remove_long_sil,
    )
    return metrics


# [E1-args] 模块级 -a/--host、-p/--port 解析(probe_api3 "args_defaults":
# host='0.0.0.0'、port=9885;parser/args/unknow 均在 dir() 公开名中),
# 与明文 start_api.py 同形(description "TTS api")。
parser = argparse.ArgumentParser(description="TTS api")
parser.add_argument(
    "-a", "--host", type=str, default="0.0.0.0", help="default: 0.0.0.0"
)
parser.add_argument(
    "-p", "--port", type=int, default="9885", help="default: 9885"
)
args, unknow = parser.parse_known_args()

# [修复 2026-09-20 短视频接入] 模块级 args 原本只有 host/port,而 tts_endpoint
# 与模型加载链引用 args.speed/model_dir/tokenizer/guidance_scale 等——全量定义
# 在 get_zip_parser()(此前仅函数内可达)→ AttributeError → start_api 捕获后
# 回 200 null、不落文件的无声失败(旧服务进程是老代码才能活,重启即坏)。
# 修法:用全量 parser 对真实 argv 再 parse_known_args 一遍(--model-name 等
# 已知参数被吃下,--port 等未知参数忽略),把缺省属性并入模块 args;
# guidance_scale/num_step 全量默认为 None,显式传 None 会压过
# generate_sentence 的签名默认,钉成签名默认值(1.0/16,与旧服务工作路径一致)。
_zip_ns, _ = get_zip_parser().parse_known_args()
for _k in vars(_zip_ns):
    if not hasattr(args, _k):
        setattr(args, _k, getattr(_zip_ns, _k))
if getattr(args, "guidance_scale", None) is None:
    args.guidance_scale = 1.0
if getattr(args, "num_step", None) is None:
    args.num_step = 16

# [E1-dict] 模块级词典装配:word_data == read_dict()(probe_api3 实测);
# pinyin_dict/custom_dict = build_pinyin_dict(word_data) 两分量
# (长度均 45047)。缓存落盘口径以 probe_api5 "import_only_fullpath" 实测为准。
word_data = get_dict()
pinyin_dict, custom_dict = build_pinyin_dict(word_data)
