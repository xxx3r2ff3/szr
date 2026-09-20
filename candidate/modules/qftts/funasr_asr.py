# -*- coding:utf-8 -*-

import argparse
import os
import traceback

# from funasr.utils import version_checker
# version_checker.check_for_update = lambda: None
from funasr import AutoModel
from tqdm import tqdm

funasr_models = {}  # 存储模型避免重复加载


def only_asr(input_file, language):
    try:
        model = create_model(language)
        text = model.generate(input=input_file)[0]["text"]
    except:
        text = ""
        print(traceback.format_exc())
    return text


def create_model(language="zh"):
    path_vad = "tools/asr/models/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    path_punc = "tools/asr/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    path_vad = (
        path_vad
        if os.path.exists(path_vad)
        else "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    )
    path_punc = (
        path_punc
        if os.path.exists(path_punc)
        else "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    )
    vad_model_revision = punc_model_revision = "v2.0.4"

    if language == "zh":
        path_asr = "tools/asr/models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        path_asr = (
            path_asr
            if os.path.exists(path_asr)
            else "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        )
        model_revision = "v2.0.4"
    elif language == "yue":
        path_asr = "tools/asr/models/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online"
        path_asr = (
            path_asr
            if os.path.exists(path_asr)
            else "iic/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online"
        )
        model_revision = "master"
        path_vad = path_punc = None
        vad_model_revision = punc_model_revision = None
        ###友情提示：粤语带VAD识别可能会有少量shape不对报错的，但是不带VAD可以.不带vad只能分阶段单独加标点。不过标点模型对粤语效果真的不行…
    else:
        raise ValueError("FunASR 不支持该语言" + ": " + language)

    if language in funasr_models:
        return funasr_models[language]
    else:
        model = AutoModel(
            model=path_asr,
            model_revision=model_revision,
            vad_model=path_vad,
            vad_model_revision=vad_model_revision,
            punc_model=path_punc,
            punc_model_revision=punc_model_revision,
        )
        print(f"FunASR 模型加载完成: {language.upper()}")

        funasr_models[language] = model
        return model


def execute_asr(input_folder, model_size, language):
    input_file_names = [
        f for f in os.listdir(input_folder) if f.lower().endswith(".wav")
    ]
    input_file_names.sort()

    model = create_model(language)
    for file_name in tqdm(input_file_names):
        try:
            print(file_name)
            file_path = os.path.join(input_folder, file_name)
            text = model.generate(input=file_path)[0]["text"]

            refer_path = file_path.replace(".wav", ".txt")
            with open(refer_path, "w", encoding="utf-8") as f:
                f.write(text)
        except:
            print(traceback.format_exc())
    print("ASR 任务完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        required=True,
        help="Path to the folder containing WAV files.",
    )
    parser.add_argument(
        "-s",
        "--model_size",
        type=str,
        default="large",
        help="Model Size of FunASR is Large",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default="zh",
        choices=["zh", "yue", "auto"],
        help="Language of the audio files.",
    )
    cmd = parser.parse_args()
    execute_asr(
        input_folder=cmd.input_folder,
        model_size=cmd.model_size,
        language=cmd.language,
    )
