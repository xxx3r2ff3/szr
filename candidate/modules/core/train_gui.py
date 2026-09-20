# -*- coding: utf-8 -*-
"""train_gui —— core__train_gui 语义重建(R031 / T4R)。

源:modules/core/train_gui.cp310-win_amd64.pyd
    sha256 b834ca170db9c5fe0cb159cedf13385c7d54ac33d27e9f98c92597b56255fdab

二进制出处(Ghidra 反编译,evidence/modules/core__train_gui/static/pseudocode/):
  * FUN_1800012b0 @0x1800012b0 = PyInit exec 段(4216 行),按 @ERR(py=N) 行号切片:
      @2108 `import argparse`
      @2110 `import torch`
      @2121 `import webview`
      @2133 `import os`
      @2155 `from modules.core.config import debug, api_host`
      @2169/2173/2177 回溯注册 debug / api_host / debug(同名再导入)
      @2190 `from modules.core.train_util import set_train_window, TrainApi`
      @2215 `import argparse`  → parser = argparse.ArgumentParser(description="Train gui")
      @2231/~2248+ `--cli/--epochs/--file/--gpu_memory/--refer_text/--silence/
                    --train_back/--train_double/--train_gfp/--train_pose/
                    --train_shutup/--type/--wav_file` 等 add_argument 序列
      @2516..@3132 `create_window("一键训练", api_host + "/train?gpu_memory=", …)`
                    + TrainApi(js_api)+ `webview.start()`
  * 模块级串(串表):"Train gui" / "一键训练" / "/train?gpu_memory=" / "&layout=train"
    / "确定要退出么?" / "训练完成，模特资源包路径：" / "--cli" 等 13 个 CLI flag

环境行为(oracle 实测,evidence/modules/core__train_gui/runtime/probeE_train_gui_out.json):
  1. `modules.core.config` 导入成功并打印五行横幅
     (client version v6.18.0 / gpu trt False / local ip … / allow accounts all /
      allow host all);
  2. `modules.core.train_util` 导入期 `torch.load("checkpoints\\yolov8-face.pt")`
     抛 FileNotFoundError(当前 VM 无 GPU 且沙箱 cwd 无 checkpoints/)→
     train_gui 整个导入失败、模块命名空间为空(dir()=None);
  3. 无论是 cwd 有 config.ini / 无 config.ini / 带 --cli,失败点与消息都相同。

未定谳项(见 reports/modules/core__train_gui-impl.md §5):
  * GPU + checkpoints 就绪环境下的完整窗口/训练路径不可达(本 VM 无 NVIDIA 驱动,
    `torch.cuda.is_available()=False`),该路径留给 Win10/GPU 环境重开;
  * 契约 `core__train_gui::<module>::B1` 在**本次捕获环境里可达**(稳定复现),
    已按政策把 static_unreachable 降级为 dynamic_pass(条目保留)。
"""
import argparse   # 反编译 @py2108
import os         # @py2133
import torch      # @py2110
import webview    # @py2121

from modules.core.config import api_host, debug   # @py2155(import 期即打印横幅)

# 反编译 @py2190 的 `from modules.core.train_util import …` 在 oracle 侧会连带导入
# pydub(probeE stdout/stderr 均有其 RuntimeWarning)。bug 复刻:显式导入同版本 pydub,
# 使 stderr 与 stdout 与 oracle 逐字节一致(证据 probeE_train_gui_out.json)。
from pydub import AudioSegment  # noqa: E402,F401

# 反编译 @py2190 `from modules.core.train_util import set_train_window, TrainApi`。
# [I008 E2E 定谳 i008-fix-5] 候选 train_util 已重建(R011),恢复**真实导入**:
# 其模块体 torch.load(checkpoints/yolov8-face.pt) + YOLO(权重) 即 oracle 的
# 导入门控/加载语义(沙箱缺权重 ⇒ 同消息 FileNotFoundError;root 下权重在场,
# ultralytics 先于 torch.load 注册 safe globals,权重可正常加载——裸 torch.load
# 复刻版在 root 下会抛 pickle.UnpicklingError(weights_only),E2E round2
# stub_b1..b3 实测证伪,故废弃复刻)。
from modules.core.train_util import set_train_window, TrainApi  # noqa: E402,F401

# ---- 以下语句在 oracle 侧沙箱形态因上方导入门控抛错而不可达;按反编译骨架 +
# ---- I008 整机 E2E 定谳(root 形态双侧观测)保留并补齐非 GUI 可达分派链。
parser = argparse.ArgumentParser(description="Train gui")            # @py2215/2231
for _flag in ("--cli", "--train_back", "--train_double", "--train_gfp",   # @py2248+
              "--train_pose", "--train_shutup"):
    parser.add_argument(_flag, action="store_true")
parser.add_argument("--epochs", type=int, default=0)
parser.add_argument("--file", default="")
parser.add_argument("--wav_file", default="")
parser.add_argument("--refer_text", default="")
parser.add_argument("--silence", default="")
parser.add_argument("--gpu_memory", default="")
parser.add_argument("--type", default="")
args, unknown = parser.parse_known_args()

if args.cli:
    # [I008 E2E 定谳 i008-fix-5] --cli 分派不建窗、不触 CUDA(E2E stub_b2 oracle
    # stderr 为 FileNotFoundError 而非 NVIDIA 驱动错,app_lines=
    # ['开始训练 nope_i008.mp4', '路径不存在: modules/dh/000']),--type model →
    # train_model_v2(file)、--type voice → train_voice(wav_file)(单参即
    # TypeError 'takes at least 2 positional arguments (1 given)',stub_b3)。
    # 窗口事件循环在真 GUI 下永不返回,该分派链仅在 webview.start 返回后可达
    # (记录桩/E2E 口径);train_util 已在模块头真导入。
    from modules.core.train_util import train_model_v2, train_voice  # noqa: E402
    if args.type == "model":
        train_model_v2(args.file)
    elif args.type == "voice":
        train_voice(args.wav_file)
else:
    # [I008 E2E 定谳 i008-fix-5] 默认路径在建窗前先取 GPU 显存
    # (stub_b1_default oracle stderr = RuntimeError 'Found no NVIDIA driver …',
    # gui_calls 为空 ⇒ create_window 未达),@2516..@3132 的 create_window 串表
    # ('一键训练'/'/train?gpu_memory='/'&layout=train')其后再组。
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    window = webview.create_window(
        "一键训练",
        api_host + "/train?gpu_memory=" + str(gpu_memory) + "&layout=train",
        min_size=(0x35c, 0x438), resizable=False,
        js_api=TrainApi(),
    )
    webview.start(private_mode=False)
