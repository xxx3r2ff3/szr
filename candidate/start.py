# 检测并修复 onnxruntime CPU 版覆盖 GPU 版的问题
import subprocess, os, sys as _sys, time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    _result = subprocess.run(
        [_sys.executable, "-m", "pip", "show", "-f", "onnxruntime"],
        capture_output=True,
        text=True,
    )
    if _result.returncode == 0 and "onnxruntime-gpu" not in _result.stdout:
        print("检测到 onnxruntime CPU 版覆盖了 GPU 版，正在自动修复...")
        subprocess.run(
            [_sys.executable, "-m", "pip", "uninstall", "onnxruntime", "-y"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                _sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "onnxruntime-gpu==1.17.1",
                "numpy<2",
                "protobuf==3.20.3",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("修复完成，10秒后自动退出，请重新启动程序...")
        time.sleep(10)
        os._exit(0)
except Exception:
    pass
del subprocess, _sys

# 执行 ADS 清理
from modules.core.streams import perform_ads_cleanup

perform_ads_cleanup()

import multiprocessing
import os
import re
import sys
import threading


def has_chinese(path):
    pattern = re.compile(r"[\u4e00-\u9fff\uFF08-\uFF09]")  # 中文字符 + 中文括号
    return bool(pattern.search(path))


current_path = os.getcwd()
if has_chinese(current_path):
    print("当前路径不支持中文：", current_path)
    time.sleep(10)
    sys.exit()

from modules.core.app_gui import start_gui as startGui
from modules.core.app_video import start_video as startVideo

if __name__ == "__main__":
    multiprocessing.freeze_support()

    pid = os.getpid()
    p = threading.Thread(target=startVideo, args=(pid,), daemon=True)
    p.start()

    startGui()
