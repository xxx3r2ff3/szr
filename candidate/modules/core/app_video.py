# -*- coding: utf-8 -*-
"""app_video —— R021 重建(源 modules/core/app_video.pyd,T4R,私有集)。

本环境(无 GPU / 依赖就绪但 app_mqtt 门控)下,本模块**导入即失败**,这是契约
登记的唯一可观察单元(static_unreachable 块 + E1-exc 分支,R021 环境域切片)。

E1 证据(VM 内 oracle 只读探针,脚本与输出见
evidence/modules/core__app_video/runtime/):

* ``probe_import.py``(沙箱 + config.ini + 应用 sys.path 形态,模块名
  modules.core.app_video):
  - 结果:failed,SystemExit(无码);
  - 回溯:app_video.py line 18, in init app_video → app_mqtt.py line 23
    → 即 app_video 第 18 行导入 modules.core.app_mqtt,后者在 GPU 门控处
    ``sys.exit()``;
  - stdout:配置链横幅(``gpu trt True``)+ app_mqtt 的
    ``缺少 nvidia显卡，当前电脑不可用``;stderr:pydub RuntimeWarning。
* ``probe_av_chain.py``(逐模块导入定位输出源):
  - ``modules.core.config`` 单导入 → 横幅 ``gpu trt False``(与 R002 一致);
  - ``modules.core.app_infer`` 单导入 → 横幅 ``gpu trt True`` + pydub 告警
    (app_infer 自带同形横幅,E1/其它代理同源结论);
  - ``modules.core.app_mqtt`` 单导入 → ``缺少 nvidia显卡，当前电脑不可用`` +
    SystemExit;
  - ``modules.tencent.hubert`` 在本环境**不存在**(ModuleNotFoundError),
    说明该导入位于 line 18 之后(未被触达),与本候选一致。

因此本候选按"实测导入链"还原:先导入 app_infer(横幅 + pydub 告警),
再导入 app_mqtt(line 18 的门控 → SystemExit)。函数体(add_cors_headers /
doVideoDiy / get_video_duration / is_port_in_use / method_unsupported /
parse_args / publish_work / serve_video / start_video)在本 VM 不可达,
契约已按 static_unreachable 登记,留待 Win10/GPU 就绪环境重开。
"""
import os  # noqa: F401  [pyd imports.txt: os]
import sys  # noqa: F401  [pyd imports.txt: sys]

import bottle  # noqa: F401  [pyd imports.txt: bottle]
import requests  # noqa: F401  [pyd imports.txt: requests]

from modules.core import app_infer  # noqa: F401  [E1 probe_av_chain:横幅 gpu trt True + pydub 告警]
from modules.core import app_mqtt  # noqa: F401  [E1 probe_import:app_video.py:18 → app_mqtt 门控 SystemExit]

import threading
import time

app_port = 3062


def is_port_in_use(port, host="127.0.0.1"):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, int(port))) == 0


def get_video_duration(filename):
    return 0.0


def serve_video(workId, filename):
    temp_dir = os.path.join(os.getcwd(), "works", str(workId))
    return bottle.static_file(filename, root=temp_dir, mimetype="video/mp4")


def start_video(pid=None):
    """启动本地视频服务与任务分发 (Bottle 服务，端口 3062)"""
    def do_request():
        time.sleep(1)
        try:
            requests.get(f"http://127.0.0.1:{app_port}", timeout=2)
        except Exception:
            pass

    threading.Thread(target=do_request, daemon=True).start()
    try:
        # T-026 (R-1D-003): 绑址改回环 127.0.0.1,消除 3062 网卡级对外监听;仅改绑址,不动其他逻辑
        bottle.run(host="127.0.0.1", port=app_port, debug=False, quiet=True)
    except Exception as e:
        print(f"[app_video] 视频服务异常: {e}")

