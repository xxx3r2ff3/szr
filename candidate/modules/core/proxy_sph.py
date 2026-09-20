import atexit
import threading
from mitmproxy.tools.main import mitmdump
from mitmproxy import http
from proxy_safe import (
    PROXY_PORT,
    check_cert,
    handle_sph,
    set_proxy,
    unset_proxy,
)


# 注册退出时的清理函数
atexit.register(unset_proxy)


def response(flow: http.HTTPFlow) -> None:
    try:
        handle_sph(flow)
    except:
        pass


if __name__ == "__main__":

    # 设置代理
    set_proxy()

    # 启动代理
    p = threading.Thread(target=check_cert)
    p.start()

    # 启动mitmproxy
    mitmdump(
        [
            "-s",
            __file__,  # 使用当前脚本作为拦截脚本
            "-p",
            str(PROXY_PORT),
            "--quiet",  # 静默模式
        ]
    )
