import atexit
from multiprocessing import Process
import os
from pathlib import Path
import subprocess
import time
from mitmproxy.tools.main import mitmdump
from mitmproxy import http
import psutil

from proxy_safe import (
    PROXY_PORT,
    handle_zbbl,
    install_cert_windows,
    is_cert_installed_windows,
    set_proxy,
    unset_proxy,
)


# 注册退出时的清理函数
atexit.register(unset_proxy)


def request(flow: http.HTTPFlow) -> None:
    handle_zbbl(flow)


def find_live_studio_shortcut():
    """查找启动菜单中的抖音直播伴侣快捷方式"""
    # 启动菜单路径
    start_menu_path = (
        Path(os.environ["ProgramData"])
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
    )

    # 搜索所有快捷方式文件
    for shortcut in start_menu_path.glob("**/*.lnk"):
        # 检查快捷方式名称是否包含关键词
        if "直播伴侣" in shortcut.name or "LiveStudio" in shortcut.name:
            return shortcut
    return None


def start_live_studio():
    """通过快捷方式启动抖音直播伴侣"""
    # 查找快捷方式
    shortcut = find_live_studio_shortcut()
    if not shortcut:
        print("❌ 未找到抖音直播伴侣的快捷方式")
        return False

    try:
        # 使用默认程序打开快捷方式（启动程序）
        subprocess.Popen(["start", "", str(shortcut)], shell=True)
        print(f"✅ 已启动抖音直播伴侣: {shortcut}")
        return True
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False


def start_mitmproxy():
    # 启动mitmproxy并加载拦截脚本
    mitmdump(
        [
            "-s",
            __file__,  # 使用当前脚本作为拦截脚本
            "-p",
            str(PROXY_PORT),
            "--quiet",  # 静默模式
        ]
    )


def is_live_studio_running():
    """通过进程名判断直播伴侣是否已启动"""
    try:
        output = subprocess.check_output(
            'tasklist /fi "imagename eq 直播伴侣.exe" /fo csv', shell=True
        ).decode("gbk")
        return "直播伴侣.exe" in output
    except subprocess.CalledProcessError:
        return False


def is_port_in_use(port):
    """检查端口是否被占用"""
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            return True
    return False


if __name__ == "__main__":
    cert_path = "checkpoints/mitmproxy-ca-cert.cer"  # 证书文件路径
    cert_name = "mitmproxy"  # 证书名称（在 certutil 中显示的名称）

    if os.path.exists(cert_path):
        if not is_cert_installed_windows(cert_name):
            install_cert_windows(cert_path)
        else:
            print("✅ 证书已安装")
    else:
        print("❌ 证书文件不存在")

    # 设置代理
    set_proxy()

    # 启动代理
    p = Process(target=start_mitmproxy, name="mitmproxy")
    p.start()

    while True:
        time.sleep(1)
        if is_port_in_use(PROXY_PORT):
            print("✅ 代理脚本已启动")

            # 启动直播伴侣
            start_live_studio()

            time.sleep(3)
            p.terminate()
            unset_proxy()
            os._exit(0)
        else:
            print("❌ 系统代理未设置")
