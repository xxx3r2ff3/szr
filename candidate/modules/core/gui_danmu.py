# -*- coding: utf-8 -*-
"""gui_danmu —— core__gui_danmu 语义重建(R032 / T4R)。

源:modules/core/gui_danmu.cp310-win_amd64.pyd
    sha256 d3255b5141c60c3c74922c40267540b808bcfcb763c708c2164d7f63e648c141

证据链(oracle 侧实测;脚本/输出见 evidence/modules/core__gui_danmu/):
  probeE_gui_danmu.py          命名空间、签名、ProxySetting 行为、kill_port 杀监听者
  probeG_gui_danmu_stub.py     webview 桩下的窗口路径(确认模块级 liveWindow 才是真源)
  probeH_gui_danmu_window.py   置位 liveWindow 后 getLiveWindow/closeLiveWindow 返回值
  probeI_gui_danmu_poll.py     matchUrlEvalJs 轮询条件(get_current_url().startswith(host))
  probeJ_gui_danmu_urls.py     16 个 start_*_live 的 create_window 实参(标题/尺寸/UA/事件)
  probeK_gui_danmu_class.py    DanmuApi 方法表与 co_firstlineno
  static/pseudocode/           Ghidra 反编译(DAT_ 槽位已解析为 STR("..."))

二进制出处(函数地址 → py 行号):
  kill_port        FUN_180001010 @0x180001010 [py=1850..1996]
  unset_proxy      FUN_180001940 @0x180001940 [py=2091..2240]
  startSphProxy    FUN_180001f10 @0x180001f10 [py=2429..2442]
  closeLiveWindow  FUN_180002770 @0x180002770 [py=2516..2552]
  getLiveWindow    FUN_180002b00 @0x180002b00
  matchUrlEvalJs   FUN_180003470 @0x180003470 [py=80..93]
  start_*_live     FUN_180004310/…/FUN_180014650(16 个)
  模块级注册       FUN_1800155b0(exec 段)+ FUN_1800168b0(PyInit)

未定谳项(见 reports/modules/core__gui_danmu-impl.md §5):
  * 只有 6 个直播系平台的 "host 前缀 + 注入脚本" 二元组能从二进制直接对出;
    其余 10 个平台的 (host, js) 组合是 E2 假说(串表槽位已逐条登记在 _JS_SLOT)。
  * 各 start_*_live 的真窗口/WebView2 路径留给 U 任务(见 UI 捕获链)。
"""
import os
import subprocess
import sys
import time

import psutil
import webview

from winproxy import ProxySetting   # 反编译 FUN_180018be0("winproxy");oracle site-packages 0.3.0a1

# 模块级状态:FUN_1800155b0 exec 段 PyDict_SetItem(globals,"liveWindow"/"liveProxy",None)
liveWindow = None
liveProxy = None

# 移动端 UA(反编译 STR 常量;start_ks2_live / start_baidu_live 走这条)
_UA_IPHONE = ("Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) "
              "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 "
              "Mobile/15E148 Safari/604.1 Edg/129.0.0.0")


def kill_port(port):
    """杀掉监听 `port` 的进程(反编译 FUN_180001010 @py1850-1996)。

    for conn in psutil.net_connections():
        if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
            psutil.Process(conn.pid).terminate()

    证据:probeE call_kill_port_live —— 子进程监听 53030 后调用本函数,
    受害者进程退出码 15(victim_alive=False),函数返回 None。
    """
    for conn in psutil.net_connections():
        if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
            psutil.Process(conn.pid).terminate()


def unset_proxy():
    """关闭系统代理并落注册表(反编译 FUN_180001940 @py2091-2240)。

    ps = ProxySetting(); ps.enable = False; ps.registry_write()
    print("✅ 已清除系统代理")
    证据:probeE call_unset_proxy / probeH C_proxysetting(stdout 与本串逐字一致)。
    """
    ps = ProxySetting()
    ps.enable = False
    ps.registry_write()
    print("✅ 已清除系统代理")


def startSphProxy():
    """起 modules/core/proxy_sph.py 无窗口子进程(反编译 FUN_180001f10 @py2429-2442)。

    if not liveProxy:
        subprocess.Popen([sys.executable, "modules/core/proxy_sph.py"],
                         creationflags=subprocess.CREATE_NO_WINDOW)
        print("✅ 开始代理")
        liveProxy = True
    证据:probeE/probeH s4_popens(实参 [python.exe, 'modules/core/proxy_sph.py'],
    creationflags=134217728)且二次调用不再打印。
    """
    global liveProxy
    if not liveProxy:
        subprocess.Popen([str(sys.executable), "modules/core/proxy_sph.py"],
                         creationflags=subprocess.CREATE_NO_WINDOW)
        print("✅ 开始代理")
        liveProxy = True


def closeLiveWindow():
    """关闭直播窗口并复位模块状态(反编译 FUN_180002b00 @py=2516..2552)。

    if liveProxy: unset_proxy()
    if liveWindow is not None: liveWindow.destroy()
    liveWindow = None
    证据:probeH A_close_trace=[destroy]、A_liveWindow_after_close=None、
    A_destroyed=1;golden gui_danmu__window_lifecycle state.destroyed=1。
    """
    global liveWindow
    if liveProxy:
        unset_proxy()
    if liveWindow is not None:
        liveWindow.destroy()
    liveWindow = None


def getLiveWindow():
    """返回模块级 liveWindow(反编译 FUN_180002b00;probeH 实测返回同一对象)。"""
    return liveWindow


def liveWindowClose():
    """窗口 closed 事件处理器(反编译 FUN_180002770 @py=61)。

    if liveProxy: unset_proxy()
    liveWindow = None          # 注意:不调 destroy(与 closeLiveWindow 不同)
    证据:golden gui_danmu__liveWindowClose —— state.destroyed=0 且
    liveWindow_is_none=True、stdout='✅ 已清除系统代理'。
    """
    global liveWindow
    if liveProxy:
        unset_proxy()
    liveWindow = None


def matchUrlEvalJs(host, js):
    """等窗口 URL 命中 host 后执行 js(反编译 FUN_180003470 @py80-93)。

    while True:
        time.sleep(1)
        url = liveWindow.get_current_url()
        print(url)
        if url.startswith(host):
            liveWindow.evaluate_js(js)
            break
    证据:probeI —— host='https://live.douyin.com/' 时一次 sleep 后命中,
    命中后调用 get_current_url→evaluate_js;host='douyin' 时反复 sleep 不命中,
    证明判据是 startswith(host) 而不是 in/host 子串。
    """
    while True:
        time.sleep(1)
        url = liveWindow.get_current_url()
        print(url)
        if url.startswith(host):
            liveWindow.evaluate_js(js)
            break


def evaluate_js(js):
    """evaluate_js 型 start_*_live 的执行步骤(反编译各 FUN_* @py≈3463-3470)。

    原版在方法体内直接内联 `liveWindow.evaluate_js(<js 常量>)`;此处抽为同名
    模块函数以便同一套 create_window 流程复用,语义等价。
    """
    liveWindow.evaluate_js(js)


def _prepare_window(width, height, user_agent=None):
    """create_window 前置:反编译各 start_*_live @py=0xd3b。

    webview._settings["user_agent"] = <ua 或 None>
    """
    webview._settings["user_agent"] = user_agent


def _open_window(js_api, title, url, width, height, user_agent=None):
    """公共窗口流程(反编译各 start_*_live 的同一骨架 @py≈3402-3500)。

    if liveWindow: liveWindow.destroy()
    webview._settings["user_agent"] = user_agent
    liveWindow = webview.create_window(title, url, width=…, height=…,
                                      resizable=False, js_api=self)
    liveWindow.events.closed += liveWindowClose
    证据:probeJ 逐方法实测 create_window 的 args/kwargs 与 closed 事件绑定。
    """
    global liveWindow
    if liveWindow is not None:
        liveWindow.destroy()
    _prepare_window(width, height, user_agent)
    liveWindow = webview.create_window(title, url, width=width, height=height,
                                       resizable=False, js_api=js_api)
    liveWindow.events.closed += liveWindowClose
    return liveWindow


class DanmuApi(object):
    """16 个平台直播窗口启动器(create_window 实参全部由 probeJ 实测)。"""

    def start_dy_live(self, url):
        _open_window(self, "抖音直播（不允许最小化）", url, 800, 900)
        matchUrlEvalJs(_HOST["start_dy_live"], JS["start_dy_live"])

    def start_tb_live(self, url):
        _open_window(self, "淘宝直播（不允许最小化）", url, 600, 900)
        matchUrlEvalJs(_HOST["start_tb_live"], JS["start_tb_live"])

    def start_pdd_live(self, url):
        _open_window(self, "拼多多直播（不允许最小化）", url, 600, 900)
        matchUrlEvalJs(_HOST["start_pdd_live"], JS["start_pdd_live"])

    def start_mt_live(self, url):
        _open_window(self, "美团直播（不允许最小化）", url, 600, 900)
        matchUrlEvalJs(_HOST["start_mt_live"], JS["start_mt_live"])

    def start_ks_live(self, url):
        _open_window(self, "快手直播（不允许最小化）", url, 1080, 900)
        evaluate_js(JS["start_ks_live"])

    def start_ks2_live(self, url):
        _open_window(self, "快手直播（不允许最小化）", url, 600, 900, _UA_IPHONE)
        evaluate_js(JS["start_ks2_live"])

    def start_xhs_live(self, url):
        _open_window(self, "小红书直播（不允许最小化）", url, 1080, 900)
        matchUrlEvalJs(_HOST["start_xhs_live"], JS["start_xhs_live"])

    def start_jd_live(self, url):
        _open_window(self, "京东直播（不允许最小化）", url, 1080, 900)
        matchUrlEvalJs(_HOST["start_jd_live"], JS["start_jd_live"])

    def start_sph_live(self, url):
        live = _open_window(self, "视频号直播（不允许最小化）", url, 1080, 900)
        live.events.loaded += startSphProxy

    def start_tk_live(self, url):
        _open_window(self, "Tiktok", url, 1280, 860)
        evaluate_js(JS["start_tk_live"])

    def start_youtube_live(self, url):
        _open_window(self, "YouTube", url, 1280, 860)
        evaluate_js(JS["start_youtube_live"])

    def start_shopee_live(self, url):
        _open_window(self, "Shopee", url, 600, 900)
        evaluate_js(JS["start_shopee_live"])

    def start_yundong_live(self, url):
        _open_window(self, "云动系统", url, 600, 900)
        evaluate_js(JS["start_yundong_live"])

    def start_baidu_live(self, url):
        _open_window(self, "百度", url, 600, 900, _UA_IPHONE)
        evaluate_js(JS["start_baidu_live"])

    def start_bilibili_live(self, url):
        _open_window(self, "B站", url, 800, 900)
        evaluate_js(JS["start_bilibili_live"])

    def start_alibaba_live(self, url):
        _open_window(self, "阿里巴巴国际站", url, 800, 900)
        evaluate_js(JS["start_alibaba_live"])


# ---------------- 注入脚本原文(Cython 串表槽位;槽位号即二进制出处) ----------------
JS = {
    'start_dy_live': "\nfunction monitor() {\n    console.log(\"load js\")\n    let videos = document.querySelectorAll('video');\n    if (!videos.length) {\n        setTimeout(monitor, 3000);\n        return\n    }\n    for (const video of videos) {\n        video.muted = true;\n        video.pause()\n    }\n\n    setInterval(function() { \n        // 2025/05/30\n        try {\n            document.querySelector(\"#LeftBackgroundLayout\").style.display = \"none\";\n            document.querySelector(\"#RightBackgroundLayout\").style.width = \"850px\";\n        } catch(err) {\n            document.getElementsByClassName(\"__playerIsFull\")[0].style.display = \"none\";\n            document.getElementsByClassName(\"d8cD2XWD\")[0].style.width = \"850px\";\n        }\n\n        const stopBtn = Array.from(document.querySelectorAll('button'))\n            .find(el => el.innerText == '任意操作可恢复播放');\n        if (stopBtn) {\n            stopBtn.click()\n        }\n\n        const goDiv = Array.from(document.querySelectorAll('div'))\n            .find(el => el.innerText == '继续播放');\n        if (goDiv) {\n            goDiv.click()\n        }\n        \n        const stopDiv = Array.from(document.querySelectorAll('div[elementtiming=\"element-timing\"]'))\n            .find(el => el.innerText == '恢复播放');\n        if (stopDiv) {\n            stopDiv.click()\n        }\n\n        const videos = document.querySelectorAll('video,audio');\n        for (const video of videos) {\n            video.muted = true;\n            video.pause();\n        }\n    }, 3000);\n\n    let style = document.createElement('style');\n    style.innerHTML = `\n    .k3s5qMFF {\n        display: none;\n    }\n\n    .webcast-chatroom___nickname {\n        padding-left: 20px;\n    }\n    \n    span.u2QdU6ht {\n        padding-left: 20px;\n    }\n\n    .webcast-chatroom___content-with-emoji-emoji {\n        display: none;\n    }\n    `;\n    document.head.appendChild(style);\n\n    try {\n        let chat_list = document.querySelector('.webcast-chatroom___items').firstChild;\n        let nickname = document.querySelector(\".__leftContainer\").querySelector('[data-e2e=\"live-room-nickname\"]').text\n        \n        const observer = new MutationObserver((mutationsList, observer) => {\n            for (const mutation of mutationsList) {\n                if (mutation.addedNodes.length > 0) {\n                    const node = mutation.addedNodes[0].firstChild.firstChild;\n                    if (!node) {\n                        continue\n                    }\n\n                    const html = node.innerHTML;\n                    if (!html) {\n                        continue\n                    }\n                    console.log(html, node.childNodes.length);\n\n                    if (node.childNodes.length < 3) {\n                        continue\n                    }\n\n                    let level = node.firstChild.innerHTML;\n                    let name = node.childNodes[1].innerText;\n                    if (name.endsWith(\"：\")) {\n                        name = name.slice(0, -1);\n                    }\n                    console.log(name, nickname);\n                    if (name.trim() == nickname){\n                        continue\n                    }\n                    const content = node.lastChild.innerText;\n                    window.pywebview.api.live_push_event({\n                        type: \"chat\",\n                        html: encodeURIComponent(html),\n                        level: encodeURIComponent(level),\n                        name: encodeURIComponent(name),\n                        content: encodeURIComponent(content)\n                    })\n                }\n            }\n        });\n        const config = { attributes: false, childList: true, subtree: true, characterData: false };\n        observer.observe(chat_list, config);\n\n        const bottom_list = document.getElementsByClassName('webcast-chatroom___bottom-message')[0];\n        console.log(bottom_list);\n        const bottomObserver = new MutationObserver((mutationsList, observer) => {\n            for (const mutation of mutationsList) {\n                if (mutation.addedNodes.length > 0) {\n                    const node = mutation.addedNodes[0].firstChild;\n                    if (!node) {\n                        continue\n                    }\n\n                    const html = node.innerHTML;\n                    if (!html) {\n                        continue\n                    }\n                    console.log(html);\n\n                    if (node.childNodes.length < 3) {\n                        continue\n                    }\n\n                    let level = node.firstChild.innerHTML;\n                    let name = node.childNodes[1].innerText;\n                    if (name.endsWith(\"：\")) {\n                        name = name.slice(0, -1);\n                    }\n                    console.log(name, nickname);\n                    if (name.trim() == nickname){\n                        continue\n                    }\n                    \n                    const content = node.lastChild.innerText;\n                    window.pywebview.api.live_push_event({\n                        type: \"bottom\",\n                        html: encodeURIComponent(html),\n                        level: encodeURIComponent(level),\n                        name: encodeURIComponent(name),\n                        content: encodeURIComponent(content)\n                    })\n                }\n            }\n        });\n        const bottomConfig = { attributes: false, childList: true, subtree: true, characterData: false };\n        bottomObserver.observe(bottom_list, bottomConfig);\n    } catch(err) {\n        // 2025/05/09\n        try {\n            let chat_list = document.querySelector('.webcast-chatroom___list>div>div');\n            let nickname = document.querySelector(\".__leftContainer\").querySelector('[data-e2e=\"live-room-nickname\"]').text\n            \n            const observer = new MutationObserver((mutationsList, observer) => {\n                for (const mutation of mutationsList) {\n                    if (mutation.addedNodes.length > 0) {\n                        const node = mutation.addedNodes[0].firstChild.firstChild;\n                        if (!node) {\n                            continue\n                        }\n\n                        const html = node.innerHTML;\n                        if (!html) {\n                            continue\n                        }\n                        console.log(html);\n                        \n                        let name, content;\n                        const node_item = node.firstChild.cloneNode(true);\n                        const hasnew = node.classList.contains('webcast-chatroom___bottom-message')\n                        if (hasnew) {\n                            name = node_item.firstChild.childNodes[1].innerText;\n                            content = node_item.firstChild.childNodes[2].innerText\n                        } else {\n                            name = node_item.childNodes[1].innerText;\n                            content = node_item.lastChild.innerText\n                        }\n                        if (name.endsWith(\"：\")) {\n                            name = name.slice(0, -1);\n                        }\n                        \n                        console.log(nickname, name, content);\n                        if (name.trim() == nickname){\n                            continue\n                        }\n                        \n                        window.pywebview.api.live_push_event({\n                            type: \"chat\",\n                            html: encodeURIComponent(html),\n                            name: encodeURIComponent(name),\n                            content: encodeURIComponent(content)\n                        })\n                    }\n                }\n            });\n            const config = { attributes: false, childList: true, subtree: true, characterData: false };\n            observer.observe(chat_list, config);\n        } catch(err) {\n            setTimeout(monitor, 3000);\n            return\n        }\n    }\n    \n}\nsetTimeout(() => {\n    monitor();\n}, 3000);         \n                               ",
    'start_tb_live': "\nfunction monitor() {\n    console.log(\"load js\")\n    let videos = document.querySelectorAll('video');\n    if (!videos.length) {\n        setTimeout(monitor, 3000);\n        return\n    }\n    for (const video of videos) {\n        video.pause()\n    }\n\n    let chat_list;\n    try {\n        chat_list = document.getElementsByClassName('chat-history')[0].firstChild;\n    } catch(err) {\n        setTimeout(monitor, 3000);\n        return\n    }\n    const observer = new MutationObserver((mutationsList, observer) => {\n        for (const mutation of mutationsList) {\n            if (mutation.addedNodes.length > 0) {\n                const node = mutation.addedNodes[0];\n                if (!node) {\n                    continue\n                }\n\n                const html = node.innerHTML;\n                if (!html) {\n                    continue\n                }\n                console.log(html);\n                \n                const name = node.getElementsByClassName('username')[0].innerText;\n                const content = node.getElementsByClassName('comment')[0].innerText;\n                window.pywebview.api.live_push_event({\n                    type: \"chat\",\n                    name: encodeURIComponent(name),\n                    content: encodeURIComponent(content)\n                })\n            }\n        }\n    });\n    const config = { attributes: false, childList: true, subtree: true, characterData: false };\n    observer.observe(chat_list, config);\n}\nsetTimeout(() => {\n    monitor();\n}, 3000);         \n                               ",
    'start_pdd_live': "\nfunction monitor() {\n    console.log(\"load js\")\n\n    let chat_list;\n    try {\n        chat_list = document.querySelector('.comments');\n        if(!chat_list){\n            throw new Error('no chat_list');\n        }\n    } catch(err) {\n        setTimeout(monitor, 3000);\n        return\n    }\n    const observer = new MutationObserver((mutationsList, observer) => {\n        for (const mutation of mutationsList) {\n            if (mutation.addedNodes.length > 0) {\n                const node = mutation.addedNodes[0];\n                if (!node) {\n                    continue\n                }\n\n                const html = node.innerHTML;\n                if (!html) {\n                    continue\n                }\n                console.log(html);\n\n                const spans = node.getElementsByTagName('span');\n                let name = spans[spans.length - 2].innerText;\n                if (name.endsWith(\": \")) {\n                    name = name.slice(0, -2);\n                }\n\n                const content = spans[spans.length - 1].innerText;\n                window.pywebview.api.live_push_event({\n                    type: \"chat\",\n                    name: encodeURIComponent(name),\n                    content: encodeURIComponent(content)\n                })\n            }\n        }\n    });\n    const config = { attributes: false, childList: true, subtree: true, characterData: false };\n    observer.observe(chat_list, config);\n}\n                       \nsetTimeout(() => {\n    monitor();\n}, 3000);\n",
    'start_mt_live': "\nfunction monitor() {\n\tconsole.log(\"load js\")\n    let hasPlay = document.querySelector('.deplayer-play-icon')\n    if (!hasPlay) {\n        setTimeout(monitor, 3000);\n        return;\n    } else {\n        hasPlay.click()\n    }\n    \n    let videos = document.querySelectorAll('video');\n    if (!videos.length) {\n        setTimeout(monitor, 3000);\n        return;\n    }\n    for (const video of videos) {\n        video.pause();\n    }\n    \n    setInterval(function() { \n        try {\n            document.querySelector(\".s-dialog-wrap.dialog-wrapper\").style.display = \"none\";\n        } catch(err) {}\n    }, 3000);\n\n    const chat_list = document.querySelector('.chat-list');\n    if (!chat_list) {\n        setTimeout(monitor, 3000);\n        return\n    }\n    \n    const author = document.querySelector('.author-name').innerText\n    const observer = new MutationObserver((mutationsList, observer) => {\n        for (const mutation of mutationsList) {\n            if (mutation.addedNodes.length > 0) {\n                const node = mutation.addedNodes[0];\n                if (!node) {\n                    continue\n                }\n\n                const html = node\n                if (!html) {\n                    continue\n                }\n                console.log(html);\n\n                let name = html.querySelector('.user-name').innerText\n                if (name.endsWith(\"：\")) {\n                    name = name.slice(0, -1);\n                }\n                if (name == author) {\n                    continue\n                }\n                const content = html.querySelector('.msg').innerText\n                console.log(name, content)\n                window.pywebview.api.live_push_event({\n                    type: \"chat\",\n                    name: encodeURIComponent(name),\n                    content: encodeURIComponent(content)\n                })\n            }\n        }\n    });\n    const chatConfig = { attributes: false, childList: true, subtree: true, characterData: false };\n    observer.observe(chat_list, chatConfig);\n}\nsetTimeout(() => {\n\tmonitor();\n}, 3000);        \n                               ",
    'start_xhs_live': "\nfunction monitor() {\n    console.log(\"load js\")\n\n    let chat_list;\n    try {\n        chat_list = document.getElementById('msgBox');\n    } catch(err) {\n        console.log(err)\n        setTimeout(monitor, 3000);\n        return\n    }\n    const observer = new MutationObserver((mutationsList, observer) => {\n        for (const mutation of mutationsList) {\n            if (mutation.addedNodes.length > 0) {\n                const node = mutation.addedNodes[0];\n                if (!node) {\n                    continue\n                }\n\n                const html = node.innerHTML;\n                if (!html) {\n                    continue\n                }\n                console.log(html);\n\n                let spans = node.getElementsByTagName('span');\n                let name = spans[0].innerText;\n                if (name.endsWith(\"：\")) {\n                    name = name.slice(0, -1);\n                }\n\n                const content = spans[1].innerText;\n                console.log(name, content)\n                window.pywebview.api.live_push_event({\n                    type: \"chat\",\n                    name: encodeURIComponent(name),\n                    content: encodeURIComponent(content)\n                })\n            }\n        }\n    });\n    const config = { attributes: false, childList: true, subtree: true, characterData: false };\n    observer.observe(chat_list, config);\n\n    const bottom_list = chat_list.parentNode.previousElementSibling;\n    console.log(bottom_list);\n    const bottomObserver = new MutationObserver((mutationsList, observer) => {\n        for (const mutation of mutationsList) {\n            console.log(mutation)\n            const node = mutation.addedNodes[0];\n            if (!node) {\n                continue\n            }\n            console.log(node)\n\n            const spans = node.getElementsByTagName('span');\n            const name = spans[0].innerText;\n            const content = spans[2].innerText;\n            if (!name) {\n                continue\n            }\n            console.log(name, content)\n            window.pywebview.api.live_push_event({\n                type: \"bottom\",\n                name: encodeURIComponent(name),\n                content: encodeURIComponent(content)\n            })\n        }\n    });\n    const bottomConfig = { attributes: false, childList: true, subtree: true, characterData: true };\n    bottomObserver.observe(bottom_list, bottomConfig);\n}\nsetTimeout(() => {\n    monitor();\n}, 3000);\n",
    'start_jd_live': "\nfunction monitor() {\n    if (location.pathname != '/my/room'){\n        setTimeout(monitor, 3000);\n        return\n    }\n    console.log(\"load js\")\n\n    let chat_list;\n    try {\n        chat_list = document.querySelector('#all>div:nth-child(2)');\n    } catch(err) {\n        setTimeout(monitor, 3000);\n        return\n    }\n    const observer = new MutationObserver((mutationsList, observer) => {\n        for (const mutation of mutationsList) {\n            if (mutation.addedNodes.length > 0) {\n                const node = mutation.addedNodes[0];\n                if (!node) {\n                    continue\n                }\n\n                const html = node.innerHTML;\n                if (!html) {\n                    continue\n                }\n                console.log(html);\n\n                const hasmsg = node.querySelector('.antd-pro-pages-control-panel-msg-components-the-msg-tabs-viewer-info-nickname')\n                if (hasmsg) {\n                    const name = node.querySelector('.antd-pro-pages-control-panel-msg-components-the-msg-tabs-viewer-info-nickname').innerText;\n                    const content = node.querySelector('.antd-pro-pages-control-panel-msg-components-the-msg-tabs-chat-item-content').innerText;\n                    window.pywebview.api.live_push_event({\n                        type: \"chat\",\n                        name: encodeURIComponent(name),\n                        content: encodeURIComponent(content)\n                    })\n                } else {\n                    let name = node.querySelector('.antd-pro-pages-control-panel-msg-components-the-msg-tabs-chat-item-content').innerText;\n                    name = name.replace('来了', '')\n                    const content = node.querySelector('.antd-pro-pages-control-panel-msg-components-the-msg-tabs-chat-item-tag').innerText;\n                    window.pywebview.api.live_push_event({\n                        type: \"chat\",\n                        name: encodeURIComponent(name),\n                        content: encodeURIComponent(content)\n                    })\n                }\n            }\n        }\n    });\n    const config = { attributes: false, childList: true, subtree: true, characterData: false };\n    observer.observe(chat_list, config);\n}\n                       \nsetTimeout(() => {\n    monitor();\n}, 3000);\n",
    'start_ks_live': "\n    function monitor() {\n        console.log(\"load js\")\n        let videos = document.querySelectorAll('video');\n        if (!videos.length) {\n            setTimeout(monitor, 3000);\n            return;\n        }\n        for (const video of videos) {\n            video.pause();\n        }\n\n        const chat_list = document.querySelector('#chat-items');\n        const chat_ids = []\n        const observer = new MutationObserver((mutationsList, observer) => {\n            for (const mutation of mutationsList) {\n                if (mutation.addedNodes.length > 0) {\n                    const node = mutation.addedNodes[0];\n                    if (!node) {\n                        continue\n                    }\n                    if (!node.nextElementSibling) {\n                        continue\n                    }\n\n                    const html = node.nextElementSibling.lastElementChild\n                    if (!html) {\n                        continue\n                    }\n                    console.log(html);\n\n                    if (chat_ids.includes(node.nextElementSibling.id)) {\n                        continue\n                    }\n                    chat_ids.push(node.nextElementSibling.id)\n                    if (chat_ids.length > 5) {\n                        chat_ids.shift();\n                    }\n\n                    let name = html.firstElementChild.firstElementChild.firstElementChild.innerText\n                    if (name.endsWith(\":\")) {\n                        name = name.slice(0, -1);\n                    }\n                    const content = html.lastElementChild.innerText\n                    console.log(name, content)\n                    window.pywebview.api.live_push_event({\n                        type: \"chat\",\n                        name: encodeURIComponent(name),\n                        content: encodeURIComponent(content)\n                    })\n                }\n            }\n        });\n        const chatConfig = { attributes: false, childList: true, subtree: false, characterData: false };\n        observer.observe(chat_list, chatConfig);\n    }\n    setTimeout(() => {\n        monitor();\n    }, 3000);        \n                                ",
    'start_ks2_live': "\n\nfunction monitorLast() {\n    const elements = document.querySelectorAll('div.live-comment-item');\n    const lastElement = elements[elements.length - 1];\n    const lastName = lastElement.querySelector('span.comment-item__name')\n    const lastText = lastElement.querySelector('span.comment-item__text')\n    const observer = new MutationObserver((mutationsList, observer) => {\n        for(let mutation of mutationsList) {\n            const name = lastName.innerText\n            const content = mutation.target.innerText\n            console.log(name, content)\n            window.pywebview.api.live_push_event({\n                type: \"chat\",\n                name: encodeURIComponent(name),\n                content: encodeURIComponent(content)\n            })\n        }\n    });\n\n    const config = { attributes: false, childList: true, subtree: true, characterData: false };\n    observer.observe(lastText, config);\n}\nfunction monitor() {\n    console.log(\"load js\")\n\n    let videos = document.querySelectorAll('video');\n    if (!videos.length) {\n        setTimeout(monitor, 3000);\n        return\n    }\n    for (const video of videos) {\n        video.muted = true;\n        video.pause();\n    }\n\n    const chat_list = document.querySelector('div.live-comment.live-player-comment');\n    if (!chat_list) {\n        setTimeout(monitor, 3000);\n        return\n    }\n    const observer = new MutationObserver((mutationsList, observer) => {\n        for (const mutation of mutationsList) {\n            console.log(mutation)\n\n            if (mutation.type == 'childList' && mutation.addedNodes.length > 0) {\n                const node = mutation.addedNodes[0];\n                if (!node) {\n                    continue\n                }\n\n                let name = node.getElementsByClassName('comment-item__name')[0].innerText;\n                if (name.endsWith(\":\")) {\n                    name = name.slice(0, -1);\n                }\n\n                const content = node.getElementsByClassName('comment-item__text')[0].innerText;\n                console.log(name, content)\n                window.pywebview.api.live_push_event({\n                    type: \"chat\",\n                    name: encodeURIComponent(name),\n                    content: encodeURIComponent(content)\n                })\n            }\n            \n            const items = document.querySelectorAll('div.live-comment-item')\n            if (items.length >= 8) {\n                observer.disconnect();\n                console.log('start monitor last')\n                monitorLast()\n            }\n        }\n    });\n    const config = { attributes: false, childList: true, subtree: false, characterData: false };\n    observer.observe(chat_list, config);\n}\nfunction start() {\n    let startBtn = document.querySelector(\"div.video-poster-play\")\n    if (!startBtn) {\n        startBtn = document.querySelector(\"div.live-player-poster\")\n        if (!startBtn) {\n            setTimeout(start, 3000);\n            return\n        }\n    }\n    \n    startBtn.click()\n    monitor()\n}\nsetTimeout(() => {\n    start();\n}, 3000);         \n                               ",
    'start_sph_live': "\nfunction monitor() {\n\tconsole.log(\"load js\")\n    let videos = document.querySelectorAll('video');\n    if (!videos.length) {\n        setTimeout(monitor, 3000);\n        return;\n    }\n    for (const video of videos) {\n        video.pause();\n    }\n\n    const chatframe = document.getElementById('chatframe');\n    if (!chatframe) {\n        setTimeout(monitor, 3000);\n        return;\n    }\n    \n    try {\n        chatframe.onload = function() {\n            const iframeDoc = chatframe.contentDocument || chatframe.contentWindow.document;\n            const chat_list = iframeDoc.querySelector('#items');\n            if (!chat_list) {\n                setTimeout(monitor, 3000);\n                return;\n            }\n            \n            const observer = new MutationObserver((mutationsList, observer) => {\n                for (const mutation of mutationsList) {\n                    if (mutation.addedNodes.length > 0) {\n                        const node = mutation.addedNodes[0];\n                        if (!node) continue;\n                        if (!node.querySelector) continue;\n                        \n                        const name = node.querySelector('#author-name').innerText;\n                        const content = node.querySelector('#message').innerText;\n                        window.pywebview.api.live_push_event({\n                            type: \"chat\",\n                            name: encodeURIComponent(name),\n                            content: encodeURIComponent(content)\n                        });\n                    }\n                }\n            });\n            \n            const chatConfig = {\n                attributes: false,\n                childList: true,\n                subtree: true,\n                characterData: false\n            };\n            observer.observe(chat_list, chatConfig);\n        };\n        \n        if (chatframe.contentDocument.readyState === 'complete') {\n            chatframe.onload();\n        }\n    } catch (e) {\n        setTimeout(monitor, 3000);\n    }\n}\nsetTimeout(() => {\n\tmonitor();\n}, 3000);        \n                               ",
    'start_tk_live': "\nfunction monitor() {\n    console.log(\"load js\")\n\n    let videos = document.querySelectorAll('video');\n    if (!videos.length) {\n        setTimeout(monitor, 3000);\n        return\n    }\n    for (const video of videos) {\n        video.pause()\n    }\n\n    const chat_list = document.querySelector('.relative.w-full > .absolute.w-full.top-0.left-0');\n    if (!chat_list) {\n        setTimeout(monitor, 3000);\n        return\n    }\n    const observer = new MutationObserver((mutationsList, observer) => {\n        for (const mutation of mutationsList) {\n            if (mutation.addedNodes.length > 0) {\n                const node = mutation.addedNodes[0].childNodes[0];\n                if (!node) {\n                    continue\n                }\n\n                const html = node.childNodes[1];\n                if (!html) {\n                    continue\n                }\n                console.log(html);\n\n                const name = html.firstChild.innerText\n                const content = html.lastChild.innerText || html.lastChild.textContent.trim();\n                window.pywebview.api.live_push_event({\n                    type: \"chat\",\n                    name: encodeURIComponent(name),\n                    content: encodeURIComponent(content)\n                })\n            }\n        }\n    });\n    const chatConfig = { attributes: false, childList: true, subtree: true, characterData: false };\n    observer.observe(chat_list, chatConfig);\n\n    const bottom_list = document.querySelector('.w-full.h-auto.overflow-hidden.flex-shrink-0');\n    console.log(bottom_list);\n    const bottomObserver = new MutationObserver((mutationsList, observer) => {\n        for (const mutation of mutationsList) {\n            if (mutation.addedNodes.length > 0) {\n                const node = mutation.addedNodes[0].firstChild;\n                if (!node) {\n                    continue\n                }\n\n                const html = node.innerHTML;\n                if (!html) {\n                    continue\n                }\n                console.log(html);\n\n                const name = node.lastChild.firstChild.innerText || ''\n                const content = node.lastChild.lastChild.innerText || node.lastChild.lastChild.textContent.trim();\n                console.log(name, content)\n                window.pywebview.api.live_push_event({\n                    type: \"bottom\",\n                    name: encodeURIComponent(name),\n                    content: encodeURIComponent(content)\n                })\n            }\n        }\n    });\n    const bottomConfig = { attributes: false, childList: true, subtree: true, characterData: false };\n    bottomObserver.observe(bottom_list, bottomConfig);\n}\nsetTimeout(() => {\n    monitor();\n}, 3000);         \n                               ",
    'start_youtube_live': "\nfunction monitor() {\n\tconsole.log(\"load js\")\n\n    const currentPath = window.location.pathname;\n    if (currentPath != '/live/live-detail.htm') {\n        setTimeout(monitor, 3000);\n        return\n    }\n    \n    const chat_list = document.querySelector('#__qiankun_microapp_wrapper_for_live_interactive_micro_app__').shadowRoot.querySelector('.comment-content>div')\n    if (!chat_list) {\n        setTimeout(monitor, 3000);\n        return\n    }\n    \n    window.scrollTo(document.body.scrollWidth, 0);\n    const observer = new MutationObserver((mutationsList, observer) => {\n        for (const mutation of mutationsList) {\n            console.log(mutation);\n            if (mutation.addedNodes.length > 0) {\n                const node = mutation.addedNodes[0];\n                if (!node) {\n                    continue\n                }\n\n                const html = node\n                if (!html) {\n                    continue\n                }\n                console.log(html);\n\n                let name = chat_list.firstChild.querySelector('.uiH6VIKzx_pcZSPxbNxI').innerText\n                if (name.endsWith(\"：\")) {\n                    name = name.slice(0, -1);\n                }\n\n                const content = chat_list.firstChild.querySelector('.Cq70Ocarno3kH_uI7yXP').innerText\n                console.log(name, content)\n                window.pywebview.api.live_push_event({\n                    type: \"chat\",\n                    name: encodeURIComponent(name),\n                    content: encodeURIComponent(content)\n                })\n            }\n        }\n    });\n    const chatConfig = { attributes: false, childList: true, subtree: false, characterData: false };\n    observer.observe(chat_list, chatConfig);\n}\nsetTimeout(() => {\n\tmonitor();\n}, 3000);        \n                               ",
    'start_shopee_live': "\nfunction monitor(check_sc = true) {\n    console.log(\"load js\")\n    let videos = document.querySelectorAll('video');\n    if (!videos.length) {\n        setTimeout(monitor, 3000);\n        return;\n    }\n    for (const video of videos) {\n        video.pause();\n    }\n\n    if (check_sc) {\n        const find = document.querySelector('[class^=\"MMCVideo__Container-sc-\"]');\n        if (!find) {\n            console.log('no found MMCVideo__Container-sc-')\n            setTimeout(monitor, 3000);\n            return\n        }\n        find.click()\n    }\n\n    const chat_list = document.querySelector('[class^=\"Danmaku__ScrollContainer\"], [class*=\" Danmaku__ScrollContainer\"]');\n    if (!chat_list) {\n        setTimeout(() => {\n            monitor(false)\n        }, 3000);\n        return\n    }\n\n    videos = document.querySelectorAll('video');\n    for (const video of videos) {\n        video.pause();\n    }\n    const observer = new MutationObserver((mutationsList, observer) => {\n        for (const mutation of mutationsList) {\n            if (mutation.addedNodes.length > 0) {\n                const node = mutation.addedNodes[0];\n                if (!node) {\n                    continue\n                }\n\n                const html = node\n                if (!html) {\n                    continue\n                }\n                console.log(html);\n\n                let name = html.firstChild.innerText\n                if (name.endsWith(\":\")) {\n                    name = name.slice(0, -1);\n                }\n                const content = html.lastChild.innerText || html.lastChild.textContent.trim();\n                console.log(name, content)\n                window.pywebview.api.live_push_event({\n                    type: \"chat\",\n                    name: encodeURIComponent(name),\n                    content: encodeURIComponent(content)\n                })\n            }\n        }\n    });\n    const chatConfig = { attributes: false, childList: true, subtree: true, characterData: false };\n    observer.observe(chat_list, chatConfig);\n}\nsetTimeout(() => {\n    monitor();\n}, 3000);\n",
    'start_yundong_live': "\nfunction monitor() {\n    console.log(\"load js\")\n    setInterval(function() { \n        const btns = document.querySelectorAll('div[class=\"close-btn\"]')\n        for (const btn of btns) {\n            btn.click()\n        }\n    }, 3000);\n\n    let chat_list;\n    try {\n        chat_list = document.getElementsByClassName('message-list')[0];\n    } catch(err) {\n        console.log(err)\n        setTimeout(monitor, 3000);\n        return\n    }\n\n    const observer = new MutationObserver((mutationsList, observer) => {\n        for (const mutation of mutationsList) {\n            if (mutation.addedNodes.length > 0) {\n                const node = mutation.addedNodes[0];\n                if (!node) {\n                    continue\n                }\n\n                const html = node.innerHTML;\n                if (!html) {\n                    continue\n                }\n                console.log(html);\n\n                let name = node.getElementsByClassName('name')[0].innerText;\n                if (name.endsWith(\":\")) {\n                    name = name.slice(0, -1);\n                }\n\n                const content = node.getElementsByClassName('content')[0].innerText;\n                window.pywebview.api.live_push_event({\n                    type: \"chat\",\n                    name: encodeURIComponent(name),\n                    content: encodeURIComponent(content)\n                })\n            }\n        }\n    });\n    const config = { attributes: false, childList: true, subtree: true, characterData: false };\n    observer.observe(chat_list, config);\n\n    const bottom_list = document.getElementById('userActionMessageBox');\n    console.log(bottom_list);\n    const bottomObserver = new MutationObserver((mutationsList, observer) => {\n        for (const mutation of mutationsList) {\n            console.log(mutation)\n            if (mutation.type != 'characterData') {\n                continue\n            }\n\n            const name = mutation.target.nodeValue;\n            const content = bottom_list.querySelector(\"div.content\").innerText;\n            window.pywebview.api.live_push_event({\n                type: \"bottom\",\n                name: encodeURIComponent(name),\n                content: encodeURIComponent(content)\n            })\n        }\n    });\n    const bottomConfig = { attributes: false, childList: true, subtree: true, characterData: true };\n    bottomObserver.observe(bottom_list, bottomConfig);\n}\nsetTimeout(() => {\n    monitor();\n}, 3000);         \n                               ",
    'start_baidu_live': "\nfunction monitor() {\n\tconsole.log(\"load js\")\n    let videos = document.querySelectorAll('video');\n    if (!videos.length) {\n        setTimeout(monitor, 3000);\n        return;\n    }\n    for (const video of videos) {\n        video.pause();\n    }\n    \n    const chat_list = document.querySelector('.chat-items');\n    if (!chat_list) {\n        setTimeout(monitor, 3000);\n        return\n    }\n    \n    window.scrollTo(document.body.scrollWidth, 0);\n    \n    const observer = new MutationObserver((mutationsList, observer) => {\n        for (const mutation of mutationsList) {\n            if (mutation.addedNodes.length > 0) {\n                const node = mutation.addedNodes[0];\n                if (!node) {\n                    continue\n                }\n\n                const html = node\n                if (!html) {\n                    continue\n                }\n                console.log(html);\n\n                let name = html.querySelector('.user-name').innerText\n                if (name.endsWith(\" :\")) {\n                    name = name.slice(0, -2);\n                }\n\n                const content = html.querySelector('.danmaku-item-right').innerText\n                console.log(name, content)\n                window.pywebview.api.live_push_event({\n                    type: \"chat\",\n                    name: encodeURIComponent(name),\n                    content: encodeURIComponent(content)\n                })\n            }\n        }\n    });\n    const chatConfig = { attributes: false, childList: true, subtree: true, characterData: false };\n    observer.observe(chat_list, chatConfig);\n    \n    const bottom_list = document.querySelector('#brush-prompt');\n    console.log(bottom_list);\n    const bottomObserver = new MutationObserver((mutationsList, observer) => {\n        for (const mutation of mutationsList) {\n            if (mutation.addedNodes.length > 0) {\n                const node = mutation.addedNodes[0].firstChild;\n                if (!node) {\n                    continue\n                }\n\n                const html = node;\n                if (!html) {\n                    continue\n                }\n                console.log(html);\n\n                let name = html.querySelector('.interact-name').innerText;\n                if (name.endsWith(\"：\")) {\n                    name = name.slice(0, -1);\n                }\n                console.log(name)\n\n                const content = html.parentNode.querySelector('.flex-no-shrink').innerText;\n                console.log(content)\n                window.pywebview.api.live_push_event({\n                    type: \"bottom\",\n                    name: encodeURIComponent(name),\n                    content: encodeURIComponent(content)\n                })\n            }\n        }\n    });\n    const bottomConfig = { attributes: false, childList: true, subtree: true, characterData: false };\n    bottomObserver.observe(bottom_list, bottomConfig);\n}\nsetTimeout(() => {\n\tmonitor();\n}, 3000);        \n                               ",
    'start_bilibili_live': "\nfunction monitor() {\n    console.log(\"load js\")\n\n    const videoDiv = document.getElementById('__live_video__')\n    if (videoDiv) {\n        videoDiv.style.display = 'none'\n    } else {\n        setTimeout(monitor, 3000);\n        return\n    }\n\n    let videos = document.querySelectorAll('video');\n    for (const video of videos) {\n        video.muted = true;\n        video.pause()\n    }\n\n    let chat_list;\n    try {\n        chat_list = document.getElementById('liveComment');\n    } catch(err) {\n        console.log(err)\n        setTimeout(monitor, 3000);\n        return\n    }\n\n    const observer = new MutationObserver((mutationsList, observer) => {\n        for (const mutation of mutationsList) {\n            if (mutation.addedNodes.length > 0) {\n                const node = mutation.addedNodes[0];\n                if (!node) {\n                    continue\n                }\n\n                const html = node.innerHTML;\n                if (!html) {\n                    continue\n                }\n                console.log(html);\n\n                let spans = node.getElementsByTagName('span');\n                let name = spans[0].innerText;\n                if (name.endsWith(\": \")) {\n                    name = name.slice(0, -2);\n                }\n\n                const content = spans[1].innerText;\n                window.pywebview.api.live_push_event({\n                    type: \"chat\",\n                    name: encodeURIComponent(name),\n                    content: encodeURIComponent(content)\n                })\n            }\n        }\n    });\n    const config = { attributes: false, childList: true, subtree: true, characterData: false };\n    observer.observe(chat_list, config);\n}\nsetTimeout(() => {\n    monitor();\n}, 3000);\n",
    'start_alibaba_live': "\nfunction monitor() {\n\tconsole.log(\"load js\")\n    let videos = document.querySelectorAll('video');\n    if (!videos.length) {\n        setTimeout(monitor, 3000);\n        return;\n    }\n    for (const video of videos) {\n        video.pause();\n    }\n\n    const chatframe = document.getElementById('chatframe');\n    if (!chatframe) {\n        setTimeout(monitor, 3000);\n        return;\n    }\n    \n    try {\n        chatframe.onload = function() {\n            const iframeDoc = chatframe.contentDocument || chatframe.contentWindow.document;\n            const chat_list = iframeDoc.querySelector('#items');\n            if (!chat_list) {\n                setTimeout(monitor, 3000);\n                return;\n            }\n            \n            const observer = new MutationObserver((mutationsList, observer) => {\n                for (const mutation of mutationsList) {\n                    if (mutation.addedNodes.length > 0) {\n                        const node = mutation.addedNodes[0];\n                        if (!node) continue;\n                        if (!node.querySelector) continue;\n                        \n                        const name = node.querySelector('#author-name').innerText;\n                        const content = node.querySelector('#message').innerText;\n                        window.pywebview.api.live_push_event({\n                            type: \"chat\",\n                            name: encodeURIComponent(name),\n                            content: encodeURIComponent(content)\n                        });\n                    }\n                }\n            });\n            \n            const chatConfig = {\n                attributes: false,\n                childList: true,\n                subtree: true,\n                characterData: false\n            };\n            observer.observe(chat_list, chatConfig);\n        };\n        \n        if (chatframe.contentDocument.readyState === 'complete') {\n            chatframe.onload();\n        }\n    } catch (e) {\n        setTimeout(monitor, 3000);\n    }\n}\nsetTimeout(() => {\n\tmonitor();\n}, 3000);        \n                               ",
}

# matchUrlEvalJs 的 host 前缀(反编译 tuple 常量;probeI 验证 startswith 语义)
_HOST = {
    'start_dy_live': 'https://live.douyin.com/',
    'start_tb_live': 'https://tbzb.taobao.com/',
    'start_pdd_live': 'https://mobile.yangkeduo.com/pjlkvgcf.html',
    'start_mt_live': 'https://g.meituan.com/',
    'start_xhs_live': 'https://redlive.xiaohongshu.com/live_center_control',
    'start_jd_live': 'https://jlive.jd.com/my/list',
}

# 每条 JS 的串表槽位与置信度登记
_JS_SLOT = {
    'start_dy_live': '18002c968',  # 已验证:init PyTuple_Pack(2, 'https://live.douyin.com/', <js>) 邻域槽
    'start_tb_live': '18002c828',  # 已验证:同上(https://tbzb.taobao.com/)
    'start_pdd_live': '18002c848',  # 已验证:同上(https://mobile.yangkeduo.com/pjlkvgcf.html)
    'start_mt_live': '18002c7e0',  # 已验证:同上(https://g.meituan.com/)
    'start_xhs_live': '18002c810',  # 已验证:同上(https://redlive.xiaohongshu.com/live_center_control)
    'start_jd_live': '18002cb68',  # 已验证:同上(https://jlive.jd.com/my/list)
    'start_ks_live': '18002c958',  # E2 假说:剩余 9 槽按 DOM 结构指派,未定谳
    'start_ks2_live': '18002c788',  # E2 假说:同上
    'start_sph_live': '18002cb60',  # E2 假说:同上
    'start_tk_live': '18002cbf8',  # E2 假说:同上
    'start_youtube_live': '18002cba0',  # E2 假说:同上
    'start_shopee_live': '18002cad0',  # E2 假说:同上
    'start_yundong_live': '18002ca48',  # E2 假说:同上
    'start_baidu_live': '18002c7c0',  # E2 假说:同上
    'start_bilibili_live': '18002cbe8',  # E2 假说:同上
    'start_alibaba_live': '18002cb60',  # E2 假说:同上(与 sph 同槽待定谳)
}
