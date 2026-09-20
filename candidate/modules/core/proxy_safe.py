# -*- coding: utf-8 -*-
"""proxy_safe —— R008 重建(源 modules/core/proxy_safe.cp310-win_amd64.pyd,T4R)。

证据标签:[pyd 0x…] Ghidra 反编译(evidence/modules/core__proxy_safe/static/decomp/out/)、
[串表] pyd 字符串表、[E1 eN] VM 内 oracle 探针(runtime/probe_proxy_eN*.py + _out.json)。

E1 结论(全部实测):
* 模块命名空间 21 名:PROXY_IP='127.0.0.1'、PROXY_PORT=9886、Path(pathlib)、
  ProxySetting(winproxy 包同名类)、ps(= ProxySetting() 实例)、http(mitmproxy.http)、
  json/os/psutil/requests/subprocess/time + 9 个函数;
* is_port_in_use(p):psutil.net_connections() 遍历 laddr.port,命中 True,否则 False
  (str 端口不报错、返回 False)[E1 e1];
* is_cert_installed_windows(name):`certutil -store root <name>`(check=True/text=True),
  返回 `name in result.stdout`;异常路径打印 '❌ 检查证书失败: …' 返回 False[e1];
* install_cert_windows(path):`certutil -addstore root <path>`;异常路径打印
  `'❌ 证书安装失败: ' + str(exc)` 并返回 None[e1];
* callapi(data):requests.post('http://127.0.0.1:3060/local_danmu',
  data=json.dumps(data), headers={'Content-Type': 'application/json'})[e1/e6];
* handle_sph(flow):`'channels.weixin.qq.com/micro/live/cgi-bin/mmfinderassistant-bin/
  live/msg' in flow.request.pretty_url` 时解析 flow.response.content(utf-8)→
  data.data.msgList,逐条 print('✅ 弹幕: <nickname>: <content>') 并
  callapi({'type':'chat','name':…,'content':…,'clientMsgId':…});解析异常静默;
  未命中不做任何事[e6];
* handle_zbbl(flow):pretty_url.startswith(<拦截前缀配置>) 时
  `flow.response = http.Response.make(200, 预设JSON,
  {'Content-Type':'application/json'})` 并 print('✅ 拦截并返回预设响应: ' + host);
  未命中调 `flow.resume()`[e6]。原实现的前缀是硬编码原厂 SDK 更新检查 URL
  (COM-D002 已删除该字面量,改由 SZR_ZBBL_INTERCEPT_PREFIX 提供,默认空 ⇒ 不拦截);
* set_proxy():ps.enable=True;ps.server='127.0.0.1:9886';
  ps.override=['<-loopback>','localhost','127.0.0.1','::1'];ps.registry_write();
  print('✅ 已设置系统代理: 127.0.0.1:9886')[e1:注册表 ProxyEnable 0→1];
* unset_proxy():ps.enable=False;ps.registry_write();print('✅ 已清除系统代理')[e1];
* check_cert():`while True: sleep(1) …` 常驻循环(签名 ());以
  check_cert(1) 触发 `TypeError: check_cert() takes no arguments (1 given)`[e1]。
"""
import json
import os
import subprocess
import time
from pathlib import Path

import psutil
import requests
from mitmproxy import http
from winproxy import ProxySetting

# [COM-D002 语义替换] 原实现把该拦截面写死成原厂代理主机上的 SDK 更新检查 URL。
# 字面量已删除:拦截前缀改为显式配置(环境变量 SZR_ZBBL_INTERCEPT_PREFIX),
# **默认空串 ⇒ 不拦截任何主机**(fail-closed),候选不内置任何原厂/第三方主机。
# 注意:空串必须显式判空,因为 str.startswith('') 恒为 True。
_ZBBL_INTERCEPT_PREFIX = os.environ.get('SZR_ZBBL_INTERCEPT_PREFIX', '').strip()


def is_port_in_use(port):
    # [pyd 0x180001010] for conn in psutil.net_connections(): conn.laddr.port == port
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            return True
    return False


def is_cert_installed_windows(cert_name):
    # [pyd 0x180001540] certutil -store root <name>(PIPE/text)
    # E1 实测:证书不存在时 certutil 退出码 2148073489,但 oracle 仍**不抛异常**、
    # 直接 `in stdout` 得 False(故此处不传 check=True;反编译快照里的 check 参数
    # 与实测不符,以实测为准,登记在报告 §偏差)。
    try:
        result = subprocess.run(['certutil', '-store', 'root', cert_name],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        return cert_name in result.stdout
    except subprocess.CalledProcessError as exc:
        print('❌ 检查证书失败: ' + str(exc.stderr))
        return False


def install_cert_windows(cert_path):
    # [pyd 0x180002360] certutil -addstore root <path>;失败打印并返回 None
    try:
        subprocess.run(['certutil', '-addstore', 'root', cert_path],
                       check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        print('❌ 证书安装失败: ' + str(exc))


PROXY_IP = '127.0.0.1'          # [pyd 模块级 PyDict_SetItem PROXY_IP]
PROXY_PORT = 9886               # [pyd DAT_1800125d0 = PyLong_FromLong(0x269e)]
ps = ProxySetting()             # [pyd 模块级 ps = ProxySetting()]


def set_proxy():
    # [pyd 0x180002ea0;E1 e1 注册表前后对照]
    ps.enable = True
    ps.server = PROXY_IP + ':' + str(PROXY_PORT)
    ps.override = ['<-loopback>', 'localhost', '127.0.0.1', '::1']
    ps.registry_write()
    print(f'✅ 已设置系统代理: {PROXY_IP}:{PROXY_PORT}')


def unset_proxy():
    # [pyd 0x180003df0;E1 e1:ProxyEnable → 0]
    ps.enable = False
    ps.registry_write()
    print('✅ 已清除系统代理')


def callapi(data):
    # [pyd 0x1800041c0;URL 常量 0x1800121d0]
    requests.post('http://127.0.0.1:3060/local_danmu',
                  data=json.dumps(data),
                  headers={'Content-Type': 'application/json'})


def handle_sph(flow):
    """捕获 HTTPS 响应并解析响应数据 [pyd 0x180004700;docstring 原文]"""
    if 'channels.weixin.qq.com/micro/live/cgi-bin/mmfinderassistant-bin/live/msg' \
            in flow.request.pretty_url:
        try:
            data = json.loads(flow.response.content.decode('utf-8'))
            for msg in data.get('data', {}).get('msgList', []):
                print(f"✅ 弹幕: {msg.get('nickname')}: {msg.get('content')}")
                callapi({'type': 'chat', 'name': msg.get('nickname'),
                         'content': msg.get('content'),
                         'clientMsgId': msg.get('clientMsgId')})
        except Exception:                       # noqa: BLE001
            # [pyd 尾部异常路径] 弹幕链路任何解析/回调异常都不得中断 mitmproxy 线程;
            # 原始实现静默吞掉(E1 e6:坏 JSON/缺键 → 无输出无异常)。
            return


def handle_zbbl(flow):
    """mitmproxy 请求事件钩子，拦截目标URL并返回预设JSON响应 [pyd 0x180005d80;docstring 原文]

    [COM-D002] 拦截目标不再是硬编码的原厂 SDK 更新检查 URL,而是
    SZR_ZBBL_INTERCEPT_PREFIX 配置的前缀;未配置(空串)时不拦截任何请求,
    直接走 `flow.resume()`(fail-closed)。
    """
    prefix = _ZBBL_INTERCEPT_PREFIX
    if prefix and flow.request.pretty_url.startswith(prefix):
        flow.response = http.Response.make(
            200,
            '{"err_code":0,"err_message":"success","data":{"needUpdate":false,"isFromRedis":true}}',
            {'Content-Type': 'application/json'})
        print('✅ 拦截并返回预设响应: ' + flow.request.host)
    else:
        flow.resume()


def check_cert(*args):
    # [pyd 0x180006590] while True: sleep(1) → 端口在用时装 mitmproxy 根证书;
    # Cython cyfunction 的签名门控消息为 "check_cert() takes no arguments (1 given)",
    # 纯 Python 默认消息不同,故显式复现(E1 e1 / 用例 proxy_safe__check_cert)。
    if args:
        raise TypeError('check_cert() takes no arguments (%d given)' % len(args))
    while True:
        time.sleep(1)
        if is_port_in_use(PROXY_PORT):
            cert_path = os.path.join(Path.home(), '.mitmproxy/mitmproxy-ca-cert.cer')
            if os.path.exists(cert_path):
                if not is_cert_installed_windows('mitmproxy'):
                    install_cert_windows(cert_path)
