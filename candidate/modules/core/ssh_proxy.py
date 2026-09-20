# -*- coding: utf-8 -*-
"""modules.core.ssh_proxy —— T4R 语义重建(R036)。

源 pyd:modules/core/ssh_proxy.cp310-win_amd64.pyd
        sha256 dc319b65dac09e32fc12a33d6017127cc167756d571f019cb03a3e630f8b4716

二进制出处(Ghidra 12.1.3 反编译,证据 runtime/ghidra/):
  * parse_ssh     @0x180001010  code: argcount=1 nlocals=11 firstlineno=8
  * run_command   @0x180002be0  code: argcount=5 nlocals=10 firstlineno=57
  * start_tunnel  @0x180004f60  code: argcount=5 nlocals=8  firstlineno=96
  * run_proxy     @0x1800064e0  code: argcount=3 nlocals=6  firstlineno=137
  * 模块 exec @0x1800076c0;InitCachedConstants @0x180006e20(0x180012120 =
    2 元常量 —— 一条含原厂跳板主机的 SSH 命令行 + 一个 32 字符口令,即
    __main__ 块 run_proxy(*常量) 的实参)。
    [COM-D002] 这两个**凭据/原厂基础设施字面量的值已删除**(与 cloud_ssh.py
    同形副本同步清除):__main__ 改为从凭据保险箱/配置读取,缺失即显式报错,
    绝不回落字面量。行为要点(run_proxy/parse_ssh 等)不变。

E1 探针(两模块同形,证据证据目录 evidence/modules/core__cloud_ssh/runtime/):
  probeA_env_ssh.py / probeA2_ssh_edges.py / probeB_ssh_stub.py(源码级同一份,
  仅模块名与 __file__ 不同:probeA 对比两模块 parse_ssh 全矩阵逐条一致)。

行为要点(全部由探针实测):
  * parse_ssh:shlex.split → 首元素为 'ssh' 时丢弃 → argparse(add_help=False,
    -p/--port type=int default=22, destination nargs='*') 的 parse_known_args,
    '@' 扫描对象是 `list(args.destination) + unknown`(probeA2[0] 判别:
    'ssh -p 22 -oProxyCommand=a@b host' → ('b', 22, '-oProxyCommand=a')),
    取首个含 '@' 项并以 split('@', 1) 切分;异常打印后回落到 (None, 22, None)。
  * run_command:connect 超时 15;成功打印两行;exec_command 的三元组解包后
    未读取(probeB 的 fake 记录零 read 调用);异常分三层;finally 关连接。
  * start_tunnel:SSHTunnelForwarder((host, port), remote_bind_address=
    ('127.0.0.1', local_port), local_bind_address=('0.0.0.0', local_port),
    set_keepalive=10.0),start 后 1s 保活循环,KeyboardInterrupt 打印停隧道;
    start 失败路径不调用 stop(probeB 记录)。
  * run_proxy = parse_ssh → run_command(nohup 启动远端服务) → start_tunnel。
"""
import argparse
import os
import shlex
import time

import paramiko
from sshtunnel import SSHTunnelForwarder

# [COM-D002] 凭据来源表:凭据保险箱/环境变量/配置文件。候选**绝不**内置任何
# 口令、密钥或原厂跳板主机字面量,也不接受"内置默认值"。
_CREDENTIAL_ENV = {
    'ssh_command': 'SZR_CLOUD_SSH_COMMAND',
    'ssh_password': 'SZR_CLOUD_SSH_PASSWORD',
}


def _require_credential(option):
    """按 option 取凭据:环境变量 → config.ini [Credentials] → 缺失即硬失败。

    [COM-D002] 返回**配置提供的值**;缺失时抛显式 RuntimeError,绝不回落到任何
    内置字面量。值本身不打印、不落盘、不进报告(与 tools/secret_scan.py 同一纪律);
    正式的凭据存储接口由 platform_adapters/common/credential_vault.py 提供,
    本模块只消费其注入的环境/配置,不直接依赖适配层。
    """
    env_name = _CREDENTIAL_ENV[option]
    value = os.environ.get(env_name, '')
    if not value:
        parser = __import__('configparser').ConfigParser()
        parser.read(os.path.join(os.getcwd(), 'config.ini'), encoding='utf-8')
        value = parser.get('Credentials', option, fallback='')
    if not value:
        raise RuntimeError(
            '缺少凭据 %s:请通过凭据保险箱/配置提供(环境变量 %s 或 config.ini '
            '[Credentials] %s);客户端不内置任何凭据。' % (option, env_name, option))
    return value


def parse_ssh(ssh_cmd):
    """
    简化的argparse解析版本，专注于解析-p和user@host

    Args:
        ssh_cmd (str): SSH命令字符串

    Returns:
        tuple: (ssh_host, ssh_port, ssh_user)
    """
    ssh_port = 22
    ssh_user = None
    ssh_host = None
    try:
        parts = shlex.split(ssh_cmd)
        all_args = parts[1:] if parts and parts[0] == "ssh" else parts
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("-p", "--port", type=int, default=22)
        parser.add_argument("destination", nargs="*")
        args, unknown = parser.parse_known_args(all_args)
        ssh_port = args.port
        for arg in list(args.destination) + unknown:
            if "@" in arg:
                ssh_user, ssh_host = arg.split("@", 1)
                break
    except Exception as e:
        print("解析SSH命令时出错: " + str(e))
    return ssh_host, ssh_port, ssh_user


def run_command(ssh_host: str, ssh_port: int, ssh_user: str, ssh_password: str, command: str):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=ssh_host, port=ssh_port, username=ssh_user,
                       password=ssh_password, timeout=15)
        print("✅ 连接成功。")
        print(f"正在执行命令: **{command}**")
        stdin, stdout, stderr = client.exec_command(command)
        print("✅ 命令执行完成。")
    except paramiko.AuthenticationException:
        print("❌ 认证失败，请检查用户名和密码。")
    except paramiko.SSHException as e:
        print("❌ SSH 连接或会话错误: " + str(e))
    except Exception as e:
        print("❌ 发生其他错误: " + str(e))
    finally:
        client.close()
        print("\n连接已关闭。")


def start_tunnel(ssh_host: str, ssh_port: int, ssh_user: str, ssh_password: str, local_port: int = 9887):
    print(f"正在连接到 {ssh_host}:{ssh_port} ...")
    try:
        remote_bind_address = ("127.0.0.1", local_port)
        server = SSHTunnelForwarder(
            (ssh_host, ssh_port),
            ssh_username=ssh_user,
            ssh_password=ssh_password,
            remote_bind_address=remote_bind_address,
            local_bind_address=("0.0.0.0", local_port),
            set_keepalive=10.0,
        )
        server.start()
        print("✅ 隧道建立成功！")
        print(f"映射地址: 127.0.0.1:{local_port} -> 远程:{remote_bind_address}")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n正在停止隧道...")
        finally:
            server.stop()
    except Exception as e:
        print(f"❌ 发生错误: {e}")


def run_proxy(ssh_cmd: str, ssh_password: str, port: int = 9887):
    ssh_host, ssh_port, ssh_user = parse_ssh(ssh_cmd)
    # [I-C E2E 定谳 ic2-fix-6] I006 整机 E2E 实测(round3 观测
    # i006_ssh_tunnel.ssh_run.candidate.json 同槽对照 oracle 记录桩):
    # oracle pyd 以**全关键字**实参调用 run_command/start_tunnel,候选原为位置实参
    # —— 绑定值等价但调用约定不符,按 oracle 形态改为关键字调用。
    run_command(ssh_host=ssh_host, ssh_port=ssh_port, ssh_user=ssh_user,
                ssh_password=ssh_password,
                command="nohup /root/start.sh > /root/start.log 2>&1 &")
    start_tunnel(ssh_host=ssh_host, ssh_port=ssh_port, ssh_user=ssh_user,
                 ssh_password=ssh_password, local_port=port)


if __name__ == "__main__":
    # [COM-D002 语义替换] 原为 run_proxy(<原厂跳板 SSH 命令行字面量>,
    # <32 字符口令字面量>)——两个值均已删除。现从凭据保险箱/配置读取:
    # 缺失即 _require_credential 抛显式 RuntimeError,绝不回落到任何内置值。
    run_proxy(_require_credential('ssh_command'),
              _require_credential('ssh_password'))
