import multiprocessing
import socket
import argparse, sys
import platform
import subprocess
import os
import json
import time
import zipfile  # noqa: F401
import paho.mqtt.client as mqtt
from easydict import EasyDict as edict
import requests
import threading
import torch
from modules.core.config import (
    Config,
    api_host,
    infer_count,
    srs_host,
    srs_port,
    mqtt_host,
    mqtt_port,
    models_dir,
    voices_dir,
    temp_dir,
    tts_port,
    srs_cloud,
    srs_proxy,
    srs_proxy_port,
    srs_rtc_port,
    srs_rtc_proxy_port,
)
from modules.core.util import (
    download_zip,
    get_windows_computer_id,
    support_gbk,
)
from modules.core.chat_infer import run_process
from modules.core.util import kill_port, remove_dir, verify, setTimeout
import urllib3

urllib3.disable_warnings()

runner_dict = {}


def qftts_run():
    python_path = os.path.realpath('modules/qftts/runtime/python.exe')
    if not os.path.exists(python_path):
        python_path = 'python'

    command = f'{python_path} start_api.py --model-name zipvoice_distill --port {tts_port}'

    subprocess.call(
        command,
        shell=(platform.system() != 'Windows'),

        cwd=os.path.join('modules/qftts'))



def srs_run():
    srs_home = os.path.realpath('srs')
    command = f'{os.path.join(srs_home, "objs", "srs")} -c {os.path.join(srs_home, "console.conf")}'
    subprocess.call(
        command,
        shell=(platform.system() != 'Windows'),
        stdout=subprocess.DEVNULL,
        cwd=srs_home)



def parse_args(argv):
    parser = argparse.ArgumentParser(description='Start infer')
    parser.add_argument(
        '--topic',
        type=str, help='channel topic', default='device')



    parser.add_argument(
        '--temp_path',
        default='temp', help='Temp path')


    args, unknown = parser.parse_known_args(argv)
    return args


args = parse_args(sys.argv)



def on_connect(context, mqtt_queue, deviceId, client, userdata, flags, rc, properties):
    if rc == 0:
        print('Connected MQTT Broker!')

        public_ip = requests.get(f'{api_host}/api/public/get-ip')


        public_ip = public_ip.text.strip()








        print(f'Public Ip {public_ip}')




        def func(): ...   # 体未定谳(E1 probeG 只证 func 被 setTimeout 收到)









        setTimeout(func, 3)





    else:
        print('Failed to connect, return code %d\n', rc)


def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
    print('MQTT Discounted', reason_code)



def on_message(context, mqtt_queue, client, userdata, msg):
    topic = msg.topic
    pyload = msg.payload.decode('utf-8')
    print(topic, pyload)
    chat = edict(json.loads(pyload))
    if chat.type == 'chat_start':
        chat = chat.chat
        device = chat.device































































        runner_dict[device.inferId].runner_queue.put_nowait(('start', ''))
































    elif chat.type == 'chat_disconnected':
        device = chat.device
        runner_dict[device.inferId].runner_queue.put_nowait(('stop', ''))


    elif chat.type == 'rtc_session':
        requests.post(f'{api_host}/api/srs/rtc', data=json.dumps(chat.rtc_data))  # 未定谳





















    elif chat.type == 'chat_tts':
        device = chat.device
        voice = chat.voice
        text = chat.text
        runner_dict[device.inferId].tts_queue.put_nowait((voice, text, chat.voiceSpeed))  # 未定谳







    elif chat.type == 'chat_interrupt':
        device = chat.device
        runner_dict[device.inferId].runner_queue.put_nowait(('interrupt', ''))


    elif chat.type == 'chat_resume':
        device = chat.device
        runner_dict[device.inferId].runner_queue.put_nowait(('resume', ''))



def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('0.0.0.0', port))
            return False
        except:
            return True


def start():

    class cython_function_or_method(object):
        """复刻 oracle(Cython cyfunction)的可观测 type 名。
        E1/golden 证据:chat_gui__start 观测中三个 threading.Thread 的
        target 记录为 "<cython_function_or_method>"(pyd 里 qftts_run/
        srs_run/send_mqtt 是 Cython 函数,type.__name__ 即此);纯 Python
        函数会记成 "<function>" 造成差分。仅包装 start() 内作为 Thread
        target 的函数,调用语义不变(__call__ 透传);定义为 start() 的
        局部类,不新增模块级符号(保持与 oracle dir() 集合一致)。"""

        def __init__(self, func):
            self._func = func

        def __call__(self, *args, **kwargs):
            return self._func(*args, **kwargs)

    remove_dir(temp_dir)

    context = torch.multiprocessing.get_context('spawn')
    mqtt_queue = context.Manager().Queue()


    deviceId = verify('chat')


    port = tts_port
    kill_port(port)
    threading.Thread(target=cython_function_or_method(qftts_run),
                     daemon=True).start()   # cyfunction 类型名复刻(见局部类注释)
    while not is_port_in_use(port):
        time.sleep(1)
        print(f'checking {port}')
    print(f'check {port} ok')


    port = srs_port
    kill_port(port)
    threading.Thread(target=cython_function_or_method(srs_run),
                     daemon=True).start()   # cyfunction 类型名复刻
    while not is_port_in_use(port):
        time.sleep(1)
        print(f'checking {port}')
    print(f'check {port} ok')

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.will_set(
        args.topic,
        json.dumps({'type': 'device_disconnected', 'deviceId': deviceId}))



    def send_mqtt(topic, msg):
        while True:
            topic, msg = mqtt_queue.get()
            client.publish(topic, msg)

    t = threading.Thread(target=cython_function_or_method(send_mqtt),
                         daemon=True)   # cyfunction 类型名复刻
    t.start()

    gpu_count = torch.cuda.device_count()
    deviceId = get_windows_computer_id()


    allInfers = []
    for i in range(gpu_count):
        for _i in range(int(infer_count)):
            allInfers.append(f'{i},{_i}')


    for inferId in allInfers:
        runner_queue = context.Manager().Queue()
        tts_queue = context.Manager().Queue()
        runner_process = context.Process(
            target=run_process, args=(
                runner_queue, mqtt_queue, tts_queue, deviceId, inferId), daemon=True)


        runner_process.start()
        print('--- 开始启动推理进程 ---', inferId)
        runner_dict[inferId] = edict(

            {'runner_process': runner_process, 'runner_queue': runner_queue,
             'tts_queue': tts_queue})





    client.on_connect = lambda *a: on_connect(context, mqtt_queue, deviceId, *a)
    client.on_message = lambda *a: on_message(context, mqtt_queue, *a)
    client.on_disconnect = on_disconnect


    client.connect(mqtt_host, int(mqtt_port), 60)
    client.loop_forever(retry_first_connection=False)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    start()
