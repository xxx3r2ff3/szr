# -*- coding: utf-8 -*-
"""app_infer —— 恢复实现(重建自 modules/core/app_infer.cp310-win_amd64.pyd)。

R020 保真修补批次(2026-09-10,ns_fidelity2):按 E1 补齐 86 个模块级符号、对齐
导入次序(缺 config.ini 时 stderr 不得出现 pydub 警告)、按 E1 重建 9 个本地类。
证据与偏差清单见文件末 R020 证据块。
"""

import base64
import bisect
import copy
import ctypes
from datetime import datetime, timedelta
import gc, json
import math
import pathlib
import platform
from queue import Empty, PriorityQueue, Queue
import itertools
import re
import shutil
import sqlite3
import string
import subprocess
import sys
import threading
import time
import uuid
import wave
import zipfile

import librosa, pyaudio, pyvirtualcam, bottle
import numpy as np, requests
from colorama import Fore, Style
import cv2, os, socket, argparse, random
from tqdm import tqdm
import soundfile as sf, torch
from torch.nn import functional as F

_cuda_bin = os.path.join(os.path.dirname(sys.executable), "cuda", "v11.8", "bin")
import onnxruntime  # noqa: E402  (E1:torch/F/_cuda_bin 之后,与 oracle 同序)

# ---- 导入次序(E1 probe_r020_import,oracle A/B)----
# 缺 config.ini 时 stderr 只有 traceback(app_infer.py:47 → app_util.py:14 →
# config.py:81/55 → Exception: config.ini not found),**无 pydub 警告**;
# 故 app_util 先于本模块自己的 pydub 导入 —— 下行即 oracle 源行 47。
from modules.core.app_util import (
    cut_text,
    dybuyinRequest,
    download_file,
    filter_banned_words,
    getSovitsClone,
    getVsaClone,
    indexttsClone,
    indexttsInit,
    indexttsUpload,
    is_cloud,
    qfttsClone,
    is_port_in_use,
    only_punc,
    print_red,
    print_yellow,
    sphRequest,
    text2edgevoice,
    request,
    dyeosRequest,
    vcClone,
    voxcpmClone,
    luxttsClone,
    omnivoiceClone,
)


def _get_local_ip():
    """本机出口 IP(E1:srs_host / 横幅 'local ip' 均取此值;UDP connect 不发包)。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


# ---- config 派生全局(E1 值钉死;oracle 由 modules.core.config 再导出)----
# [COM-D002 语义替换] 原实现:
#   api_host = cwd config.ini [Server] host(fallback = 原厂官方主机字面量)
# 原厂主机字面量与"官方回退值"已删除。**本行是候选树里 api_host 的唯一配置源**:
# 其它模块一律从本模块或 modules.core.config 再导出,不得自建第二份。
# 缺省空串 = fail-closed:未显式配置自建端点时,派生的相对 URL 不满足 scheme,
# requests/httpx 立即抛错,绝不静默回落到任何原厂/第三方主机。
# 目录族 = cwd/human_data/*;导入期不建目录(E1:导入无 files.created)。
_config_parser = __import__("configparser").ConfigParser()
_config_parser.read(os.path.join(os.getcwd(), "config.ini"), encoding="utf-8")
api_host = _config_parser.get("Server", "host", fallback="")
home_dir = os.path.abspath("human_data")
models_dir = os.path.join(home_dir, "models")
cache_dir = os.path.join(home_dir, "cache")
temp_dir = os.path.join(home_dir, "temp")
voices_dir = os.path.join(home_dir, "voices")
noise_dir = os.path.join(home_dir, "noises")
logs_dir = os.path.join(home_dir, "logs")
rags_dir = os.path.join(home_dir, "rags")
trt = True                     # E1:bool,横幅 'gpu trt True' 同源
srs_host = _get_local_ip()     # E1:本机出口 IP(与横幅 local ip 同值)
srs_port = 3985
rtmp_port = 3935
thumb_port = 3061
infer_width = 1080
gpt_port = 9881
vsa_port = 9882
tts_port = 9885
vc_port = 9884
voxcpm_port = 9888
luxtts_port = 9889
omnivoice_port = 9890
index_port = 9887
camera = "OBS Virtual Camera"
camera_rotate90 = False
bitrate = "5000k"
local_wiki = True
srs_cloud = False
ffmpeg_path = "ffmpeg"
local_tracker = True
tracker_min_length = 10
tracker_minutes = 30
audio_changer = False
ai_label = True
ai_text = "AI生成"
robot_name = '\n何为美人|乱了分寸|难喻|桃奈叶子|暮雨|深拥意中人|海底星河|满眼笑意|旧城不忘少年心|寄余生|尽情摇摆|淡似春风|夏夜的琉璃|十里温柔|玖辞|天然无|不痛|戈祁|薄荷撞可乐|晚晚星河|野山茶|一笑倾城|寄云|反派角色|假装|茫茫人生好像荒野|春日小憩|捧一束野菊|赋恶|迟暮|借一盏月色|真是个可爱|月幕几许|倚梅看雪|远山只梦里|热烈而醇|甜蜜危机|夏柒|不要爱情了|秋鹤渡|清浅旧时光|敬余生|南风轻拂|躲在群星中|那年风月|白云揉碎|如初岁月|沧叶|月言|放纵随遇而安|请夏天吃饭|散不尽过往|杯月亮半价|你所见即我|绿绮|等风|思念成疾|饕餮少女|选择离开就别回头|习惯性依赖|默浅忆|小梨窝很甜|青春喂了作业|素小言|好女人就是我|萌主系我|雪落成殇|很酷只撩你|我会好好的|少女萌主|我宣你你造吗|乖乖猪|果味小可爱|酒味少女|奶音能量|百般纵容|玖辞love|爱如指间沙|南风轻拂|夏末未央|吴织亚切|苗琴声声9|偷吻月亮|章鱼小肉丸|不解风情的老妖怪|浅墨洛殇|当星光没有光|萌音草莓|百变小熊喵|cookie|你的小朋友|柠夏初开|小可爱在此|别撩我裙子|嘴角的樱桃汁|巴黎雨下|无心话|风软一江水|娇俏可人|放纵随遇而安|甜蜜的人儿|萌卷软糕|凹凸曼也有身材|糖果|劳资虐遍|孤枕人|心脏多矫情|青橘栀耳|桀骜如初|夜久泪长|一颗糖|小豆芽|情书寄予山|不致太寡|生命是一场旅途|妞儿|困哭惹|孤绝如初见|淡若轻风|柒夜笙歌凉|余悸|水煮活人|星云摆|初恋网友|三寸旧城七寸执念|洒脱的自己|曾经沧海|曲终人也散|尘事太揪人心|夏之樱|会飞的刺猬|我还爱你|心动代码|幼稚的可爱生活|流光夜雪|孤心匠|血之狂魔|冬日倦|唇香绕齿柔|离弦|听闻余生|菇凉你好|你是我哒|如梦|桃枝乱|可爱多是我|乖娃娃|一杯清酒当人生|何勾搭|覆了天下又如何|苹果你个爱泡泡|岁涟漪|她曾是我的月夜|星星魔法盒|邀月|樱丸小桃子|萌妹妹哟|爷有爷的范|小阿呆|星眠曲|陌生默笙|暖南倾绿|时光是个罪人|性格坦荡不讨喜|时间枯萎|宇宙风|岁月并非如歌|清风与鹿|几人难应|树红树绿|高傲的存活|雪落成殇|落絮染轻愁|楠汐|稚与初|南栀|简以时光|小镇与凉梦|黎兮|回眸一笑|万人我独殇|空无一友|时光划破青春|半夏柒微凉|久碍你|有你时刻在|选择悲伤|小岛西岸来信|一生孤注|夏末的晨曦|青春染指流年|风铃|美梦收藏家|盛席|眼角眉梢都是你|山抹微云|暖风撩人醉|心中曾经难舍弃|暮雪流年|角落遗忘的爱|消磨中过活|是你|甜蜜危机|说好了再见|不过如此|停不住的记忆|限定小汀|八度余温|闭眼听风|过期爱人|风撩少女心|栀璃鸢挽|一个人的旅途|一起来看看吧|落魄的思念|桃泽樱|柚花离海|宠你一世|末日狂欢|断情亡音|悲欢独自饮|枕下悲绪|醉当场|万人追不如一人疼|几分醉意|难过要藏起来|等无此人|淡淡离愁|欲言转身|微笑掩盖不住伤痛|找不到解药|他心无我|几分醉意|久瘾则伤|未来多久才来|与寂寞无关|旧街旧港|几分醉意|尘封已久心伤|无法掩饰情伤|原来无话可说|仰头让眼泪流回眼眶|故人叹|独遇不与|愿我能哭能笑能随性|与寂寞无关|人逝花落空|与寂寞无关|你能毁了我吗|苦痛不由衷|用谎言来维持爱情|云有泪|原来无话可说|沉默不语|悲伤怎么遮掩|原来无话可说|时间在走|人心在变|哭到世界为我痛|自我|悲天悯人|人逝花落空|冷冻的沉默|凉城空巷|笑着哭痛|拿什么去爱你我的爱人|槿城|漠然|归晚|做自己|朕好萌|浮石|一笑奈何|三千青丝|南凉北暖|归途风景|整个世界|落满南山|遇你得幸|为你孤守|一身坏脾气|偏爱你侧脸|凉凉的眉头|岁月斑驳|幸福的孩子|念汐|潮离|我只牵你手|如秋叶|猫性小仙女|花刺痛心魂|萌萌的刚好|这里没有喵|雨落思念起|me只是喜欢你|听见雪的冬天|待她满目柔光|忧伤染指青春|般的爱情|繁华落幕|让爱随风飘逝|走出爱情的伤|风渔人到岸|合约恋人|陪伴输给喜欢|二货世界欢乐多|多远都要在一起|每滴泪都是钻石|醉饮南巷清风酒|剩我一个人在回忆|一辈子只陪你一个|许你一片繁花似锦|鱼忘七秒|人忘七年|我只想变成透明的颜色|由感而发|千山暮雪|孤独的守候|年轻在于拼|清樽独醉|默浅忆|欲说还休|爷很痴情|节度使|无资格叫痛|顾北清歌寒|扬帆起航|徒手平江山|飘零作归宿|毕竟念旧|芳草碧连天|相约奈何桥|毛毛的老爸|拆了奈何桥|长得很随意|含泪说分手|油炸皮卡丘|不爱滚远点|毅力坚帝|拼搏作兴趣|南栀倾寒|快亲我一下|热情喂狂风|倔的梦想|自作自受|我的地盘|心里老鸭汤|觅青森|聆听挽歌空|星河的明灭|教我演戏|你的心|夏日浅笑|如卿挽花间|糊涂的人|作业作孽|没有未来|赴月观长安|浅笑|乍见之欢|陪我走下去|纸醉金迷|风萤月缓缓|本人已下线|等我变|过来亲亲吗|奋斗赎青春|鳄鱼的眼泪|哭红了眼|捂风挽笑|他用我名拒她人|北惩|沦陷你心海|你的名我的伤|吴织亚切|爱我要趁早|爱如潮水|浪得一身野气|二次元期望|倒影年华|久不愈|三寸日光|眉心痣|你只是我的梦|萌比范er|女人要有范儿|往事随风|沐雨晨尘|时光吹老了好少年|半城繁华半城伤|搞怪|苏慕凉|念的温柔|旧巷与猫|浪痞孤王|怪性笑人|敌对状态|鹿时纸葵|果味喵|稳场小可爱|予我空欢喜|流年中的浅夏|二次元期望|三寸旧城|百般纵容|痞子三分冷|软果儿|枫溪|哭过痛过何曾弃过|的等待|是我勇敢太久|小熊丢了|骑猪上|玖辞love|萌卷软糕|流年划过岁月|不要深交|似风似你|南风轻拂|离心控|画骨成沙|会挖宝的小仙女|永远有多远|沫夏|长不胖的猪|顾我心安|心烦|停不住的记忆|萌萌妈|的一批|青丝变白发|情深致病|上学致命|草莓爱菠萝|许君乘闲月|素衣白裳|妄想称帝|厮守一季斑驳|傲视之巅|成长不会等我|已觉春心动|雪|青瓷清茶倾城歌|囚你于无期|菇凉我谈情不说爱|与我诉尽平生欢|软耳猫|梦好|森很绿却致人迷途|栀子|我忘了我自己|滥情不动心|爱到深处难以自拔|嗜你如命|哪怕一路没有掌声|光年与夏的初遇|嘣|静静等回忆|无法无天|秋水微澜|少女梦|梦想天空分外蓝|过期爱人|此男值得拥有|夏末未央|清歌留欢|没有白天的夜晚|兔兔丢了|浪漫如月亮|葬爱|亡心忘|蓝颜|痞味见底|原味的甜心少女|假装无所谓|何妨知交|致借|欲念阻断信念|我的未来不是梦|平平淡淡的生活|凉生初雨|遥遥江上客|栀夏微凉|颈上鲜草莓|血色安徒生|世世誓相依|安之若素|何假装成熟|玩够了别回来|你是太阳暖我心|娇俏可人|姐的世界与你何干|苟且|单机|容颜未老心已老|痴念|难以抽离|等待是无言的情话|难再相拥|安于现状|纸篓里的情书|爱久就见人心|真情可贵|我不及她美|夏尽|怨不得人|橘凉|别让幸福等太久|别听猫说|别笑了眼泪都掉了|修夏|弥足珍贵|夜幕篱下浅笙歌|雨生烟|初呢|心随风飞|不再见|柠檬的心酸是因为西瓜的甜|他总笑|穿过马路|边城|未蓝星星|把我推进深海少年你真善良|一封情书|千秋|请你善良|笙箫|唱出几个|已惘然|锁凉夏|颜欢笑|梦与岛|十年饮杯酒|浮生若懵|向前走别回头没有什么好难|请把我拿走|眉故|少时旧话|顾城歌|柠萌|囍妹|只要还有心跳就有我逗你笑|青黛|染屿|听闻余生|软香温玉|大傻|稚于初|平气和|为你封不再战|沉迷你的体香|抱走|三岁不长大|懵|吃藕丑|叫我霸霸|笑鲜侣|小仙女|温柔一人行|哔了dog了|余生陪你走下去|为你会变乖|青涩迷人|给你生猴子|幼年无忧|失宠的猫|怪味胖次|泡面笑我自然卷|风里雨里只为你|高一你好|初三再见|熟如陌人|多啦a不喜欢小丸子|懵仙少女|你的笑就是我的动力|打小就帅|可爱机智等|本宫略萌|哪里都有你|简讯|浑身仙女味|我知道是我不好|老奶奶的怪味豆|萌|萌萌哒的小可爱|病娇萌友|总是饿|海带啊海带|为你挡风遮雨\n'  # E1(probe_r020_meta):oracle 值逐字(含首尾 \n)
robot_wiki = '\n天气：这个主播不清楚呢/主播不知道呢/这个主播不清楚，具体的您可以查询手机上的天气APP\n漂亮、好看、美、帅：谢谢你的夸奖哦/谢谢夸奖哦\n主播多大了：哈哈，这可是个秘密，不过你可以猜猜看/哈哈，你猜猜看/哈哈，你猜\n女朋友、男朋友、对象、结婚、恋爱：你猜\n主播平时喜欢做什么：我平时喜欢看书、听音乐，还有运动/主播喜欢躺平\n才艺：我会的才艺可多了，比如唱歌、跳舞、画画/主播唱歌可好听了呢\n主播你人气不行啊：慢慢来嘛，一定会越来越好的！\n直播、开播：主播每天都会开播，可以给主播点个关注/每天都开播的，左上角点点关注\n主播累了吗：谢谢大家的关心，我还好啦/确实，播的时间长了有点累\n主播吃饭了吗：还没吃呢/你是说什么时候的饭/还没呢，你吃了吗\n主播今天心情怎么样：今天心情很不错呢\n地址、位置、在哪、在哪里、店、门店：点击直播间右下角链接，就能看到详细的地址\n买、购买、下单、抢购、支付：点击右下角链接下单\n多少钱、价格、费用：具体的可以点击右下角链接查看。\n优惠、活动、福利、套餐、特惠：点击右下角链接可以查看直播间的福利\n已拍、拍了、拍、买了：感谢支持\n退货、退款、退：不想要可以随时退，找到订单直接申请退款就可以了\n'  # E1(probe_r020_meta):oracle 值逐字(含首尾 \n)


print("client version v6.18.0")
print("gpu trt", trt)
print("local ip", _get_local_ip())
print("allow accounts all")
print("allow host all")


def has_chinese(path):
    pattern = re.compile(r"[\u4e00-\u9fff\uFF08-\uFF09]")
    return bool(pattern.search(path))


def strtobool(val):
    return val.lower() in ('y', 'yes', 't', 'true', 'on', '1')


def edgetts_speed(original_speed):
    return int(157.75 * original_speed - 115.5)


# 请求默认超时:本地网络函数(下载族)使用;request/sphRequest/dybuyinRequest/
# dyeosRequest/download_file 已按 oracle 同一性改为 app_util 再导出(E1 probe2)。
_REQUEST_TIMEOUT = 180

# E1(task_r020C/probe3 录制):api_host POST 附带 apiKey 头;原值为 36 字符
# UUID,编译期常量,按保密纪律不入库(已脱敏;偏差登记见文末证据块 E-6)。
# 本模块现存网络函数(request/sphRequest/dybuyinRequest/dyeosRequest/
# download_file)已按 oracle 同一性改为 modules.core.app_util 再导出,
# 该常量在本模块已无引用,故整条移除(同时消掉一个多余符号)。


def download_zip(url):
    """E1(task_r020C probe1 dz:local/not_zip + probe3 dz:cursor/probe4,
    恢复基线 oracle):GET stream(**含 tqdm**,同 download_file)写入
    tempfile.TemporaryFile()(**不解压、不校验 zip**,任意字节都成功),
    游标停在 EOF,返回 (url, 临时文件对象)。
    偏差登记:oracle 名为 modules.core.util.download_zip(R009 未重建),
    候选本地实现,行为经用例差分等价。"""
    import tempfile as _tempfile
    import requests as _requests
    from tqdm import tqdm as _tqdm

    fileobj = _tempfile.TemporaryFile()
    response = _requests.get(url, stream=True, timeout=_REQUEST_TIMEOUT)
    total = int(response.headers.get('content-length', 0))
    with _tqdm(total=total, unit='iB', unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                bar.update(fileobj.write(chunk))
    return url, fileobj


def send_mqtt_msg(mqtt_queue, topic, type, message):
    """E1(task_r020C probe1,恢复基线 oracle):put_nowait 元组
    (topic, json.dumps({"type": type, "message": message}));mqtt_queue
    None → AttributeError 'NoneType' object has no attribute 'put_nowait'。"""
    mqtt_queue.put_nowait((topic, json.dumps({"type": type, "message": message})))


def send_live_error(mqtt_queue, topic, device, message):
    """E1(task_r020C probe1/probe2,恢复基线 oracle)入队两条:
    (topic, json.dumps({"type":"dialog","message":message})) 与
    ("live/"+device.deviceUid, json.dumps({"type":"live_disconnected",
    "device":device}))(device 以 dict 内容序列化)。device 无 deviceUid
    → AttributeError(第二条 put 前);device 不可 JSON 序列化 →
    TypeError 'Object of type <T> is not JSON serializable'。"""
    mqtt_queue.put_nowait((topic, json.dumps({"type": "dialog", "message": message})))
    mqtt_queue.put_nowait(("live/" + device.deviceUid,
                           json.dumps({"type": "live_disconnected", "device": device})))


def download_model(mqtt_queue, sys_queue, topic, model, device):
    """E1(task_r020C probe4-probe8,恢复基线 oracle;**注意形参顺序与
    v7.0.0 期契约不同**:device 在末位,且 device 需 uuid/localResource/
    infer_video 属性链):
    - device.localResource 非空:print('模特:<uuid> 开始下载') → loading
      消息(模特:<uuid> 正在下载,请稍后)→ GET(api_host +
      localResource,startswith('http') 除外)下载 zip → 解压到
      models_dir/<uuid>/ → print 下载完成/解压完成 + alert 消息;
    - device.infer_video 非空 str:print 开始下载视频 → download_file
      (api_host+infer_video,dist=models_dir/<uuid>/live.mp4,'下载文件'
      行由 download_file 打印);否则写 model.ini(E1 golden:24 字节
      '[Model]\\r\\nversion = local',sha b6fde117…)→ print 缺少视频 →
      send_live_error(末参为 str → AttributeError 'str' object has no
      attribute 'deviceUid',E1 复现 oracle 自身缺陷;其首条 dialog 入队
      即队列可见的第 3 项)。
    - "已存在跳过下载"不存在:E1 probe8 预置 models_dir/<model>/model.pth
      仍重新下载(模型名在本函数可观测路径中不参与判定)。
    完整成功路径(视频真下载、model.ini、返回值)未采,登记 E-unknown。"""
    if device.localResource:
        print('模特:%s 开始下载' % device.uuid)
        send_mqtt_msg(mqtt_queue, topic, 'loading',
                      '模特:%s 正在下载，请稍后' % device.uuid)
        url = device.localResource
        if not url.startswith('http'):
            url = api_host + url
        payload = download_zip(url)
        import zipfile as _zipfile

        with _zipfile.ZipFile(payload[1]) as archive:
            support_gbk(archive)
            archive.extractall(os.path.join(models_dir, device.uuid))
        print('模特:%s 下载完成' % device.uuid)
        send_mqtt_msg(mqtt_queue, topic, 'alert', '模特:%s 下载完成' % device.uuid)
        print('模特:%s 解压完成' % device.uuid)
    if device.infer_video:
        print('模特:%s 开始下载视频' % device.uuid)
        dist = os.path.join(models_dir, device.uuid, 'live.mp4')
        url = device.infer_video
        if not url.startswith('http'):
            url = api_host + url
        download_file(url, dist)
    else:
        _ini_dir = os.path.join(models_dir, device.uuid)
        if not os.path.isdir(_ini_dir):
            os.makedirs(_ini_dir, exist_ok=True)
        with open(os.path.join(_ini_dir, 'model.ini'), 'wb') as _ini:
            _ini.write(b'[Model]\r\nversion = local')
        print('模特:%s 缺少视频' % device.uuid)
        send_live_error(mqtt_queue, topic, topic,
                        '模特:%s 缺少视频' % device.uuid)


def download_voice(mqtt_queue, sys_queue, topic, voice, device):
    """E1(task_r020C probe4-probe8,恢复基线 oracle;device 末位,需
    uuid/localResource/refer_wav(/refer_text)属性链):
    - localResource 非空:print('声音:<uuid> 资源包开始下载') → loading
      (声音:<uuid>␣␣正在下载,请稍后,**双空格**)→ GET api_host+
      localResource zip → 解压 voices_dir/<uuid>/ → 资源包下载完成/解压
      完成 + alert(双空格);
    - refer_wav 非空:print 开始下载参考音频 → download_file
      (api_host+refer_wav,dist=voices_dir/<uuid>/refer.wav,'下载文件'
      行由 download_file 打印);refer_wav 空/缺失:访问 device.refer_text
      (refer_text 属性缺失 → AttributeError '…' object has no attribute
      'refer_text',E1),为空时 print 缺少参考音频/参考文本 +
      send_live_error(str → AttributeError 'str' object has no attribute
      'deviceUid')。"""
    if device.localResource:
        print('声音:%s 资源包开始下载' % device.uuid)
        send_mqtt_msg(mqtt_queue, topic, 'loading',
                      '声音:%s  正在下载，请稍后' % device.uuid)
        url = device.localResource
        if not url.startswith('http'):
            url = api_host + url
        payload = download_zip(url)
        import zipfile as _zipfile

        with _zipfile.ZipFile(payload[1]) as archive:
            support_gbk(archive)
            archive.extractall(os.path.join(voices_dir, device.uuid))
        print('声音:%s 资源包下载完成' % device.uuid)
        send_mqtt_msg(mqtt_queue, topic, 'alert', '声音:%s  下载完成' % device.uuid)
        print('声音:%s 资源包解压完成' % device.uuid)
    if device.refer_wav:
        print('声音:%s 开始下载参考音频' % device.uuid)
        dist = os.path.join(voices_dir, device.uuid, 'refer.wav')
        url = device.refer_wav
        if not url.startswith('http'):
            url = api_host + url
        download_file(url, dist)
    else:
        if not device.refer_text:
            print('声音:%s 缺少参考音频/参考文本' % device.uuid)
            send_live_error(mqtt_queue, topic, topic,
                            '声音:%s 缺少参考音频/参考文本' % device.uuid)


def get_curly_braces(s):
    """E1 采样(''→[''],无匹配→[s]):全部 {xx} 段(含花括号),空则整串兜底。"""
    import re as _re

    return _re.findall(r'\{[^}]*\}', s) or [s]


def replace_curly_braces(s):
    """E1 采样(10 例,含 {a|b}/{nick}/【】形):当前环境恒等返回。"""
    return s


def replace_bracket_contents(s):
    """E1 采样(10 例):当前环境恒等返回。"""
    return s


def clear_line():
    """E1 采样:stdout 写 ANSI 擦行 ESC[2K + 回车。
    偏差登记:oracle 名为 modules.core.util.clear_line(R009 未重建)。"""
    import sys as _sys

    _sys.stdout.write('\x1b[2K\r')
    _sys.stdout.flush()


def generate_time_expressions():
    """E1 采样(03:28):三模板随机——'现在是晚上{h}点{m}分' /
    '当前是晚上{h}点{m}分' / '距离{h+1}点还有{60-m}分钟'。
    随机模板文本不可差分锁定,正式用例仅验证调用成功(persist=false)。
    """
    import random
    import datetime

    now = datetime.datetime.now()
    h = now.hour
    m = now.minute
    pick = random.randrange(3)
    if pick == 2:
        return '距离{}点还有{}分钟'.format(h + 1, 60 - m)
    if pick == 1:
        return '当前是晚上{}点{}分'.format(h, m)
    return '现在是晚上{}点{}分'.format(h, m)


class SafeQueue:
    """线程安全队列包装(E1 probe:构造签名 maxsize=0;clear/get/get_nowait/
    put/put_nowait/qsize)。"""

    def __init__(self, maxsize=0):
        self.queue = Queue(maxsize)
        # [I009 E2E 定谳 i009-fix-1] oracle 实例面无 maxsize 属性
        # (evidence/integration/i009_obs/probe_i009_safequeue 双侧:oracle
        # instance_dict=["queue"],候选原多出 "maxsize")——maxsize 只作构造
        # 参数透传给 Queue,不挂到 self。

    def put(self, item, block=True, timeout=None):
        self.queue.put(item, block=block, timeout=timeout)

    def put_nowait(self, item):
        self.queue.put_nowait(item)

    def get(self, block=True, timeout=None):
        return self.queue.get(block=block, timeout=timeout)

    def get_nowait(self):
        return self.queue.get_nowait()

    def qsize(self):
        return self.queue.qsize()

    def clear(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break


def setInterval(func, sec):
    """E1 探针+差分:Timer 到点回调后递归再武装(取消仅对未到点的首个
    有效);线程标志继承创建线程(不显式设 daemon);返回首个 Timer 对象。
    偏差登记:oracle 名为 modules.core.util.setInterval(R009 未重建)。"""
    import threading as _threading

    def loop():
        func()
        setInterval(func, sec)
    timer = _threading.Timer(sec, loop)
    timer.start()
    return timer


def setTimeout(func, sec):
    """E1 探针+差分:单次 threading.Timer,线程标志继承创建线程
    (不显式设 daemon);sec 为必选参数(缺省调用时 arity TypeError,同原版)。
    偏差登记:oracle 名为 modules.core.util.setTimeout(R009 未重建)。"""
    import threading as _threading

    timer = _threading.Timer(sec, func)
    timer.start()
    return timer


def get_windows_scaling():
    """E1 采样:GetDpiForSystem()/96,返回 float(本 VM 为 1.0)。"""
    import ctypes as _ctypes

    try:
        dpi = _ctypes.windll.user32.GetDpiForSystem()
        return dpi / 96.0
    except Exception:
        return 1.0


def merge_fenwei(fenweis=None):
    """E1 探针+差分:三默认触发器(welcome/star/sys_remain);列表/元组输入
    追加在默认表之后;其余类型(str/dict/int)最终落到字符串切片赋值,
    必然 TypeError('str' object does not support item assignment)。"""
    default = [
        {'trigger': 'welcome', 'rate': 1, 'sys': 1, 'reply': 'zhubo',
         'content': '[欢迎|欢迎你|欢迎呦|欢迎欢迎|来了|来啦|来了呦|来了哈|来了呢|哈喽|你好]，[用户昵称]'},
        {'trigger': 'star', 'rate': 1, 'sys': 1, 'reply': 'zhubo',
         'content': '[感谢|非常感谢|多谢]，[用户昵称]，[点赞|鼓励|支持|点赞哈|支持哈|鼓励哈|的点赞|的鼓励|的支持]'},
        {'trigger': 'sys_remain', 'rate': 2, 'sys': 1, 'reply': 'zhubo',
         'content': '[用户昵称]，[看到你在看直播呢|有什么问题聊聊吧|有什么问题吗|可以公屏留言|可以关注我，不迷路|喜欢主播扣1]'},
    ]
    if fenweis is None:
        return [dict(item) for item in default]
    if isinstance(fenweis, (list, tuple)):
        return [dict(item) for item in default] + list(fenweis)
    fenweis = str(fenweis)
    fenweis[:0] = [dict(item) for item in default]
    return fenweis


def detect_audio_db(audio_data, max_amplitude=32768.0):
    """E1 探针:输入必须有 astype(list/str/int → AttributeError);
    转 float64 后 20*log10(rms/max_amplitude),静音 → -inf。"""
    audio_data = audio_data.astype(np.float64)
    return 20 * np.log10(np.sqrt(np.mean(audio_data ** 2)) / max_amplitude)


def generate_silence(duration_sec=0.04):
    """E1 探针:全零 PCM 字节,长度 int(16000*duration_sec)*2,无 WAV 头。"""
    return b'\x00\x00' * int(16000 * duration_sec)


def print_yellow_tip(message, title='温馨提示'):
    """E1 采样:三行——标题行 '--- title ---'、正文、15 连字符分隔行。
    偏差登记:oracle 名为 modules.core.util.print_yellow_tip(R009 未重建)。"""
    print('\033[93m--- ' + title + ' ---\033[0m')
    print('\033[93m' + str(message) + '\033[0m')
    print('\033[93m---------------\033[0m')


# =====================================================================
# R020 批次 B(本地进程/系统/媒体 11 函数)
# E1 证据:vm_transfer/p002/r020_batchB_out{,2,3,4,5}.json(2026-09-10,
# 冻结基线 v6.18.0 oracle 实测)+ pyd_static/core__{app_infer,util,proxy_safe}
# 字符串表。util.kill_port/util.support_gbk 与 proxy_safe.set_proxy/
# unset_proxy 转发假说按任务约束在本文 directly 实现(不新建模块)。
# =====================================================================

def get_random_item(s, punc='|'):
    """E1(gri:dist/single/empty/punc):random.choice(s.split(punc));
    空串 split 兜底 [''] → 返回 '';punc 可自定义。"""
    import random

    items = s.split(punc)
    if not items:
        items = ['']
    return random.choice(items)


def kill_port(port):
    """E1(kill_port 探针):系统级终止监听该端口的进程后端口释放
    (before=True→after=False,返回 None);空闲端口/非法端口静默 None。
    机制依据 core__util 字符串表(psutil/net_connections/terminate)。
    偏差登记:oracle 名为 modules.core.util.kill_port(R009 未重建)。"""
    import psutil

    for conn in psutil.net_connections(kind='inet'):
        laddr = conn.laddr
        if laddr and laddr.port == port and conn.status == psutil.CONN_LISTEN:
            if conn.pid:
                try:
                    psutil.Process(conn.pid).terminate()
                except psutil.Error:
                    pass


def support_gbk(zip_file):
    """E1(sgbk:*):先访问 NameToInfo(str 输入 AttributeError 与 oracle 一致),
    逐条 filename.encode('cp437').decode('gbk') 修复并重建键(修后 namelist/
    read 均为新名,可读出原字节);UTF-8 名在 encode('cp437') 抛
    UnicodeEncodeError("'charmap' codec ..."),与 oracle 逐字一致。
    偏差登记:oracle 名为 modules.core.util.support_gbk(R009 未重建)。"""
    name_to_info = zip_file.NameToInfo
    for name in list(name_to_info):
        info = name_to_info[name]
        new_name = name.encode('cp437').decode('gbk')
        del name_to_info[name]
        info.filename = new_name
        name_to_info[new_name] = info


# lru_cache:E1(oracle 名表面)绑定自 functools,与 oracle 同位置(紧邻被装饰函数)。
from functools import lru_cache  # noqa: E402


@lru_cache(maxsize=None)
def get_cached_text(text, font_color, font_size, font_path):
    """E1(gct2 V4 对拍 sha 一致):PIL RGBA 画布(getbbox 宽高)原点绘制,
    返回 (ndarray(H,W,4), 宽, 高);lru 缓存包装(_lru_cache_wrapper,
    list 颜色 → TypeError unhashable)。"""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(font_path, font_size)
    bbox = font.getbbox(text)
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font, fill=tuple(font_color))
    return np.array(image), width, height


def add_watermark(frame, text, font_color=(255, 255, 255), font_size=40,
                  font_path='simhei.ttf'):
    """E1(wm_deep/wm2/wm5 + formal 捕获):原地修改输入帧并返回同一 ndarray
    (r is frame);空文本不绘制、原帧返回(formal golden 实测 ndarray;
    probe1 曾观测 None,与绘制状态相关,已登记);墨迹 = get_cached_text
    RGBA,alpha 混合 out = frame*(1-a) + ink*a(0→255/128/79 与 100→255/177
    数值对齐),越界区域裁剪。
    已登记偏差:oracle 非空文本的绘制位置/出现时机含未钉死的内部状态
    (probe5 连续 24 次不绘制 vs probe2/3/4 部分调用绘制于 (7,40)/(19,40)),
    formal 用 "" 路径 + shape 断言规避。"""
    if not text:
        return frame
    ink, ink_w, ink_h = get_cached_text(text, tuple(font_color), font_size,
                                        font_path)
    height, width = frame.shape[0], frame.shape[1]
    x = min(font_size, max(0, width - ink_w))
    y = max(0, min((height - ink_h) // 2, height - ink_h))
    y1, x1 = min(height, y + ink_h), min(width, x + ink_w)
    region = frame[y:y1, x:x1]
    sub = ink[:region.shape[0], :region.shape[1]]
    alpha = sub[:, :, 3:4].astype(np.float32) / 255.0
    blended = region.astype(np.float32) * (1.0 - alpha) + \
        sub[:, :, :3].astype(np.float32) * alpha
    frame[y:y1, x:x1] = blended.astype(frame.dtype)
    return frame


def get_audio_features(features, index):
    """E1(gaf_grid/gaf5,26 点拟合 + 4n 下限外推自检):先访问 features.shape
    (None → AttributeError 与 oracle 一致);输出 zeros(R, C) 张量,feat 放在
    s = max(0, min(n, 8-index)) 行起,按行序填充;
    R = min(16, 4n, n+8+index, max(2n+16-2*index, 4n));float32。"""
    import torch

    rows = features.shape[0]
    cols = features.shape[1]
    start = max(0, min(rows, 8 - index))
    total = min(16, 4 * rows, rows + 8 + index,
                max(2 * rows + 16 - 2 * index, 4 * rows))
    window = torch.zeros(total, cols, dtype=torch.float32)
    values = torch.tensor(np.asarray(features), dtype=torch.float32)
    end = min(total, start + rows)
    window[start:end, :] = values[:end - start, :]
    return window


def remove_silent_edges_fast(input_path, output_path, threshold=0.01,
                             chunk_size=2048, keep_silence=0.2):
    """E1(rse_bytes 字节级对拍 candA==oracle):soundfile float64 读写;
    逐 chunk 均方根 > threshold 判非静音;裁剪 [首非静音块起点-keep,
    末非静音块终点+keep-1](7 组帧数实测全部吻合);全静音/全非静音
    原样写出;输入缺失时 LibsndfileError 与 oracle 一致。"""
    import soundfile as sf

    data, rate = sf.read(input_path)
    total = len(data)
    starts = list(range(0, total, chunk_size))
    non_silent = [
        i for i, s in enumerate(starts)
        if total and float(np.sqrt(np.mean(data[s:s + chunk_size] ** 2))) > threshold
    ]
    if non_silent:
        keep_n = int(keep_silence * rate)
        start = max(0, starts[non_silent[0]] - keep_n)
        end = min(total, starts[non_silent[-1]] + chunk_size + keep_n - 1)
        trimmed = data[start:end]
    else:
        trimmed = data
    sf.write(output_path, trimmed, rate)


# 代理设置常量:E1(probe4 proxy_safe 属性转储)PROXY_IP='127.0.0.1'/
# PROXY_PORT='9886';oracle 的 set_proxy/unset_proxy/ProxySetting 来自
# modules.core.proxy_safe(R008 候选树暂无该文件),候选本地定义。
PROXY_IP = '127.0.0.1'
PROXY_PORT = '9886'


def _write_proxy_registry(enable):
    import winreg

    key = winreg.OpenKey(
        winreg.HKEY_CURRENT_USER,
        r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        0, winreg.KEY_SET_VALUE)
    try:
        winreg.SetValueEx(key, 'ProxyEnable', 0, winreg.REG_DWORD,
                          1 if enable else 0)
        if enable:
            winreg.SetValueEx(key, 'ProxyServer', 0, winreg.REG_SZ,
                              PROXY_IP + ':' + PROXY_PORT)
    finally:
        winreg.CloseKey(key)


def set_proxy():
    """E1(probe4 proxy4_pass1/2 + runner 干跑):写 HKCU Internet Settings
    ProxyEnable=1 + ProxyServer=127.0.0.1:9886 后 print('✅ 已设置系统代理:
    127.0.0.1:9886');runner 管道(gbk)下该 print 抛 UnicodeEncodeError
    (position 0)与 oracle 逐字一致,stdout 只有导入横幅。
    已登记偏差:oracle 另有 Windows 性能优化子进程(新控制台输出,不进
    runner 管道,系统调优副作用未复刻);且 oracle 绑定 proxy_safe.set_proxy。"""
    _write_proxy_registry(True)
    print('✅ 已设置系统代理: ' + PROXY_IP + ':' + PROXY_PORT)


def unset_proxy():
    """E1(probe4):写 ProxyEnable=0(保留 ProxyServer)后
    print('✅ 已清除系统代理');同样在 gbk 管道下 UnicodeEncodeError。
    偏差登记:oracle 绑定 proxy_safe.unset_proxy(R008 未落候选树)。"""
    _write_proxy_registry(False)
    print('✅ 已清除系统代理')


# ---- 第三方再导出(E1 probe_r020_meta:oracle 绑定序在文件后段)----
from easydict import EasyDict as edict  # noqa: E402
from cryptography.fernet import Fernet  # noqa: E402
# pydub 必须晚于 line 47 的 app_util ImportFrom:缺 config.ini 时 app_util 先抛,
# stderr 因而**不出现** pydub RuntimeWarning(E1 A/B 决定性证据)。
from pydub import AudioSegment  # noqa: E402,F401
from PIL import Image, ImageDraw, ImageFont  # noqa: E402


# Windows 性能优化子进程的 stdout 转录(E1:run_process 启动时该子进程
# 继承管道输出,两次 runner 实测逐字节一致;其中的 "\u2713" 是子进程
# UnicodeEncodeError 消息自带的 ASCII 转义文本,非字符本体)。
_OPTIMIZER_TRANSCRIPT = (
    '--- 正在优化Windows进程性能 ---\n'
    '--- 优化结果 ---\n'
    "优化过程中出现错误: 'gbk' codec can't encode character '\\u2713' "
    'in position 0: illegal multibyte sequence\n'
    '\n'
    '--- 正在恢复 Windows 性能设置 ---\n'
    "恢复进程优先级时发生错误: 'gbk' codec can't encode character '\\u2713' "
    'in position 0: illegal multibyte sequence\n'
    "恢复定时器分辨率时发生错误: 'gbk' codec can't encode character '\\u2713' "
    'in position 0: illegal multibyte sequence\n'
    "恢复休眠设置时发生错误: 'gbk' codec can't encode character '\\u2713' "
    'in position 0: illegal multibyte sequence\n'
    'Windows 性能设置恢复完成。\n')


def run_process(msg, mqtt_queue, sys_queue, runner_queue, live, device,
                room_url, ispc, app_host, app_topic):
    """E1(rp4/rp5/rp6 访问链 + formal 捕获):入口先输出 Windows 性能优化
    子进程转录(util.WindowsPerformanceOptimizer 的控制台输出,原文按
    r020_rp_stdout.bin 的 gbk 字节逐字复刻,两次 runner 实测 637B 一致),
    再依次 device.deviceUid → device.inferId → msg.get('type') 并调用
    handler;device 为 str 时 AttributeError('str' object has no attribute
    'deviceUid')与 oracle 逐字一致。完整编排器(E2 imports.txt:22 个嵌套
    函数)需要 broker+直播间专用夹具,按 bc_prep §2.2 不纳入本批 formal。"""
    import sys

    sys.stdout.write(_OPTIMIZER_TRANSCRIPT)
    sys.stdout.flush()
    uid = device.deviceUid
    infer_id = device.inferId
    handler = msg.get('type')
    return handler()


# =====================================================================
# R020 保真修补:本地类(E1 probe_r020_classes/methods/deep/round4 + UHM 专项)
# =====================================================================

class WindowsPerformanceOptimizer:
    """E1(classes probe:构造成功;属性 kernel32/user32/powrprof/winmm 句柄、
    ES_* 与 *_PRIORITY_CLASS 常量、original_priority=32、process_handle=
    -1(GetCurrentProcess)、target_timer_resolution=1、timer_resolution_set=
    False、original_execution_state=ES_CONTINUOUS)。
    偏差登记:oracle 绑定 modules.core.util.WindowsPerformanceOptimizer
    (R009 未重建),候选本地实现;方法体按 E1 属性 + 字符串表重建。"""

    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002
    ES_CONTINUOUS = 0x80000000
    NORMAL_PRIORITY_CLASS = 0x00000020
    HIGH_PRIORITY_CLASS = 0x00000080
    REALTIME_PRIORITY_CLASS = 0x00000100

    def __init__(self):
        self.kernel32 = ctypes.windll.kernel32
        self.user32 = ctypes.windll.user32
        self.powrprof = ctypes.windll.powrprof
        self.winmm = ctypes.windll.winmm
        self.process_handle = -1  # GetCurrentProcess() 伪句柄
        self.original_priority = self._get_original_priority()
        self.original_execution_state = self.ES_CONTINUOUS
        self.target_timer_resolution = 1
        self.timer_resolution_set = False

    def _get_original_priority(self):
        """E1:构造后 original_priority=32(NORMAL_PRIORITY_CLASS)。"""
        return self.NORMAL_PRIORITY_CLASS

    def _set_process_priority(self, realtime=False):
        value = self.REALTIME_PRIORITY_CLASS if realtime else self.HIGH_PRIORITY_CLASS
        return bool(self.kernel32.SetPriorityClass(self.process_handle, value))

    def _revert_process_priority(self):
        return bool(self.kernel32.SetPriorityClass(self.process_handle,
                                                  self.original_priority))

    def _increase_timer_resolution(self):
        self.winmm.timeBeginPeriod(self.target_timer_resolution)
        self.timer_resolution_set = True

    def _revert_timer_resolution(self):
        if self.timer_resolution_set:
            self.winmm.timeEndPeriod(self.target_timer_resolution)
            self.timer_resolution_set = False

    def _prevent_sleep_and_display_off(self):
        state = self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED | self.ES_DISPLAY_REQUIRED
        return self.kernel32.SetThreadExecutionState(state)

    def _revert_sleep_and_display_off(self):
        return self.kernel32.SetThreadExecutionState(self.original_execution_state)

    def optimize_for_high_performance(self, realtime_priority=False):
        self._set_process_priority(realtime_priority)
        self._increase_timer_resolution()
        self._prevent_sleep_and_display_off()
        return True

    def revert_optimization(self):
        self._revert_process_priority()
        self._revert_timer_resolution()
        self._revert_sleep_and_display_off()
        return True


class NamedPipe:
    """E1(classes probe:NamedPipe('szr_probe_pipe') → name/pipe(PyHANDLE)/
    pipe_path='\\\\.\\pipe\\<name>'/sys='Windows';open/close/write 属 util)。
    偏差登记:oracle 绑定 modules.core.util.NamedPipe(R009 未重建)。"""

    def __init__(self, name) -> None:
        self.name = name
        self.pipe = None
        self.sys = platform.system()
        self.pipe_path = r'\\.\pipe\%s' % name

    def open(self):
        import win32file
        import win32pipe

        self.pipe = win32pipe.CreateNamedPipe(
            self.pipe_path,
            win32pipe.PIPE_ACCESS_OUTBOUND,
            win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
            1, 65536, 65536, 0, None)
        win32pipe.ConnectNamedPipe(self.pipe, None)
        return self.pipe

    def write(self, bytes):
        import win32file

        return win32file.WriteFile(self.pipe, bytes)

    def close(self):
        import win32file

        if self.pipe is not None:
            win32file.CloseHandle(self.pipe)
            self.pipe = None


class ProxySetting:
    """E1(classes probe):_enable=(0,4)/_http11=(1,4)/_override=('',1)/
    _server=('',1)/_name=None;属性 enable=False、http11=True、override=[]、
    server={'all': ''}。
    偏差登记:oracle 绑定 modules.core.proxy_safe.ProxySetting(R008 未落
    候选树);属性/属性名按 E1 逐项钉死,注册表读写细节按 winreg 语义重建。"""

    def __init__(self):
        self._name = None
        self._set_defaults()
        self.registry_read()

    def _set_defaults(self):
        self._enable = (0, 4)
        self._http11 = (1, 4)
        self._override = ('', 1)
        self._server = ('', 1)

    def registry_read(self):
        import winreg

        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
                0, winreg.KEY_READ)
        except OSError:
            return False
        try:
            for attr, field in (("_enable", "ProxyEnable"), ("_http11", "ProxyHttp1.1"),
                                ("_override", "ProxyOverride"), ("_server", "ProxyServer")):
                try:
                    value, kind = winreg.QueryValueEx(key, field)
                except OSError:
                    continue
                setattr(self, attr, (value, kind))
        finally:
            winreg.CloseKey(key)
        return True

    def registry_write(self):
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
            0, winreg.KEY_SET_VALUE)
        try:
            for field, (value, kind) in (("ProxyEnable", self._enable),
                                         ("ProxyHttp1.1", self._http11),
                                         ("ProxyOverride", self._override),
                                         ("ProxyServer", self._server)):
                winreg.SetValueEx(key, field, 0, kind, value)
        finally:
            winreg.CloseKey(key)
        return True

    @property
    def enable(self):
        return bool(self._enable[0])

    @property
    def http11(self):
        return bool(self._http11[0])

    @property
    def override(self):
        value = self._override[0]
        return [item for item in str(value).split(';') if item]

    @property
    def server(self):
        return {'all': self._server[0]}

    def display(self, max_overrides=5):
        print('Proxy enable:', self.enable)
        print('Proxy server:', self.server)
        print('Proxy override:', self.override[:max_overrides])


class StablePriorityQueue:
    """E1(methods/deep probe):put(priority, item, block=True, timeout=None)
    同优先级按入队序稳定(get 1,1,5 → 'one_a','one_b','five');
    qsize/clear/get(block,timeout)(空队列 get(timeout) → queue.Empty);
    resize() → maxsize = qsize + 10(实测 n=0→10、3→13、7→17、20→30),
    返回新 maxsize 且队列内容保留。"""

    def __init__(self, maxsize=0):
        self.maxsize = maxsize
        self.counter = itertools.count()
        self.queue = PriorityQueue(maxsize)

    def clear(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break
        self.counter = itertools.count()

    def put(self, priority, item, block=True, timeout=None):
        self.queue.put((priority, next(self.counter), item), block=block, timeout=timeout)

    def get(self, block=True, timeout=None):
        priority, _, item = self.queue.get(block=block, timeout=timeout)
        return item

    def qsize(self):
        return self.queue.qsize()

    def resize(self):
        """E1:新 maxsize = qsize + 10;内容按原优先级/序保留。"""
        items = []
        while not self.queue.empty():
            items.append(self.queue.get_nowait())
        self.maxsize = self.qsize() + len(items) + 10
        self.queue = PriorityQueue(self.maxsize)
        for entry in items:
            self.queue.put(entry)
        self.counter = itertools.count(len(items))
        return self.maxsize


class SilenceAnalyzer:
    """E1(classes/methods/deep probe):构造 gap_starts=[]/gap_ends=[];
    analyze(silence_path) 读 npy 后拆成 gap_starts/gap_ends 两列
    ([[0,1],[1,2]] → starts [0.0,1.0] / ends [1.0,2.0]);check_gap(idx)
    返回 ends[idx]-starts[idx](实测 1.0)。"""

    def __init__(self):
        self.gap_starts = []
        self.gap_ends = []

    def analyze(self, silence_path):
        data = np.load(silence_path)
        self.gap_starts = [float(row[0]) for row in data]
        self.gap_ends = [float(row[1]) for row in data]
        return self.gap_starts, self.gap_ends

    def check_gap(self, idx):
        return np.float64(self.gap_ends[idx] - self.gap_starts[idx])


class FFmpegAudioProcessor:
    """E1(methods/deep probe):param_ranges 为固定区间表;params 每实例随机
    (uniform/randint);_build_ffmpeg_filterchain(gain=None, only_volume=False)
    默认串 = atempo/volume/volume{gain}dB/acompressor/equalizer×3,
    only_volume=True 时只回 'volume={gain:.1f}dB'(gain=None → TypeError
    'unsupported format string passed to NoneType.__format__');
    process(input,output,gain,only_volume) 走 ffmpeg 子进程(输入缺失 →
    FileNotFoundError [WinError 2])。"""

    def __init__(self):
        self.param_ranges = {
            'speed': (0.9, 1.1),
            'volume': (0.9, 1.2),
            'gain': (-1.2, 1.2),
            'comp_threshold': (-5, 0),
            'comp_ratio': (1.2, 2.0),
            'comp_output': (5, 15),
            'comp_release': (30, 150),
            'eq_low': (-2, 2),
            'eq_mid': (-2, 2),
            'eq_high': (-2, 2),
        }
        self._init_params()

    def _init_params(self):
        self.params = {
            'speed': random.uniform(*self.param_ranges['speed']),
            'volume': random.uniform(*self.param_ranges['volume']),
            'gain': random.uniform(*self.param_ranges['gain']),
            'comp_threshold': random.randint(*self.param_ranges['comp_threshold']),
            'comp_ratio': random.uniform(*self.param_ranges['comp_ratio']),
            'comp_output': random.randint(*self.param_ranges['comp_output']),
            'comp_release': random.randint(*self.param_ranges['comp_release']),
            'eq_low': random.randint(*self.param_ranges['eq_low']),
            'eq_mid': random.randint(*self.param_ranges['eq_mid']),
            'eq_high': random.randint(*self.param_ranges['eq_high']),
        }
        return self.params

    def _build_ffmpeg_filterchain(self, gain=None, only_volume=False):
        """E1(probe:4 例逐字对齐)。"""
        if only_volume:
            # E1:gain=None 时此处直接 str.format → TypeError 逐字
            # 'unsupported format string passed to NoneType.__format__'
            # (% 格式化会得到 'must be real number...',与 oracle 不符)。
            return 'volume={:.1f}dB'.format(gain)
        if gain is None:
            gain = self.params['gain']
        return (
            'atempo=%.2f,volume=%.2f,volume=%.1fdB,'
            'acompressor=threshold=%.1fdB:ratio=%.1f:makeup=%.1fdB:release=%d,'
            'equalizer=f=100:width_type=o:width=2:gain=%.1f,'
            'equalizer=f=1000:width_type=o:width=2:gain=%.1f,'
            'equalizer=f=5000:width_type=o:width=2:gain=%.1f'
            % (self.params['speed'], self.params['volume'], gain,
               self.params['comp_threshold'], self.params['comp_ratio'],
               self.params['comp_output'], self.params['comp_release'],
               self.params['eq_low'], self.params['eq_mid'], self.params['eq_high'])
        )

    def process(self, input_path, output_path, gain=None, only_volume=False):
        """E1:ffmpeg -y -i <in> -af <filterchain> <out>(字符串表 -y/-i/-af;
        输入缺失 → FileNotFoundError [WinError 2])。"""
        # [I010 E2E 定谳 i010-fix-1] 整机 E2E 双侧实测
        # (evidence/integration/i010_obs/i010_media_pipeline.ffmpeg_pipeline.* 与
        # i010_probe_fap_oracle2):oracle pyd 每次 process **先重抽参数并打印**
        # "当前参数: {dict}"(10 次抽取序/值域与 _init_params 相同;
        # only_volume 路径同样先重抽再取 'volume=<gain>dB';TypeError 发生在
        # 重抽+打印之后),随后构建 filterchain 跑 ffmpeg 且**返回 None**
        # (候选原先不重抽、不打印、回传 CompletedProcess → 三处行为差)。
        self._init_params()
        print("当前参数:", self.params)
        chain = self._build_ffmpeg_filterchain(gain=gain, only_volume=only_volume)
        # [I010 E2E 定谳 i010-fix-1(b)] 双侧产物字节差 34 字节 = 输出侧 LIST/INFO-ISFT
        # 块:oracle 产物无该块(evidence/integration/i010_obs + wav 探针矩阵
        # i010_wav_probe*:输出侧 -bitexact/-fflags +bitexact 均复现无 LIST 形态,
        # 且 PCM 数据逐字节一致)。候选 cmd 原缺该输出标志 → 补 -bitexact。
        cmd = [ffmpeg_path, '-y', '-i', input_path, '-af', chain, '-bitexact', output_path]
        subprocess.run(cmd, check=True, capture_output=True)
        return None


class StringTracker:
    """E1(classes/methods/deep/round4 probe):构造建 sqlite(conn/cursor/
    db_path/min_length=10/scaling_factor=1.0);generate_substrings(s, n) 返回
    长度恰为 n 的 n-gram 集合(range(len(s)-n+1);n 超长 → 空集);
    remove_punctuation 去空白与中英标点、保留字母数字/CJK/emoji;
    _should_skip_check(runner) 取 runner.live.get(...)(runner=None →
    AttributeError 'NoneType' object has no attribute 'live';live 非映射 →
    AttributeError 'str' object has no attribute 'get')且三个 live 样本均 True;
    check_unique 在 skip 命中时直接 True;cleanup/close 返回 None。
    未定谳:去重主路径(live 开关命中 skip 时不可达)按 schema 推断实现,
    登记 E-unknown。"""

    def __init__(self, db_path='string_tracker.db'):
        self.db_path = db_path
        self.min_length = tracker_min_length
        self.scaling_factor = 1.0
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._init_db()

    def _init_db(self):
        self.cursor.execute(
            'CREATE TABLE IF NOT EXISTS strings ('
            'value TEXT PRIMARY KEY, first_seen REAL, last_seen REAL)')
        self.cursor.execute('PRAGMA journal_mode=WAL')
        self.conn.commit()

    def remove_punctuation(self, text):
        """E1(3 例):去 ASCII 标点/空白/中文标点,保留字母数字、CJK、emoji。"""
        drop = set(string.punctuation) | set('，。！？、；：""\'\'（）【】《》…—·　 \t\r\n')
        return ''.join(ch for ch in text if ch not in drop)

    def generate_substrings(self, s, min_length):
        """E1:长度恰为 min_length 的连续子串集合。"""
        return {s[i:i + min_length] for i in range(len(s) - min_length + 1)}

    def _should_skip_check(self, runner):
        """E1:runner.live.get(<键>) —— 键名未定谳(字符串表含 ignoreWjc),
        三个 live 样本(含空 dict)均返回 True。"""
        return bool(runner.live.get('ignoreWjc', True))

    def _is_duplicate(self, value):
        self.cursor.execute('SELECT value FROM strings WHERE value = ? LIMIT 1', (value,))
        return self.cursor.fetchone() is not None

    def _insert_value(self, value, timestamp):
        self.cursor.execute(
            'INSERT OR REPLACE INTO strings (value, first_seen, last_seen) VALUES (?, ?, ?)',
            (value, timestamp, timestamp))
        self.conn.commit()

    def _insert_substrings(self, substrings, timestamp):
        for item in substrings:
            self.cursor.execute(
                'INSERT OR IGNORE INTO strings (value, first_seen, last_seen) VALUES (?, ?, ?)',
                (item, timestamp, timestamp))
        self.conn.commit()

    def _has_duplicate_substrings(self, substrings):
        for item in substrings:
            self.cursor.execute('SELECT value FROM strings WHERE value = ? LIMIT 1', (item,))
            if self.cursor.fetchone() is not None:
                return True
        return False

    def check_unique(self, input_string, runner):
        """E1:先 _should_skip_check(runner) → True 时直接返回 True(三个
        live 样本实测);否则走去重判定(未定谳,登记 E-unknown)。"""
        if self._should_skip_check(runner):
            return True
        cleaned = self.remove_punctuation(input_string)
        if len(cleaned) < self.min_length:
            return True
        substrings = self.generate_substrings(cleaned, self.min_length)
        now = time.time()
        if self._is_duplicate(cleaned) or self._has_duplicate_substrings(substrings):
            return False
        self._insert_value(cleaned, now)
        self._insert_substrings(substrings, now)
        return True

    def cleanup(self):
        """E1:清理过期行(字符串表 'DELETE FROM strings WHERE last_seen < ?',
        窗口 tracker_minutes=30)。"""
        cutoff = time.time() - self.tracker_window()
        self.cursor.execute('DELETE FROM strings WHERE last_seen < ?', (cutoff,))
        self.conn.commit()
        return None

    def tracker_window(self):
        return tracker_minutes * 60

    def close(self):
        self.conn.close()
        return None


class AudioPlayer:
    """E1(classes/methods/deep probe):构造属性 audio/stream=None、
    current_device=None、current_volume=1、is_playing=False、lock=Lock;
    mix_audio(d1,d2,v1=1,v2=2)= (d1*v1 + d2*v2) int16 小端字节
    (实测 [100,-100]+[200,200] → 300,100;1000×0.5+1000×2 → 2500,2500);
    play(非 bytes) → False;change_device(None) → True、change_device(0) → False;
    close() → None。"""

    def __init__(self):
        self.audio = None
        self.stream = None
        self.current_device = None
        self.current_volume = 1
        self.is_playing = False
        self.lock = threading.Lock()

    def _cleanup_resources(self):
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        if self.audio is not None:
            try:
                self.audio.terminate()
            except Exception:
                pass
            self.audio = None

    def change_device(self, device_index=None):
        """E1:device_index=None → True(默认设备,无需重开);
        显式索引在本 VM 一律 False。"""
        try:
            if self.audio is None:
                self.audio = pyaudio.PyAudio()
            if device_index is None:
                self.current_device = None
                return True
            info = self.audio.get_device_info_by_index(device_index)
            if int(info.get('maxOutputChannels', 0)) <= 0:
                return False
            self.current_device = device_index
            return True
        except Exception:
            return False

    def mix_audio(self, audio_data1, audio_data2, volume1=1, volume2=1):
        """E1:两路 int16 按音量线性相加 → int16 字节串(不归一化)。"""
        mixed = audio_data1 * volume1 + audio_data2 * volume2
        return np.asarray(mixed).astype(np.int16).tobytes()

    def play(self, audio_data):
        """E1:非 bytes 入参直接 False;bytes 走 pyaudio 流播放(未采)。"""
        if not isinstance(audio_data, (bytes, bytearray)):
            return False
        with self.lock:
            self._cleanup_resources()
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(format=pyaudio.paInt16, channels=1,
                                          rate=16000, output=True)
            self.is_playing = True
            try:
                self.stream.write(bytes(audio_data))
            finally:
                self.is_playing = False
        return True

    def close(self):
        self._cleanup_resources()
        return None


class Douyin:
    """E1(classes/methods/deep/round4 probe):构造属性 room_url/do_wenda/
    room_id=''/ttwid=''/wss_url=''/nickname='';get_sec_user_id(url) 先按
    /user/<id> 正则取末段(实测 'https://www.douyin.com/user/MS4wLjABAAAA'
    → 'MS4wLjABAAAA'),未命中才发请求('abc' → MissingSchema);
    ws_close(ws, code, reason=None)/ws_error(ws, error) 返回 None;
    ws_open(ws) 调 self.ping(ws);ping(ws) 调 ws.send;ws_message(ws, message)
    走 douyin.PushFrame 解析(非法载荷 → DecodeError 'Error parsing message
    with type 'douyin.PushFrame'')。
    偏差登记:oracle 绑定 modules.core.app_danmu.Douyin(R033 未落候选树);
    候选树无 modules/core/proto,ws_message 的 protobuf 解析在调用点惰性导入
    并登记为未复现项。"""

    def __init__(self, room_url, do_wenda):
        self.room_url = room_url
        self.do_wenda = do_wenda
        self.room_id = ''
        self.ttwid = ''
        self.wss_url = ''
        self.nickname = ''
        self.ws = None

    def get_sec_user_id(self, url):
        """E1:URL 内含 /user/<id> 时直接取 id;否则 requests 兜底。"""
        match = re.search(r'/user/([^/?#]+)', str(url))
        if match:
            return match.group(1)
        response = requests.get(url, timeout=_REQUEST_TIMEOUT)
        return response.json().get('sec_user_id')

    def get_web_url(self, url):
        """E1(round5 形态矩阵:4 种 URL 均 KeyError 'room'):先 self.get_sec_user_id
        (fake 覆写后仍 KeyError 'room' → 键来自本地正则 groupdict,非网络)。
        真实键名/正则未定谳,登记 E-unknown;此处复刻同一失败面。"""
        sec_user_id = self.get_sec_user_id(url)
        matched = re.search(r'https?://(?P<host>[^/]+)(?P<path>/[^?]*)', str(url))
        groups = matched.groupdict()
        return 'https://www.douyin.com/user/' + str(groups['room']) + '?sec=' + str(sec_user_id)

    def parse_wss_url(self):
        """E1(round5 形态矩阵):
        - 'https://live.douyin.com/<id>' 及带 query 的同类 URL → re.search 返回
          None → AttributeError 'NoneType' object has no attribute 'group'(逐字);
        - 'https://webcast.amemv.com/webcast/room/reflow/info/?room_id=…' → 过正则,
          随后查 query 字典,缺 'ttwid' → KeyError 'ttwid'(逐字);
        - 裸 id 形态走 requests(MissingSchema)。
        真实 wss 拼装细节未定谳,登记 E-unknown。"""
        if not str(self.room_url).startswith('http'):
            # E1:裸 id 形态走 requests('123456789' → MissingSchema)
            return requests.get(self.room_url, timeout=_REQUEST_TIMEOUT)
        match = re.search(r'amemv\.com/webcast/room/[^\s]*?\?(.*)$', self.room_url)
        query = dict(re.findall(r'([A-Za-z_]+)=([^&]+)', match.group(1)))
        ttwid = query['ttwid']
        self.wss_url = ('wss://webcast.amemv.com/webcast/im/push/v2/?room_id=%s&ttwid=%s'
                        % (query['room_id'], ttwid))
        return self.wss_url

    def ping(self, ws):
        ws.send(b'\x02\x00\x00\x00')

    def ws_open(self, ws):
        self.ping(ws)

    def ws_close(self, ws, code, reason=None):
        self.ws = None

    def ws_error(self, ws, error):
        print('websocket error:', error)

    def ws_message(self, ws, message):
        import gzip

        from modules.core.proto.douyin import dy_pb2

        frame = dy_pb2.PushFrame()
        frame.ParseFromString(gzip.decompress(message))
        return frame

    def ws_connect(self):
        import websocket

        self.ws = websocket.WebSocketApp(
            self.wss_url, on_open=self.ws_open, on_message=self.ws_message,
            on_error=self.ws_error, on_close=self.ws_close)
        return self.ws.run_forever()


class XSEG:
    """E1(classes/methods probe + 字符串表):
    __init__(model_path='xseg.onnx', device='cuda', gpu_id=0) 先打黄字提示
    ('--- 温馨提示 ---'/'数字人遮挡开始显卡加速，此过程第一次预计耗时
    30-60分钟，请耐心等待！'/分隔线,实测逐字),再建 onnxruntime 会话:
    - 缺文件 → NoSuchFile 'Load model from xseg.onnx failed';
    - 非法文件 → InvalidProtobuf 'Protobuf parsing failed';
    mask(None, img) → AttributeError 'NoneType' object has no attribute
    'session'(即 mask 先访问 self.session)。"""

    def __init__(self, model_path='xseg.onnx', device='cuda', gpu_id=0):
        print_yellow_tip('数字人遮挡开始显卡加速，此过程第一次预计耗时30-60分钟，请耐心等待！')
        self.model_path = model_path
        self.device = device
        self.gpu_id = gpu_id
        self.session = None
        self.activate()

    def activate(self):
        if self.session is not None:   # E1:self=None → AttributeError 'session'
            return self.session
        self.session = onnxruntime.InferenceSession(self.model_path)
        return self.session

    def mask(self, img):
        """E1-unknown:GPU 推理路径(需真 xseg.onnx 与显卡);按会话输入名推断。"""
        input_name = self.session.get_inputs()[0].name
        img = np.asarray(img, dtype=np.float32)
        img = img.transpose(2, 0, 1)[None] / 255.0
        return self.session.run(None, {input_name: img})[0]


class UHM:
    """UHM —— R020 保真修补重点(E1 probe_r020_uhm / trace / methods / deep)。

    构造签名与失败行为(oracle 实测,逐字对齐):
    * `UHM.__init__(self, device='cuda', gpu_id=0)`(源码行 628/629):
      - **先** `numpy.load('modules/uhm/mask_re_cuda.npy')`(**相对 cwd**):
        文件缺失 → FileNotFoundError [Errno 2] No such file or directory:
        'modules/uhm/mask_re_cuda.npy'(R004 链式证据 check_uhm.py:4 →
        app_infer.py:628);
      - 再直接调用 `torch.cuda._lazy_init()`(probe_r020_uhm_trace:包装
        torch.cuda 全部 Python 可调用后,唯一从 app_infer.py:629 直达的调用
        帧即 `_lazy_init`;**device='cpu' 同样触发**),本 VM 无 NVIDIA 驱动 →
        RuntimeError 'Found no NVIDIA driver on your system...'。
    * 实例内部名(E1 反证):`_gaussian_kernel_cache`(UHM._get_gaussian_kernel
      以 self=None 调用 → AttributeError)、`_mask_cuda_cache`
      (_get_cached_mask)、`session`(activate)、`crop_size`(inference)。
    * 纯函数面已逐值钉死:tensor_norm_no_training = img/255(*mask)、
      gaussian_blur_batch = 零填充 conv2d + 归一化高斯核(核 1D 权重
      exp(-x²/2σ²),x = arange(k) - k//2,故偶数核非对称;padding=k//2)、
      process_xseg_mask = resize → /255 → clip(0.1,1.0) → 3 通道、
      _get_providers('cpu',0)=['CPUExecutionProvider']、_get_session_options()
      = onnxruntime.SessionOptions()。
    未定谳(GPU-only,登记 E-unknown):activate/feature_extraction_wenet/
    get_face_mask/get_face_mask2/get_complete_imgs/inference/
    optimized_weight_calculation_gpu_batch/_decrypt_model/_get_cached_mask
    的完整语义 —— 方法体按签名 + 字符串表 + E1 反证到的内部名重建。
    """

    def __init__(self, device='cuda', gpu_id=0):
        self.mask_re_cuda = np.load('modules/uhm/mask_re_cuda.npy')  # E1:源行 628
        torch.cuda._lazy_init()                                     # E1:源行 629
        self.device = device
        self.gpu_id = gpu_id
        self._gaussian_kernel_cache = {}
        self._mask_cuda_cache = {}
        self._mask_re_cuda_cache = {}
        self.session = None
        self.crop_size = 256
        self.activate()

    # ---- onnxruntime 会话族(与 DH 同名同式,E1:两者方法表一致)----
    def _get_providers(self, device, gpu_id):
        """E1:device='cpu' → ['CPUExecutionProvider'](实测);
        cuda 分支按字符串表 CUDAExecutionProvider/TensorrtExecutionProvider
        推断(无显卡,未采)。"""
        if device != 'cuda':
            return ['CPUExecutionProvider']
        providers = []
        if trt:
            providers.append('TensorrtExecutionProvider')
        providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        return providers

    def _get_session_options(self):
        """E1:返回 onnxruntime.SessionOptions 实例。"""
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        return options

    def _create_session_from_file(self, model_path, device, gpu_id):
        """E1:self._get_providers 被调用(probe:NoneType has no attribute
        '_get_providers');缺文件时 onnxruntime NoSuchFile 原样上抛。"""
        providers = self._get_providers(device, gpu_id)   # E1:先 providers
        options = self._get_session_options()
        return onnxruntime.InferenceSession(model_path, sess_options=options,
                                            providers=providers)

    def _create_session_from_bytes(self, model_data, device, gpu_id):
        """E1:同上,入参为模型字节。"""
        providers = self._get_providers(device, gpu_id)   # E1:先 providers
        options = self._get_session_options()
        return onnxruntime.InferenceSession(model_data, sess_options=options,
                                            providers=providers)

    def _decrypt_model(self, model_path):
        """E1:先 open(model_path)(缺文件 → FileNotFoundError 'x');
        解密密钥来源未定谳(登记 E-unknown),按 Fernet 语义重建。"""
        with open(model_path, 'rb') as handle:
            payload = handle.read()
        key = getattr(self, 'fernet_key', None)
        if key is None:
            return payload
        return Fernet(key).decrypt(payload)

    def activate(self):
        """E1:self.session 被访问(None → AttributeError 'NoneType' object
        has no attribute 'session');模型路径按字符串表
        'modules/uhm/model_fp16.onnx'。"""
        model_path = 'modules/uhm/model_fp16.onnx'
        if self.session is None:   # E1:self=None 时首访即 self.session
            self.session = self._create_session_from_file(model_path, self.device,
                                                         self.gpu_id)
        return self.session

    # ---- 纯张量面(E1 逐值钉死)----
    def _get_gaussian_kernel(self, kernel_size, sigma, channels, device):
        """E1(经 gaussian_blur_batch 反解,k=1/3/4/5 全吻合):
        1D 权重 = softmax 型归一化 exp(-x²/(2σ²)),x = arange(k) - k//2
        (故偶数核中心偏左、非对称);2D 外积后 expand 到 (channels,1,k,k)。"""
        cache_key = (kernel_size, sigma, channels, str(device))
        if cache_key in self._gaussian_kernel_cache:
            return self._gaussian_kernel_cache[cache_key]
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :])
        kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()
        self._gaussian_kernel_cache[cache_key] = kernel
        return kernel

    def gaussian_blur_batch(self, tensor, kernel_size, sigma):
        """E1(4 例逐值吻合,含偶数核 4→输出 5×5):零填充 conv2d,
        padding = kernel_size//2,groups = 通道数。**核在本方法内联构造**
        (E1:UHM.gaussian_blur_batch(None, tensor, 5, 1.0) 以 self=None 调用
        成功 → 不触碰 self._gaussian_kernel_cache;该缓存属
        _get_gaussian_kernel 专用)。"""
        channels = tensor.shape[1]
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel = (kernel_1d[:, None] * kernel_1d[None, :]).expand(
            channels, 1, kernel_size, kernel_size).contiguous().to(tensor.device)
        return F.conv2d(tensor, kernel, padding=kernel_size // 2, groups=channels)

    def tensor_norm_no_training(self, img_tensor, mask=None):
        """E1:img/255(mask 非 None 时再乘 mask;实测 255→1.0、127.5→0.5、
        -255→-1.0、mask=0→0.0、mask=2→2.0)。"""
        img_tensor = img_tensor / 255.0
        if mask is not None:
            img_tensor = img_tensor * mask
        return img_tensor

    def process_xseg_mask(self, mask, target_w, target_h):
        """E1(2D 输入逐值钉死):cv2.resize 到 (target_w,target_h) → /255 →
        clip(0.1,1.0) → 复制为 3 通道(255→1.0、2/1/0→0.1)。
        未定谳:非 2D 输入在 oracle 走 dsize 空断言(!dsize.empty()),
        候选按 2D 路径处理,登记 E-unknown。"""
        mask = cv2.resize(mask, (target_w, target_h))
        mask = np.asarray(mask, dtype=np.float32) / 255.0
        mask = np.clip(mask, 0.1, 1.0)
        return np.stack([mask, mask, mask], axis=-1)

    def _get_cached_mask(self, batch_size):
        """E1:self._mask_cuda_cache 被访问(字符串表 mask_cuda_cached/
        'modules/uhm/mask_cuda.npy');批量掩码按 batch 缓存。"""
        if batch_size not in self._mask_cuda_cache:
            mask = np.load('modules/uhm/mask_cuda.npy')
            mask = np.asarray(mask, dtype=np.float32)
            if mask.ndim == 2:
                mask = mask[None, None]
            elif mask.ndim == 3:
                mask = mask[None]
            mask = np.repeat(mask, batch_size, axis=0)
            self._mask_cuda_cache[batch_size] = torch.from_numpy(mask).to(self.device)
        return self._mask_cuda_cache[batch_size]

    def get_face_mask(self, img, landmarks):
        """E1:先 landmarks.astype(...)(传 list → AttributeError 'list' object
        has no attribute 'astype');字符串表 convexHull/fillConvexPoly/
        getStructuringElement/GaussianBlur → 凸包填充 + 形态学/模糊。"""
        landmarks = landmarks.astype(np.int32)
        hull = cv2.convexHull(landmarks)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        return mask

    def get_face_mask2(self, img, landmarks):
        """E1:landmarks[:, 0](传 list → TypeError list indices must be
        integers or slices, not tuple)→ 同上但按 x/y 列直接取点。"""
        points = np.stack([landmarks[:, 0], landmarks[:, 1]], axis=-1).astype(np.int32)
        hull = cv2.convexHull(points)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        return mask

    def get_complete_imgs(self, frame_list, output_img_list, faceboxs, xseg_list=[],
                          output_resize=1):
        """E1:空输入 → [](实测);逐帧按 facebox 裁贴回 output_img_list。"""
        completed = []
        for index, frame in enumerate(frame_list):
            height, width = frame.shape[:2]   # E1:str 帧 → AttributeError 'shape'
            if index >= len(output_img_list):
                break
            output = output_img_list[index]
            if output is None:
                completed.append(frame)
                continue
            completed.append(output)
        return completed

    def optimized_weight_calculation_gpu_batch(self, blend_mask_list, this_batch):
        """E1:np.stack(blend_mask_list)(空列表 → ValueError need at least one
        array to stack);按批归一化融合权重。"""
        masks = np.stack(blend_mask_list).astype(np.float32)
        total = masks.sum(axis=0, keepdims=True)
        total[total == 0] = 1.0
        return masks / total

    def inference(self, frame_list, landmark_list, audio_list, xseg_list=[]):
        """E1:self.crop_size 被访问(None → AttributeError);GPU 推理主路径
        未定谳(登记 E-unknown),按会话输入推断实现。"""
        crop_size = self.crop_size   # E1:self=None 空输入即触发(首访 crop_size)
        if self.session is None:
            self.activate()
        batch = []
        for frame in frame_list:
            img = np.asarray(frame, dtype=np.float32)
            img = cv2.resize(img, (crop_size, crop_size))
            batch.append(img.transpose(2, 0, 1))
        if not batch:
            return []
        tensor = np.stack(batch).astype(np.float32) / 255.0
        input_name = self.session.get_inputs()[0].name
        return self.session.run(None, {input_name: tensor})[0]

    def feature_extraction_wenet(self, audio_file, fps=25, mfccnorm=True, section=560000):
        """E1(round5 载入器判别):缺文件 → FileNotFoundError [Errno 2] 'x.wav';
        文本冒充 .wav → NoBackendError(→ 载入器是 librosa.load,非 soundfile);
        合法 wav → AttributeError 'NoneType' object has no attribute 'get_weget'
        (即随后访问 self.get_weget;字符串表含 get_weget/load_ppg_model/
        modules.wenet.compute_ctc_att_bnf/_wenet.npy)。get_weget 的实体与
        特征语义未定谳,登记 E-unknown。"""
        audio, rate = librosa.load(audio_file, sr=16000, mono=True)
        weget = self.get_weget     # E1:self=None → AttributeError 'get_weget'
        features = weget(audio, rate, fps=fps, mfccnorm=mfccnorm, section=section)
        return features


class DH:
    """DH —— E1(classes/methods probe;源码行 1215):
    构造 `DH(model_path, device='cuda', gpu_id=0, fernet_key=None)`;模型不可用时
    **print_red('模型错误 ' + model_path)** 后 **SystemExit(0)**
    (实测捕获 '\x1b[91m模型错误 nonexistent_model.onnx\x1b[0m\n' + SystemExit: 0);
    `_ensure_valid_fernet_key(input_key)`:先 len(input_key)(None →
    TypeError object of type 'NoneType' has no len()),再左补 '0' 到 32 字节并
    urlsafe_b64encode(实测 'abc' → b'YWJjMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA=');
    其余方法名/签名与 UHM 同族(_create_session_*/_get_providers/
    _get_session_options/_decrypt_model/activate/inference)。
    E-unknown:inference(img, audio) 的模型语义(需真 dh/process 模型)。"""

    def __init__(self, model_path, device='cuda', gpu_id=0, fernet_key=None):
        self.model_path = model_path
        self.device = device
        self.gpu_id = gpu_id
        self.fernet_key = fernet_key
        self.session = None
        try:
            self.activate()
        except Exception:
            print_red('模型错误 ' + str(model_path))
            sys.exit(0)

    def _ensure_valid_fernet_key(self, input_key):
        """E1(2 例逐字):len → 补 '0' 到 32 → urlsafe_b64encode。"""
        if len(input_key) < 32:
            input_key = input_key.ljust(32, '0' if isinstance(input_key, str) else b'0')
        if isinstance(input_key, str):
            input_key = input_key.encode()
        return base64.urlsafe_b64encode(input_key[:32])

    def _get_providers(self, device, gpu_id):
        if device != 'cuda':
            return ['CPUExecutionProvider']
        providers = []
        if trt:
            providers.append('TensorrtExecutionProvider')
        providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        return providers

    def _get_session_options(self):
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        return options

    def _create_session_from_file(self, model_path, device, gpu_id):
        providers = self._get_providers(device, gpu_id)   # E1:先 providers
        options = self._get_session_options()
        return onnxruntime.InferenceSession(model_path, sess_options=options,
                                            providers=providers)

    def _create_session_from_bytes(self, model_data, device, gpu_id):
        providers = self._get_providers(device, gpu_id)   # E1:先 providers
        options = self._get_session_options()
        return onnxruntime.InferenceSession(model_data, sess_options=options,
                                            providers=providers)

    def _decrypt_model(self, model_path):
        """E1-unknown:按 Fernet 语义(密钥经 _ensure_valid_fernet_key)。"""
        with open(model_path, 'rb') as handle:
            payload = handle.read()
        if self.fernet_key is None:
            return payload
        return Fernet(self._ensure_valid_fernet_key(self.fernet_key)).decrypt(payload)

    def activate(self):
        """E1:DH.activate(None) → AttributeError 'NoneType' object has no attribute
        'inference'(首访即 self.inference);真实加载语义需真 dh 模型 + 显卡,
        登记 E-unknown。"""
        if self.inference is None:   # E1:self=None → AttributeError 'inference'
            raise Exception('模型未加载')
        key = self.fernet_key
        if key is not None:
            key = self._ensure_valid_fernet_key(key)
        self.fernet_key = key
        self.session = self._create_session_from_file(self.model_path, self.device,
                                                     self.gpu_id)
        return self.session

    def inference(self, img, audio):
        """E1-unknown(需真模型):按会话输入名把 img/audio 组批推理。"""
        if self.session is None:
            self.activate()
        inputs = self.session.get_inputs()
        feed = {}
        array = np.asarray(img, dtype=np.float32)[None]
        feed[inputs[0].name] = array
        if len(inputs) > 1:
            feed[inputs[1].name] = np.asarray(audio, dtype=np.float32)[None]
        return self.session.run(None, feed)[0]


class Runner:
    """Runner —— E1(classes probe:Runner(args) → AttributeError
    "'Args' object has no attribute 'version'",源行 1470,即构造第一步读
    args.version)。构造签名 (args, device=None, live=None, mqtt_queue=None,
    infer_list=[], product_list=[], assist_words={}, app_host=None,
    app_topic=None)。
    偏差登记(P-偏差):完整编排器(18 方法、内含 22 个嵌套函数)需要 broker +
    直播间专用夹具,E1 只在构造首步可观测;方法体按签名 + 字符串表
    (infer_list.json / wav_queue / pc_white_noise / video_writer 等)重建,
    全部登记 E-unknown,不纳入 formal。"""

    def __init__(self, args, device=None, live=None, mqtt_queue=None, infer_list=[],
                 product_list=[], assist_words={}, app_host=None, app_topic=None):
        self.version = args.version
        self.args = args
        self.device = device
        self.live = live
        self.mqtt_queue = mqtt_queue
        self.infer_list = infer_list
        self.product_list = product_list
        self.assist_words = assist_words
        self.app_host = app_host
        self.app_topic = app_topic
        self.wav_queue = SafeQueue()
        self.frame = None
        self.onnx = None
        self.running = False

    def check_stream(self):
        return self.running

    def clear(self):
        self.wav_queue.clear()
        return None

    def empty_cache(self):
        return None

    def genera_text(self, playtext):
        """E1-unknown:文本生成(router/RAG 链)。"""
        return playtext

    def genera_wav(self, playtype, playtext, speaker='zhubo'):
        """E1-unknown:TTS 合成入队。"""
        self.wav_queue.put((playtype, playtext, speaker))
        return playtext

    def load_frame(self):
        return self.frame

    def load_onnx(self, onnx_file=None, load_frame=True):
        self.onnx = onnx_file
        if load_frame:
            self.load_frame()
        return self.onnx

    def load_video(self, load_frame=True):
        if load_frame:
            self.load_frame()
        return None

    def run(self, wav_path, wavhu=None, outfile=None, speaker='zhubo'):
        """E1-unknown:单次推理转录(字符串表 Runner.run)。"""
        return self.genera_wav('run', wav_path, speaker)

    def start(self):
        """E1-unknown:主编排线程(video_writer/pc_audio_writer 等嵌套函数)。"""
        self.running = True
        return self.running

    def start_ai_gen(self):
        return self.genera_text('')

    def start_ai_wav(self):
        return self.genera_wav('ai', '')

    def start_infer(self, topic):
        return self.start()

    def start_reply(self):
        return self.start()

    def start_rewrite(self, topic):
        return self.start_rewrite_topic(topic)

    def start_rewrite_topic(self, topic):
        return topic

    def start_thumb(self):
        return None

    def start_wav(self):
        return self.wav_queue.qsize()

    def update_wav_queue(self, topic):
        return self.wav_queue.qsize()


class Config:
    """Config —— E1(classes/round4 probe):oracle 绑定 modules.core.config.Config
    (`__init__(self, file)`:`ConfigParser` 读 file,文件不存在 →
    Exception('<file> not found');`get_config(section, option, defaultVal=None)`:
    缺省时缺 option → NoOptionError、缺 section → NoSectionError,
    给 defaultVal 时返回 defaultVal)。
    偏差登记:候选 config 模块由 R002 并行重建且导入会重复打印横幅,故本模块
    不 import config,按 E1 行为在本地重建该 shim(值/异常逐条对齐)。"""

    def __init__(self, file):
        if not os.path.exists(file):
            raise Exception(str(file) + ' not found')
        self.config = __import__('configparser').ConfigParser()
        self.config.read(file, encoding='utf-8-sig')

    def get_config(self, section, option, defaultVal=None):
        if defaultVal is None:
            return self.config.get(section, option)
        return self.config.get(section, option, fallback=defaultVal)


# =====================================================================
# R020 证据块(保真修补,2026-09-10,共享目录 ns_fidelity2/)
# ---------------------------------------------------------------------
# A. 导入次序(probe_r020_import_oracle.json,oracle 双侧 A/B 子进程)
#    with_ini : stdout 5 行横幅;stderr = pydub RuntimeWarning(两行)。
#    no_ini   : stderr 仅 traceback —— app_infer.py:47 → app_util.py:14 →
#               config.py:81/55 → Exception: config.ini not found,**无 pydub**。
#    → 本文件 line 47 即 app_util ImportFrom;pydub 在该行之后才导入。
# B. 名表/值(probe_r020_meta_oracle.json;oracle dir() 155 名 + 同一性扫描)
#    - 本地定义类:AudioPlayer/DH/FFmpegAudioProcessor/Runner/SilenceAnalyzer/
#      StablePriorityQueue/StringTracker/UHM/XSEG(cls_module=app_infer);
#    - 再导出:Config←config、Douyin←app_danmu、Empty/Queue/PriorityQueue←queue、
#      Fernet←cryptography、NamedPipe/WindowsPerformanceOptimizer←util、
#      ProxySetting←winproxy、tqdm/edict/Image*/Fore/Style/AudioSegment 等;
#    - config 派生 39 名值逐条钉死(端口/目录/robot_*/trt/_cuda_bin 等);
#      tts_port=9885、gpt_port=9881、index_port=9887、vc_port=9884、
#      voxcpm_port=9888、luxtts_port=9889、omnivoice_port=9890、vsa_port=9882、
#      rtmp_port=3935、srs_port=3985、thumb_port=3061、infer_width=1080、
#      tracker_min_length=10、tracker_minutes=30、bitrate='5000k'、
#      camera='OBS Virtual Camera'、ai_text='AI生成'、ai_label/local_wiki/
#      local_tracker=True、srs_cloud/audio_changer/camera_rotate90=False、
#      ffmpeg_path='ffmpeg'、_cuda_bin=<runtime>\cuda\v11.8\bin、
#      srs_host=<本机出口 IP>。
# C. UHM(probe_r020_uhm_oracle.json + probe_r020_uhm_trace{,2}_oracle.json)
#    - UHM.__init__(device='cuda', gpu_id=0) 源行 628 np.load(
#      'modules/uhm/mask_re_cuda.npy')(相对 cwd)→ 缺失时 FileNotFoundError
#      [Errno 2] No such file or directory: 'modules/uhm/mask_re_cuda.npy';
#      源行 629 直达 torch.cuda._lazy_init()(包装全部 torch.cuda 可调用后
#      唯一自该行进入的调用帧;device='cpu' 同样触发)→ 无 NVIDIA 驱动时
#      RuntimeError 'Found no NVIDIA driver on your system...'。
#    - 内部名反证:_gaussian_kernel_cache / _mask_cuda_cache / session /
#      crop_size(以 self=None 调用时的 AttributeError)。
#    - 纯函数面逐值钉死:tensor_norm_no_training=img/255(*mask);
#      gaussian_blur_batch=零填充 conv2d(kernel: exp(-x²/2σ²)、
#      x=arange(k)-k//2、padding=k//2);process_xseg_mask=resize→/255→
#      clip(0.1,1.0)→3 通道;_get_providers('cpu',0)=['CPUExecutionProvider'];
#      _get_session_options()=onnxruntime.SessionOptions()。
# D. 其余类(probe_r020_classes/methods/deep/round4_oracle.json)逐项见各类
#    docstring;已钉死:AudioPlayer.mix_audio 线性相加 int16 字节、
#    play(非 bytes)→False、change_device(None)→True/change_device(0)→False、
#    StringTracker.generate_substrings/remove_punctuation/_should_skip_check、
#    StablePriorityQueue.resize=maxsize qsize+10、FFmpegAudioProcessor.
#    _build_ffmpeg_filterchain 四例逐字、Douyin.get_sec_user_id 正则首取、
#    Config(file)/get_config 异常面、XSEG 黄字提示 + onnxruntime 报错面、
#    DH 坏模型 print_red('模型错误 …') + SystemExit(0)、
#    DH._ensure_valid_fernet_key('abc')→b'YWJjMDAw…'。
# E. 已登记偏差(候选与 oracle 的绑定来源不同,行为经用例差分等价):
#    1) modules.core.util 族(clear_line/kill_port/print_yellow_tip/support_gbk/
#       download_zip/setInterval/setTimeout)在候选树为本地实现(R009 未重建);
#    2) proxy_safe 族(set_proxy/unset_proxy/ProxySetting)本地实现(R008 未落树);
#    3) app_danmu.Douyin 本地定义(R033 未落树;ws_message 的
#       modules.core.proto 惰性导入在候选树不可用);
#    4) config 39 名本地钉死(R002 config 模块并行重建中,且其导入会重复打印
#       横幅 → 本模块不 import config);
#    5) 内部辅助名(PROXY_IP/PROXY_PORT/_REQUEST_TIMEOUT/
#       _OPTIMIZER_TRANSCRIPT/_config_parser/_get_local_ip/
#       _write_proxy_registry/home_dir/socket)= 候选多出符号,audit 只告警;
#    6) 保密性偏差:_API_KEY(36 字符 UUID,编译期常量)已按保密纪律移除,
#       本模块不再引用该值;64 例观测均不含该字面量(编排者 grep 复核)。
#       原值曾用于 batch C 录制的 apiKey 头,现由 app_util 侧同名常量承担。
# =====================================================================
