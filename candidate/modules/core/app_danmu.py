# -*- coding: utf-8 -*-
"""modules.core.app_danmu —— T4R 语义重建(R033)。

源 pyd:modules/core/app_danmu.cp310-win_amd64.pyd
        sha256 917c95fc9b013bfc4a83a0ee480bb138d19c2ae83d17781bf624f1e966f2fae2

二进制出处(Ghidra 12.1.3 反编译,证据 runtime/ghidra/):
  * get_user_unique_id @0x180001090   code: argcount=0 nlocals=0
  * get_x_ms_stub      @0x1800012a0   本地 (params, sig_params, k, v)
  * build_request_url  @0x180001d40   本地 (url, parsed_url, existing_params,
                                            new_query_string, new_url)
  * load_webmssdk      @0x180002ea0   本地 (js_file, dir_path, js_path, f)
  * get_signature      @0x180003de0   本地 (x_ms_stub, jsengine, ctx, js_dom,
                                            js_enc, final_js, function_caller, signature)
  * Douyin.*           @0x180004ab0(__init__)/0x180005880(get_sec_user_id)/
                       0x180006160(get_web_url)/0x180006fa0(parse_wss_url)/
                       0x1800091e0(ws_connect)/0x18000ab30(ws_open)/
                       0x180009d50(ping)/0x18000b200(ws_message)/
                       0x18000d3d0(ws_error)/0x18000d770(ws_close)
  * 模块 exec @0x18000ec50(__main__ 块内定义 wenda 并以 do_wenda=wenda 构造 Douyin)

E1 探针(evidence/modules/core__app_danmu/runtime/):
  probeE_app_danmu.py / probeG_app_danmu_js.py。

行为要点(探针实测):
  * USER_AGENT / headers 为模块级常量(probeE 逐字节取值,见下)。
  * get_user_unique_id() = str(random.randint(7300000000000000000,
    7999999999999999999)):种子 20260910/1/2147483647 的三连值逐条吻合,
    上下界字面量亦见于 pyd 串表。
  * get_x_ms_stub(params) = hashlib.md5(",".join(f"{k}={v}").encode()).hexdigest()
    ({} → d41d8cd98f00b204e9800998ecf8427e = md5(""))。
  * build_request_url(url):urlparse → parse_qs → 追加 aid/device_platform/
    browser_language/browser_platform/browser_name(=UA.split("/")[0])/
    browser_version(=UA 去掉 "Mozilla/")→ urlencode(doseq=True) → urlunparse。
  * load_webmssdk(js_file):os.path.dirname(os.path.realpath(__file__)) + js_file,
    open(..., "r", encoding="utf-8").read() → 返回 JS 源串。
  * get_signature(x_ms_stub):函数内 `import jsengine` → jsengine.jsengine() 建 ctx →
    ctx.eval(prelude.strip() + webmssdk.js 源) → ctx.eval(f"get_sign('{x_ms_stub}')");
    异常时 raise Exception("get_signature error")。返回值每次不同(JS 侧随机),
    仅长度 16 稳定(probeG signature_shape)。
  * parse_qs/urlencode/urlparse/urlunparse 是 urllib.parse 的原样再导出(同一性 True)。

未定谳(见 reports/modules/core__app_danmu-impl.md §5):get_web_url/parse_wss_url
需真实抖音端点(webcast.amemv.com / live.douyin.com),按串表 + 上游同源结构还原,
未做双侧断言。
"""
import datetime
import gzip
import hashlib
import json
import os
import random
import re
import time
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests
import websocket
import _thread
from google.protobuf import json_format

from modules.core.proto.douyin.dy_pb2 import (
    ChatMessage,
    LikeMessage,
    MemberMessage,
    PushFrame,
    Response,
)

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0'

# [COM-D002 语义替换] 原实现把**抓包得到的账号会话 Cookie**整段硬编码在默认请求头
# 里(含 passport_assist_user / sid_guard / ttwid / msToken / odin_tt /
# csrf_session_id / __ac_signature 等会话令牌,以及抓包机器的设备指纹项)。
# 该字面量已整段删除(值不入库、不入报告、不进日志)。
# 现 'cookie' 默认空串 —— 不携带任何凭据(fail-closed);真实 Cookie 由平台适配层
# 在运行时按账号从凭据保险箱注入(见 platform_adapters/common/credential_vault.py),
# 未注入即不发 Cookie,绝不回落到任何内置会话。
headers = {
    'authority': 'live.douyin.com',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'cache-control': 'max-age=0',
    'cookie': '',
    'referer': 'https://live.douyin.com/721566130345?cover_type=&enter_from_merge=web_live&enter_method=web_card&game_name=&is_recommend=&live_type=game&more_detail=&room_id=7317569386624125734&stream_type=vertical&title_type=&web_live_tab=all',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0',
}


def get_user_unique_id():
    return str(random.randint(7300000000000000000, 7999999999999999999))


def get_x_ms_stub(params):
    sig_params = ",".join([f"{k}={v}" for k, v in params.items()])
    return hashlib.md5(sig_params.encode()).hexdigest()


def build_request_url(url):
    parsed_url = urlparse(url)
    existing_params = parse_qs(parsed_url.query)
    existing_params["aid"] = "6383"
    existing_params["device_platform"] = "web"
    existing_params["browser_language"] = "zh-CN"
    existing_params["browser_platform"] = "Win32"
    existing_params["browser_name"] = USER_AGENT.split("/")[0]
    existing_params["browser_version"] = USER_AGENT.split(
        existing_params["browser_name"] + "/", 1)[1]
    new_query_string = urlencode(existing_params, doseq=True)
    new_url = urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path,
                          parsed_url.params, new_query_string, parsed_url.fragment))
    return new_url


def load_webmssdk(js_file):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    js_path = os.path.join(dir_path, js_file)
    with open(js_path, "r", encoding="utf-8") as f:
        return f.read()


def get_signature(x_ms_stub):
    try:
        import jsengine
        ctx = jsengine.jsengine()
        js_dom = f"""
document = {{}}
window = {{}}
navigator = {{
'userAgent': '{USER_AGENT}'
}}
""".strip()
        js_enc = load_webmssdk("proto/douyin/webmssdk.js")
        final_js = js_dom + js_enc
        ctx.eval(final_js)
        function_caller = f"get_sign('{x_ms_stub}')"
        signature = ctx.eval(function_caller)
        return signature
    except Exception:
        raise Exception("get_signature error")


class Douyin(object):

    def __init__(self, room_url, do_wenda):
        self.room_url = room_url
        self.ttwid = ""
        self.room_id = ""
        self.wss_url = ""
        self.nickname = ""
        self.do_wenda = do_wenda

    def get_sec_user_id(self, url):
        response = requests.get(url, headers=headers)
        redirect_url = response.url
        room_id = redirect_url.split("?")[0].rsplit("/", maxsplit=1)[-1]
        return room_id

    def get_web_url(self, url):
        room_id = self.get_sec_user_id(url)
        res = requests.get(
            url="https://webcast.amemv.com/webcast/room/reflow/info/?type_id=0"
                "&live_id=1&room_id=" + room_id + "&version_code=99.99.99&app_id=1128",
            headers=headers)
        resjson = json.loads(res.text)
        roomjson = resjson["data"]["room"]
        create_time = datetime.datetime.fromtimestamp(int(roomjson["create_time"]))
        self.room_id = room_id
        self.nickname = resjson["data"]["owner"]["nickname"]
        print(f"【{self.nickname}】直播间:{roomjson['title']} 开播时间:{create_time}")
        self.room_url = "https://live.douyin.com/" + resjson["data"]["owner"]["web_rid"]
        return self.room_url

    def parse_wss_url(self):
        res = requests.get(url=self.room_url, headers=headers)
        self.ttwid = res.cookies.get_dict()["ttwid"]
        data = res.text
        res_room_info = re.search(
            r'room\\":{.*\\"id_str\\":\\"(\d+)\\".*,\\"status\\":(\d+).*'
            r',\\"status_str\\":\\"(\d+)\\",\\"title\\":\\"([^"]*)\\".*'
            r'\,\\"nickname\\":\\"([^"]*)\\"', data)  # [ST 0x18001e1b8 n=146] `\,`=字面逗号(I-C E2E 修正)
        self.nickname = res_room_info.group(5)
        room_status = res_room_info.group(2)
        room_title = res_room_info.group(4)
        if room_status == "4":
            raise ConnectionError(room_title + ", 直播已结束")
        res_room = re.search(r'roomId\\":\\"(\d+)\\"', data)
        self.room_id = res_room.group(1)
        USER_UNIQUE_ID = get_user_unique_id()
        VERSION_CODE = 180800  # [pyd FUN_180006fa0 PyLong_FromLong(0x2c240)→SetItem("version_code") L1005/1300;I-C E2E 定谳,原 R033 误复用 webcast_sdk_version 串]
        WEBCAST_SDK_VERSION = "1.0.14-beta.0"  # [ST 0x18001db28]
        sig_params = {
            "live_id": "1",
            "aid": "6383",
            "version_code": VERSION_CODE,
            "webcast_sdk_version": WEBCAST_SDK_VERSION,
            "room_id": self.room_id,
            "sub_room_id": "",
            "sub_channel_id": "",
            "did_rule": "3",
            "user_unique_id": USER_UNIQUE_ID,
            "device_platform": "web",
            "device_type": "",
            "ac": "",
            "identity": "audience",
        }
        signature = get_signature(get_x_ms_stub(sig_params))
        webcast5_params = {
            "room_id": self.room_id,
            "compress": "gzip",
            "version_code": VERSION_CODE,
            "webcast_sdk_version": WEBCAST_SDK_VERSION,
            "live_id": "1",
            "did_rule": "3",
            "user_unique_id": USER_UNIQUE_ID,
            "identity": "audience",
            "signature": signature,
        }
        self.wss_url = ("wss://webcast5-ws-web-lf.douyin.com/webcast/im/push/v2/?"
                        + "&".join([f"{k}={v}" for k, v in webcast5_params.items()]))
        self.wss_url = build_request_url(self.wss_url)
        # [I-C E2E/probe_parsewss3 E1 定谳] pyd 此处无显式 return(调用方只用属性),
        # 保持返回 None;勿加 `return self.wss_url`(R033 结构还原时的增补,已删)。

    def ws_connect(self):
        headers_ws = {
            "cookie": "ttwid=" + self.ttwid,
            "user-agent": USER_AGENT,
        }
        print(f"ws_connect {self.room_url}")
        self.ws = websocket.WebSocketApp(
            self.wss_url,
            header=headers_ws,
            on_message=self.ws_message,
            on_error=self.ws_error,
            on_close=self.ws_close,
            on_open=self.ws_open,
        )
        self.ws.run_forever()

    def ws_open(self, ws):
        print(f"ws_open {self.room_url}")
        _thread.start_new_thread(self.ping, (ws,))

    def ping(self, ws):
        while True:
            obj = PushFrame()
            obj.payloadType = "hb"
            data = obj.SerializeToString()
            ws.send(data, websocket.ABNF.OPCODE_BINARY)
            time.sleep(5)

    def ws_message(self, ws, message):
        # [pyd FUN_18000b200 + I-C E2E/probe_wsmsg E1 定谳(R033 §5.3 未定谳项收口)]
        # 真实行为:needAck 时回 PushFrame(logId=obj.logId, payloadType=res.internalExt
        # —— "ack" 赋值被 internalExt 覆盖,payload 不设);三类消息共用同一 wenda 桥:
        # MessageToDict 后 user.nickName != self.nickname 即回调 self.do_wenda(d),
        # 无任何 print([ST] 无 '聊天msg' 系串,do_wenda/'nickName'/'method' 槽位)。
        obj = PushFrame()
        obj.ParseFromString(message)
        data = gzip.decompress(obj.payload)
        res = Response()
        res.ParseFromString(data)
        if res.needAck:
            ack = PushFrame()
            ack.payloadType = "ack"
            ack.logId = obj.logId
            ack.payloadType = res.internalExt
            ws.send(ack.SerializeToString(), websocket.ABNF.OPCODE_BINARY)
        for item in res.messagesList:
            if item.method == "WebcastChatMessage":
                msg = ChatMessage()
                msg.ParseFromString(item.payload)
                d = json_format.MessageToDict(msg, preserving_proto_field_name=True)
                if d.get("user").get("nickName") != self.nickname:
                    self.do_wenda(d)
            elif item.method == "WebcastMemberMessage":
                msg = MemberMessage()
                msg.ParseFromString(item.payload)
                d = json_format.MessageToDict(msg, preserving_proto_field_name=True)
                if d.get("user").get("nickName") != self.nickname:
                    self.do_wenda(d)
            elif item.method == "WebcastLikeMessage":
                msg = LikeMessage()
                msg.ParseFromString(item.payload)
                d = json_format.MessageToDict(msg, preserving_proto_field_name=True)
                if d.get("user").get("nickName") != self.nickname:
                    self.do_wenda(d)

    def ws_error(self, ws, error):
        print(f"ws_error {self.room_url} {error}")

    def ws_close(self, ws, code, reason=None):
        print(f"ws_close {self.room_url} {code} {reason}")


if __name__ == "__main__":
    def wenda(room_id, nickname, content):
        print(f"【问答】{nickname}: {content}")

    douyin = Douyin("https://live.douyin.com/7317569386624125734", wenda)
    douyin.parse_wss_url()
    douyin.ws_connect()
