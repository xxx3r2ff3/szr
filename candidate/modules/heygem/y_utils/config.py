# -*- coding: utf-8 -*-
# 出处:candidate/modules/heygem/y_utils/config.py ← y_utils/config.pyc
# 反汇编:evidence/modules/_heygem_orphans/disasm/y_utils_config.pyc.dis(py3.8)
# 结构/签名按 .dis 对齐:
#   模块级 get_config()                                —— 0 参
#   GeneralConfig.__init__(self, config_path, section) —— 3
#   GeneralConfig.__getattr__(self, item)              —— 2
#   GeneralConfig.__get_option(self, option, section='app') —— 3(defaults=('app',))
#   GlobalConfig.__init__(self)                        —— 1(逐项 get_config 赋属性)
#   GlobalConfig.instance(cls, *args, **kwargs)        —— @classmethod,3(含 *args/**kwargs)
#   GlobalConfig.get_config(self, section, key, default_value) —— 4
#   模块级 base_dir = './'
# 语义重建说明:各函数体按 .dis 控制流逐条还原(含 try/except Exception 的
#   __get_option、单例 instance、GlobalConfig.__init__ 的 30 项配置读取)。
# 未还原细节:无。
"""
File: config.py
Author: YuFangHui
Date: 2020-11-25
Description:
"""
import configparser
import os

from y_utils import config
from y_utils import tools

base_dir = './'


def get_config():
    config = configparser.ConfigParser()
    config.read(os.path.join(base_dir, 'config/config.ini'), encoding='utf-8')
    return config


class GeneralConfig:

    def __init__(self, config_path, section):
        self.conf = configparser.ConfigParser()
        self.conf.read(config_path)
        self.section = section
        print(self.conf.__dict__)

    # 下面是类内私有方法 __get_option;CPython 名字改写后类体 STORE_NAME 的键
    # 即 .dis 中的 _GeneralConfig__get_option,调用点同理。
    def __getattr__(self, item):
        return self.__get_option(item, self.section)

    def __get_option(self, option, section='app'):
        try:
            cfg_val = self.conf.get(section, option)
            return cfg_val
        except Exception as e:
            print(e)
            return None


class GlobalConfig:

    def __init__(self):
        self.server_ip = self.get_config('http_server', 'server_ip', '0.0.0.0')
        self.server_port = self.get_config('http_server', 'server_port', 8383)
        self.temp_dir = self.get_config('temp', 'temp_dir', os.path.join(base_dir, 'temp'))
        self.temp_clean_switch = self.get_config('temp', 'clean_switch', 0)
        self.result_dir = self.get_config('result', 'result_dir', os.path.join(base_dir, 'result'))
        self.result_clean_switch = self.get_config('result', 'clean_switch', 0)
        self.access_key_id = self.get_config('obs', 'access_key_id', None)
        self.secret_access_key = self.get_config('obs', 'secret_access_key', None)
        self.obs_server = self.get_config('obs', 'obs_server', None)
        self.bucket = self.get_config('obs', 'bucket', None)
        self.obs_dir = self.get_config('obs', 'obs_dir', None)
        self.batch_size = self.get_config('digital', 'batch_size', 8)
        self.watermark_path = self.get_config('watermark', 'watermark_path', None)
        self.digital_auth_path = self.get_config(
            'digital_human_authentication', 'digital_auth_path', None)
        self.model_version = self.get_config('model', 'model_version', '256v1')
        self.blend_dynamic = self.get_config('model', 'blend_dynamic', 'lmk')
        self.chaofen_before = self.get_config('model', 'chaofen_before', 1)
        self.chaofen_after = self.get_config('model', 'chaofen_after', 0)
        self.blur_threshold = float(self.get_config('model', 'blur_threshold', 0.3))
        self.register_url = self.get_config('register', 'url', '')
        self.register_file = self.get_config('register', 'file', '/code/data/result/.reg')
        self.register_report_interval = int(self.get_config('register', 'report_interval', 3600))
        self.register_enable = int(self.get_config('register', 'enable', 1))

    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(GlobalConfig, '_instance'):
            GlobalConfig._instance = GlobalConfig(*args, **kwargs)
        return GlobalConfig._instance

    def get_config(self, section, key, default_value):
        if config.get_config().has_option(section, key):
            return config.get_config().get(section, key)
        return default_value
