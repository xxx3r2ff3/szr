# -*- coding: utf-8 -*-
"""
@File    :   LLM.py
@Time    :   2024/02/12 13:50:47
@Author  :   不要葱姜蒜
@Version :   1.0
@Desc    :   None
"""
# ---------------------------------------------------------------------------
# [R040 重建证据] 源 modules/rag/LLM.cp310-win_amd64.pyd(T4R)。
#   E2 串表:sparkai.llm.llm / sparkai.core.messages / ChatSparkLLM /
#   ChunkPrintHandler / ChatMessage / PROMPT_TEMPLATE / RAG_PROMPT_TEMPALTE /
#   spark_api_url / spark_app_id / spark_api_key / spark_api_secret /
#   spark_llm_domain / <第三方 LLM 的 wss 端点> / <app_id> / <api_key> /
#   <api_secret> / generalv3.5。
#   [COM-D002] 串表里的端点与三项凭据值**已按要求删除**(值不入库、不入报告):
#   它们现由凭据保险箱/配置提供,缺失即显式报错(见 _spark_setting);
#   generalv3.5(服务模型名,非凭据/非主机)按原样保留。
#   E1 行锚:模块 init 首个 sparkai 导入 py=11;BaseModel.chat py=30;
#   OllamaChat.chat py=42;SparkChat.chat py=93(返回语句 py=104)。
#   E1 实测:probe_llm3/4_out.json(沙箱 stub 注入后逐调用记录)。
#   E1 注意:恢复基线的 SparkChat 在 __init__ 内用字面量,不读环境变量
#   (实测:设置 SPARKAI_* 环境变量后 kwargs 不变);[COM-D002] 语义替换后
#   改为逐项读配置,该 E1 差异已登记在 reports/D002_DEOFFICIALIZATION_CLEARANCE.md。
# ---------------------------------------------------------------------------
import os
from typing import List, Dict

from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage

# [COM-D002] 第三方 LLM 端点/凭据的来源表(环境变量名或 config.ini [LLM] 选项名)。
_SPARK_ENV = {
    'spark_api_url': 'SZR_SPARK_API_URL',
    'spark_app_id': 'SZR_SPARK_APP_ID',
    'spark_api_key': 'SZR_SPARK_API_KEY',
    'spark_api_secret': 'SZR_SPARK_API_SECRET',
}


def _spark_setting(option):
    """按 option 取端点/凭据:环境变量 → config.ini [LLM] → 缺失即硬失败。

    [COM-D002] 返回**配置提供的值**;缺失时抛显式 RuntimeError,绝不回落到任何
    内置字面量。值本身不打印、不落盘、不进报告(与 tools/secret_scan.py 同一纪律)。
    正式凭据存储接口见 platform_adapters/common/credential_vault.py;本模块只消费
    其注入的环境/配置,不直接依赖适配层。
    """
    env_name = _SPARK_ENV[option]
    value = os.environ.get(env_name, '')
    if not value:
        parser = __import__('configparser').ConfigParser()
        parser.read(os.path.join(os.getcwd(), 'config.ini'), encoding='utf-8-sig')
        value = parser.get('LLM', option, fallback='')
    if not value:
        raise RuntimeError(
            '缺少 LLM 配置 %s:请通过凭据保险箱/配置提供(环境变量 %s 或 config.ini '
            '[LLM] %s);客户端不内置任何端点或凭据。' % (option, env_name, option))
    return value


PROMPT_TEMPLATE = {
    'RAG_PROMPT_TEMPALTE': """使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。用亲切的语气直接返回结果。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:"""
}


class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        # [E1] 空实现:返回 None(history/content 均未使用)
        pass

    def load_model(self):
        pass


class OllamaChat(BaseModel):
    def __init__(self, path: str = '', model: str = 'qwen2.5') -> None:
        super().__init__(path)
        self.model = model

    def chat(self, prompt: str, history: List[Dict[str, str]], content: str) -> str:
        # [E1] 函数级 import ollama;client = ollama.Client()(无参);
        # messages 只含 1 条 user 消息(PROMPT_TEMPLATE 格式化,history 被忽略);
        # client.chat(model=self.model, messages=messages) → ['message']['content']
        import ollama

        client = ollama.Client()
        messages = [{'role': 'user',
                     'content': PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(
                         question=prompt, context=content)}]
        response = client.chat(model=self.model, messages=messages)
        msg = response['message']['content']
        return msg

    def load_model(self):
        pass


class SparkChat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        # [COM-D002 语义替换] 原为 5 个字面量常量:第三方 wss 端点 + app_id +
        # api_key + api_secret + 服务模型名。端点与三项凭据的值已删除,改为逐项
        # 从凭据保险箱/配置读取(_spark_setting);缺失即显式 RuntimeError,
        # 绝不回落到任何内置值。局部常量名 SPARKAI_* 原样保留(E1 行锚),
        # 唯一非敏感的 'generalv3.5'(服务模型名)按原样保留。
        SPARKAI_URL = _spark_setting('spark_api_url')
        SPARKAI_APP_ID = _spark_setting('spark_app_id')
        SPARKAI_API_SECRET = _spark_setting('spark_api_secret')
        SPARKAI_API_KEY = _spark_setting('spark_api_key')
        SPARKAI_DOMAIN = 'generalv3.5'
        self.spark = ChatSparkLLM(spark_api_url=SPARKAI_URL,
                                  spark_app_id=SPARKAI_APP_ID,
                                  spark_api_key=SPARKAI_API_KEY,
                                  spark_api_secret=SPARKAI_API_SECRET,
                                  spark_llm_domain=SPARKAI_DOMAIN,
                                  streaming=False)

    def chat(self, prompt: str, history: List[Dict[str, str]], content: str) -> str:
        # [E1] messages 只含 1 条 ChatMessage(role='user', content=模板格式化);
        # handler = ChunkPrintHandler();self.spark.generate(messages, callbacks=[handler]);
        # 返回 res.generations[0][0].message.content(history 被忽略)
        messages = [ChatMessage(role='user',
                                content=PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(
                                    question=prompt, context=content))]
        handler = ChunkPrintHandler()
        res = self.spark.generate(messages, callbacks=[handler])
        return res.generations[0][0].message.content

    def load_model(self):
        pass
