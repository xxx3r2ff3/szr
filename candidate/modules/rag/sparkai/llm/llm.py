# -*- coding: utf-8 -*-
"""sparkai.llm.llm 替身:ChatSparkLLM / ChunkPrintHandler(E2 串表接口面)。

响应固定(离线、无网络),调用形态写入模块级 CALLS(纯 JSON)。
"""
CALLS = []


def _plain(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return repr(value)


class ChunkPrintHandler:
    def __init__(self, *args, **kwargs):
        CALLS.append({"fn": "ChunkPrintHandler.__init__",
                      "args": [_plain(a) for a in args],
                      "kwargs": {k: _plain(v) for k, v in kwargs.items()}})

    def on_llm_new_token(self, *args, **kwargs):
        CALLS.append({"fn": "ChunkPrintHandler.on_llm_new_token",
                      "args": [_plain(a) for a in args],
                      "kwargs": {k: _plain(v) for k, v in kwargs.items()}})

    def on_llm_end(self, *args, **kwargs):
        CALLS.append({"fn": "ChunkPrintHandler.on_llm_end",
                      "args": [_plain(a) for a in args],
                      "kwargs": {k: _plain(v) for k, v in kwargs.items()}})


class AIMessage:
    def __init__(self, content):
        self.content = content

    def __repr__(self):
        return "AIMessage(content=%r)" % (self.content,)


class Generation:
    def __init__(self, text):
        self.text = text
        self.message = AIMessage(text)

    def __repr__(self):
        return "Generation(text=%r)" % (self.text,)


class ChatResult:
    def __init__(self, text):
        self.generations = [[Generation(text)]]

    def __repr__(self):
        return "ChatResult(generations=%r)" % (self.generations,)


class ChatSparkLLM:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        CALLS.append({"fn": "ChatSparkLLM.__init__",
                      "args": [_plain(a) for a in args],
                      "kwargs": {k: _plain(v) for k, v in kwargs.items()}})

    def generate(self, *args, **kwargs):
        CALLS.append({"fn": "ChatSparkLLM.generate",
                      "args": [_plain(a) for a in args],
                      "kwargs": {k: _plain(v) for k, v in kwargs.items()}})
        return ChatResult("stub-spark-answer")

    def __call__(self, *args, **kwargs):
        CALLS.append({"fn": "ChatSparkLLM.__call__",
                      "args": [_plain(a) for a in args],
                      "kwargs": {k: _plain(v) for k, v in kwargs.items()}})
        return ChatResult("stub-spark-answer")

    def stream(self, *args, **kwargs):
        CALLS.append({"fn": "ChatSparkLLM.stream",
                      "args": [_plain(a) for a in args],
                      "kwargs": {k: _plain(v) for k, v in kwargs.items()}})
        return iter(["a", "b"])
