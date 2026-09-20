# -*- coding: utf-8 -*-
"""sparkai.core.messages 替身:仅 ChatMessage(E2 串表接口面)。

调用记录写入模块级 CALLS(纯 JSON 可编码),供差分用例观测两侧调用形态。
"""
CALLS = []


class ChatMessage:
    def __init__(self, role=None, content=None, **kwargs):
        self.role = role
        self.content = content
        self.extra = kwargs
        CALLS.append({"fn": "ChatMessage.__init__",
                      "args": [repr(role), repr(content)],
                      "kwargs": {k: repr(v) for k, v in kwargs.items()}})

    def __repr__(self):
        return "ChatMessage(role=%r, content=%r)" % (self.role, self.content)
