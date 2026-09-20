# -*- coding: utf-8 -*-
"""
@File    :   Reranker.py
@Time    :   2024/05/15 14:27:42
@Author  :   YueZhengMeng
@Version :   1.0
@Desc    :   None
"""
# ---------------------------------------------------------------------------
# [R041 重建证据] 源 modules/rag/Reranker.cp310-win_amd64.pyd(T4R)。
#   E1 行锚:BaseReranker.rerank py=24;BgeReranker.__init__ py=32;rerank py=36
#   (with no_grad py=40;self._model.device py=48;index[-k] py=56;return py=57);
#   load_model py=59。
#   E2 串表:AutoTokenizer / AutoModelForSequenceClassification / _model /
#   _tokenizer / from_pretrained / eval / padding / truncation / return_tensors /
#   max_length / return_dict / logits / argsort / tolist / no_grad / cuda /
#   is_available / BAAI/bge-reranker-base。
#   E1 实测:probe_beh5/6/7/14_out.json(补丁 transformers.AutoTokenizer/
#   AutoModelForSequenceClassification 后逐调用记录)。
# ---------------------------------------------------------------------------
from typing import List

import numpy as np


class BaseReranker:
    """
    Base class for reranker
    """

    def __init__(self, path: str):
        self.path = path

    def rerank(self, text: str, content: List[str], k: int) -> List[str]:
        raise NotImplementedError


class BgeReranker(BaseReranker):
    """
    class for Bge reranker
    """

    def __init__(self, path: str = 'BAAI/bge-reranker-base'):
        super().__init__(path)
        self._model, self._tokenizer = self.load_model(path)

    def rerank(self, text: str, content: List[str], k: int) -> List[str]:
        # [E1] with torch.no_grad();pairs 为 [(text, c)] 元组列表;tokenizer 关键字
        # padding/truncation/return_tensors='pt'/max_length=512;输入的每个张量
        # .to(self._model.device);模型关键字 **inputs + return_dict=True;
        # logits.view(-1).float();升序 argsort 后取 index[-k:] 再逆序输出
        # (k=0 → index[-0:] 即全量;-k 为负向切片;None/float 分别抛
        # TypeError: unary - / slice indices 实测)
        import torch

        with torch.no_grad():
            pairs = [(text, c) for c in content]
            inputs = self._tokenizer(pairs, padding=True, truncation=True,
                                     return_tensors='pt', max_length=512)
            inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
            scores = self._model(**inputs, return_dict=True).logits.view(-1).float()
            index = scores.argsort().tolist()
            index = index[-k:]
        return [content[i] for i in index][::-1]

    def load_model(self, path: str):
        # [E1] 调用次序:tokenizer.from_pretrained → model.from_pretrained →
        # model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) →
        # model.eval();返回 (model, tokenizer),由 __init__ 解包到 _model/_tokenizer
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        return model, tokenizer
