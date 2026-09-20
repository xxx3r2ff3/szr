# -*- coding: utf-8 -*-
"""
@File    :   Embeddings.py
@Time    :   2024/02/10 21:55:39
@Author  :   不要葱姜蒜
@Version :   1.0
@Desc    :   None
"""
import os
from typing import List

import numpy as np
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer


class BaseEmbeddings:
    """
    Base class for embeddings
    """

    def __init__(self, path: 'unicode', is_api: bool) -> None:
        # 出处:常量表 constants.txt:372 "Argument '%.200s' has incorrect type
        # (expected %.200s, got %.200s)" + E1 实测 probe_emb_deep4(Base_path_int
        # /…_str_sub:str 子类同样被拒 ⇒ 精确类型判定)。Cython 在绑定期校验
        # `path: 'unicode'`;纯 Python 侧在体内复刻同一异常类型与文案。
        if path is not None and type(path) is not str:
            raise TypeError("Argument 'path' has incorrect type (expected str, got %s)"
                            % type(path).__name__)
        self.path = path
        self.is_api = is_api

    def get_embedding(self, text: 'unicode', model: 'unicode') -> List[float]:
        raise NotImplementedError

    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> 'float':
        """
        calculate cosine similarity between two vectors
        """
        # 出处:E1 probe_emb_deep3_out.json "cos_compare"(17 组输入与
        # dot/(norm*norm)+`if not magnitude: return 0` 三种变体逐一等价:
        # 零模长返回 int 0,其余返回 np.float64;长度不匹配由 np.dot 抛 ValueError)。
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude


class OllamaEmbedding(BaseEmbeddings):
    """
    class for OpenAI embeddings
    """

    def __init__(self, path: 'unicode' = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        # 出处:E1 probe_emb_deep2/3(Ollama_2args_true_stub 记录
        # ollama.Client.__init__ args=() kwargs={}:path 未传给 Client;
        # 无 ollama 包时此处 ModuleNotFoundError,消息见 golden emb__ollama_unreachable)。
        if is_api:
            from ollama import Client
            self.client = Client()

    def get_embedding(self, text: 'unicode',
                      model: 'unicode' = 'quentinz/bge-large-zh-v1.5:latest') -> List[float]:
        # 出处:constants.txt:490 默认模型名;E1 probe_emb_deep3(stub_ollama_get:
        # client.embeddings(model=…, prompt=text) → response['embedding'];is_api=False
        # 时 raise NotImplementedError,消息见 constants.txt:514)。
        # [I003-E2E 差分修正 2026-09-11] constants.txt 串表本函数名下紧跟 'replace'
        # 槽位(候选原缺);E2E 实测(e2e i003_rag_chain/chain_full,ns_ib 双侧):
        # oracle 传给 client.embeddings 的 prompt 将 text 中 '\n' 替换为 ' '
        # ("…1号\n…2号\n" → "…1号 …2号 "),候选原样直传 → 向量与检索度量全链偏差。
        if self.is_api:
            response = self.client.embeddings(model=model,
                                              prompt=text.replace("\n", " "))
            return response['embedding']
        raise NotImplementedError('Local embedding generation not implemented')


class LocalEmbedding(BaseEmbeddings):
    """
    class for OpenAI embeddings
    """

    def __init__(self, path: 'unicode' = '', is_api: bool = True,
                 device: 'unicode' = 'cuda') -> None:
        super().__init__(path, is_api)
        if device is not None and type(device) is not str:
            raise TypeError("Argument 'device' has incorrect type (expected str, got %s)"
                            % type(device).__name__)
        # 出处:E1 probe_emb_deep3(stub_local_true:snapshot_download(
        # 'AI-ModelScope/bge-large-zh-v1.5', revision='master',
        # cache_dir=os.path.dirname(__file__), local_files_only=True) →
        # SentenceTransformer(model_dir, device=device);路径出自常量表
        # constants.txt:471 AI-ModelScope/bge-large-zh-v1.5、431 master、429 cache_dir)。
        if is_api:
            model_dir = snapshot_download('AI-ModelScope/bge-large-zh-v1.5', revision='master',
                                          cache_dir=os.path.dirname(__file__),
                                          local_files_only=True)
            self.model = SentenceTransformer(model_dir, device=device)

    def get_embedding(self, text: 'unicode',
                      model: 'unicode' = 'quentinz/bge-large-zh-v1.5:latest') -> List[float]:
        # 出处:E1 probe_emb_deep3(stub_local_get:model.encode(text,
        # normalize_embeddings=True);real_local_false_get:NotImplementedError)。
        if self.is_api:
            return self.model.encode(text, normalize_embeddings=True)
        raise NotImplementedError('Local embedding generation not implemented')


class BgeEmbedding(BaseEmbeddings):
    """
    class for BGE embeddings
    """

    def __init__(self, path: 'unicode' = 'BAAI/bge-base-zh-v1.5',
                 is_api: bool = False) -> None:
        super().__init__(path, is_api)
        # 出处:E1 probe_emb_deep(Bge_2arg_true_stub / Bge_2arg_false_stub 调用序列
        # 完全相同 ⇒ 与 is_api 无关,恒走 load_model;默认路径见 constants.txt:486)。
        self.load_model(path)

    def load_model(self, path: 'unicode'):
        # 出处:E1 probe_emb_deep3(stub_bge_ctor:torch.cuda.is_available →
        # torch.device('cpu') → AutoTokenizer.from_pretrained(path) →
        # AutoModel.from_pretrained(path) → model.to(device) → model.eval();
        # 模型缺失时异常来自 transformers,见 golden emb__bge_model_missing)。
        import torch
        from transformers import AutoModel, AutoTokenizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModel.from_pretrained(path)
        self._model.to(device)
        self._model.eval()

    def get_embedding(self, text: 'unicode') -> List[float]:
        # 出处:E1 probe_emb_deep3/4(stub_bge_get 调用序列:tokenizer([text],
        # padding=True, truncation=True, return_tensors='pt') →
        # {k: v.to(self._model.device)} → torch.no_grad() → model(**encoded_input)
        # → model_output[0][:, 0] → torch.nn.functional.normalize(p=2, dim=1)
        # → sentence_embeddings[0].tolist() 外包 np.array,见 constants.txt:463/494/516/519)。
        import torch
        encoded_input = self._tokenizer([text], padding=True, truncation=True,
                                        return_tensors='pt')
        encoded_input = {key: value.to(self._model.device)
                         for key, value in encoded_input.items()}
        with torch.no_grad():
            model_output = self._model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return np.array(sentence_embeddings[0].tolist())


class Bgem3Embedding(BaseEmbeddings):
    """
    class for local embeddings using bge-m3
    """

    def __init__(self, path: 'unicode' = '', is_api: bool = True,
                 device: 'unicode' = 'cuda') -> None:
        super().__init__(path, is_api)
        if device is not None and type(device) is not str:
            raise TypeError("Argument 'device' has incorrect type (expected str, got %s)"
                            % type(device).__name__)
        # 出处:E1 probe_emb_deep3(stub_bgem3_true:snapshot_download('BAAI/bge-m3',
        # revision='master', cache_dir=os.path.dirname(__file__),
        # local_files_only=True) → SentenceTransformer(model_dir, device=device);
        # is_api=False 时无任何副作用,实例属性仅 path/is_api)。
        if is_api:
            model_dir = snapshot_download('BAAI/bge-m3', revision='master',
                                          cache_dir=os.path.dirname(__file__),
                                          local_files_only=True)
            self.model = SentenceTransformer(model_dir, device=device)

    def get_embedding(self, text: 'unicode') -> List[float]:
        # 出处:E1 probe_emb_deep3(stub_bgem3_get:model.encode(text,
        # normalize_embeddings=True);real_bgem3_false_get:NotImplementedError)。
        if self.is_api:
            return self.model.encode(text, normalize_embeddings=True)
        raise NotImplementedError('Local embedding generation not implemented')
