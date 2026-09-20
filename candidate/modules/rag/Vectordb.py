import os
from typing import List
from tqdm import tqdm
from nano_vectordb import NanoVectorDB
from modules.rag.Embeddings import BaseEmbeddings
# [R042 重建证据] 源 modules/rag/Vectordb.cp310-win_amd64.pyd(T4R)。
#   E1 行锚(运行期回溯):VectorStore.__init__ py=9;save_vector 的 upsert py=31;
#   query 的 self.vdb py=38。E2 串表:Calculating embeddings / /vectors.json /
#   __vector__ / __document__ / fake_dim / data_len / storage_file / upsert / save。
#   E1 实测:ns_w2rag/probe_beh5/6/7/14_out.json。
class VectorStore:
    """"""
    def __init__(self, document: List[str] = ['']) -> None:
        self.document = document
        self.fake_dim = 1024

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        # [E1] self.vectors[i] = EmbeddingModel.get_embedding(doc)(单参调用,
        # 不带 model 名);tqdm 描述串 "Calculating embeddings";返回该列表
        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors

    def load_vector(self, path: str = 'storage'):
        # [E1] 只建目录 + 建 NanoVectorDB(不 upsert、不 load);storage_file 用
        # 正斜杠拼接(path + '/vectors.json');重复调用不报错(exists 守卫)
        if not os.path.exists(path):
            os.makedirs(path)
        self.vdb = NanoVectorDB(self.fake_dim, storage_file=path + '/vectors.json')

    def save_vector(self):
        # [E1] 一次性 upsert 全部 {'__vector__':…, '__document__':…}(键序vector在前),
        # 然后 save();无 self.vectors 时抛 AttributeError
        data_len = len(self.document)
        datas = []
        for i in range(data_len):
            datas.append({'__vector__': self.vectors[i],
                          '__document__': self.document[i]})
        self.vdb.upsert(datas)
        self.vdb.save()

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        # [E2] NanoVectorDB.query(query, top_k=…, better_than_threshold=…) 全为关键字
        query_vector = EmbeddingModel.get_embedding(query)
        # [E1 py=38] 未调用 load_vector 时 AttributeError: 无 vdb 属性
        result = self.vdb.query(query_vector, top_k=k, better_than_threshold=0.2)
        return result
