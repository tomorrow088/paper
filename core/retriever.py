import chromadb
from core.embedder import ZhipuEmbedder
from config import CHROMA_DB_PATH, COLLECTION_NAME, RETRIEVAL_TOP_K
from rank_bm25 import BM25Okapi


class PaperRetriever:

    def __init__(self,embedder):
        self.embedder = embedder
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME
        )
        self.bm25 = None
        self.bm25_docs = []

    def build_index(self,documents,metadatas,ids):
        embeddings = [self.embedder.get_embedding(doc) for doc in documents]
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )

        self.bm25_docs = documents
        tokenized_cropus = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_cropus)

        return len(documents)

    def search(self,query_text,top_k=None):
        if top_k is None:
            top_k = RETRIEVAL_TOP_K
        query_vector = self.embedder.get_embedding(query_text)
        vector_results = self.collection.query(
            query_embeddings= [query_vector],
            n_results=top_k
        )

        vector_top_docs = vector_results["documents"][0]# 拿到向量检索的 Top-K 文档列表

        # 2. BM25 检索
        query_tokens = query_text.split()# 把用户的问题也切碎
        bm25_top_docs = self.bm25.get_top_n(query_tokens,self.bm25_docs,n=top_k)        # 拿到 BM25 的 Top-K 文档列表

        rrf_scores = {}
        k = 60

        # 1. 遍历向量检索的榜单
        for rank, doc in enumerate(vector_top_docs):
            # rank 是排名下标（从 0 开始），doc 是具体的文本内容
            score = 1/(k+rank+1)

            # 累加分数（如果字典里还没有这个 doc，就默认初始分为 0 再加）
            rrf_scores[doc] = rrf_scores.get(doc, 0.0) + score

        for rank, doc in enumerate(bm25_top_docs):
            score = 1 / (k + rank + 1)
            rrf_scores[doc] = rrf_scores.get(doc, 0.0) + score

        # 4. 按分数从高到低排序字典
        # sorted_docs 变成了一个列表，里面的元素是元组：[("文本A", 0.033), ("文本B", 0.031), ...]
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


        return [item[0] for item in sorted_docs[:top_k]]

