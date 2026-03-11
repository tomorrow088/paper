from core.parser import PaperParser
from core.chunker import PaperChunker
from core.retriever import PaperRetriever
from core.embedder import ZhipuEmbedder
from config import ZHIPU_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP

print("=== 1. 初始化与解析文档 ===")
# 请确保这里的路径是你电脑上真实的 PDF 路径
pdf_path = "C:/Users/admin/Desktop/PAPER/文献/Dual Attribute Adversarial Camouflage toward camouflaged object.pdf"

# 实例化嵌入模型和检索器
embedder = ZhipuEmbedder(api_key=ZHIPU_API_KEY)
retriever = PaperRetriever(embedder)

# 解析并切分 PDF
parser = PaperParser(pdf_path)
paper_content = parser.extract_text()

chunker = PaperChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
structured_paper = chunker.split_by_section(paper_content)

documents = []
metadatas = []
ids = []

# 组装数据
for key, value in structured_paper.items():
    if isinstance(value, list):
        for i, chunk in enumerate(value):
            documents.append(chunk)
            metadatas.append({"section": key})
            ids.append(f"{key}_{i}")

print("=== 2. 激活 BM25 内存并构建向量库 ===")
# 这一步非常关键：让 BM25 吃下所有文本，建立倒排索引
count = retriever.build_index(documents, metadatas, ids)
print(f"建库完成，共存入 {count} 个 Chunk！\n")

print("=== 3. 极端压力测试 ===")
# 故意构造一个包含特殊缩写（DAAC）和特定数据集名称（COD10K）的生僻问题
test_query = "论文中提到的 DAAC 方法包含哪几个阶段？"
print(f"测试问题: {test_query}\n")

# A. 纯向量检索
query_vector = embedder.get_embedding(test_query)
vector_top_1 = retriever.collection.query(query_embeddings=[query_vector], n_results=1)["documents"][0][0]

# B. 纯 BM25 检索
query_tokens = test_query.split()
bm25_top_1 = retriever.bm25.get_top_n(query_tokens, retriever.bm25_docs, n=1)[0]

# C. 混合检索 (RRF融合)
hybrid_top_1 = retriever.search(test_query, top_k=1)[0]

print(f"【纯向量检索 Top-1】:\n{vector_top_1[:150]}...\n")
print(f"【纯 BM25 检索 Top-1】:\n{bm25_top_1[:150]}...\n")
print(f"【混合检索 (RRF) Top-1】:\n{hybrid_top_1[:150]}...\n")