from core.parser import PaperParser
from core.chunker import PaperChunker
from core.retriever import PaperRetriever
from core.generator import ChatGenerator
from core.embedder import ZhipuEmbedder
from zai import ZhipuAiClient
import chromadb

# ================= 1. 初始化基础配置 =================
from config import (
    ZHIPU_API_KEY,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    LLM_MODEL,
    LLM_TEMPERATURE,
)



pdf_path = "C:/Users/admin/Desktop/PAPER/文献/Dual Attribute Adversarial Camouflage toward camouflaged object.pdf"

# client = ZhipuAiClient(api_key=ZHIPU_API_KEY)
embedder = ZhipuEmbedder(api_key=ZHIPU_API_KEY)
retriever = PaperRetriever(embedder)
generator = ChatGenerator()

# chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
# collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# ================= 2. 文档解析与向量建库 =================
print("正在解析论文并构建向量库...")
parser = PaperParser(pdf_path)
paper_content = parser.extract_text()



chunker = PaperChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
structured_paper = chunker.split_by_section(paper_content)

documents = []
metadatas = []
ids = []


# 遍历结构化章节并组装 ChromaDB 需要的数据格式
for key, value in structured_paper.items():
    if isinstance(value, list):
        for i, chunk in enumerate(value):
            documents.append(chunk)
            metadatas.append({"section": key}) # 核心：打上章节标签
            ids.append(f"{key}_{i}")

# 计算所有 Chunk 的向量并存入数据库
count = retriever.build_index(documents, metadatas, ids)
print(f"建库完成，共存入 {count} 个 Chunk")
# ================= 3. 提问与回答 =================


question = "DAAC生成过程中的步骤具体是如何实现的？"


expanded_query = generator.rewrite_query(question)
print(f"改写后的查询: {expanded_query}")

chunks = retriever.search(expanded_query)
answer = generator.generate_answer(question, chunks)

print("\n=== 最终回答 ===")
print(answer)



