import os
from dotenv import load_dotenv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "paper_nodes"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
LLM_MODEL = "glm-4.7-flash"
LLM_TEMPERATURE = 0.3
RETRIEVAL_TOP_K = 10
