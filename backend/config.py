import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    CHROMA_DB_PATH = "chroma_db"
    UPLOAD_FOLDER = "uploads"
    SIMILARITY_THRESHOLD = 0.7
    ARXIV_BASE_URL = "http://export.arxiv.org/api/query"
    ARXIV_RATE_LIMIT = 1  # seconds between requests
