import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')  # For LLM service
    UPLOAD_FOLDER = "uploads"
    SCRAPED_DATA_FOLDER = "scraped_data"
    FAISS_VECTOR_DB_FOLDER = "faiss_vector_db"
    SIMILARITY_THRESHOLD = 0.7
    ARXIV_BASE_URL = "http://export.arxiv.org/api/query"
    ARXIV_RATE_LIMIT = 1  # seconds between requests
