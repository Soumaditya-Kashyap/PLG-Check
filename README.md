# 📄 Plagiarism Checker - Data Scraping & Storage System

## 🎯 **Project Overview**
A comprehensive academic plagiarism detection system that processes uploaded PDFs, extracts relevant research content from ArXiv and web sources, and stores them in organized formats for similarity analysis.

## 🚀 **Current Features (Data Pipeline)**
- ✅ **PDF Text Extraction** - Extract and process text from uploaded academic papers
- ✅ **LLM Keyword Generation** - Use Google Gemini AI to generate relevant search keywords
- ✅ **ArXiv Research Scraping** - Download and process relevant academic papers
- ✅ **Web Content Extraction** - Scrape and clean web content using Tavily API
- ✅ **Structured Data Storage** - Organize all content in JSON format for future processing
- ✅ **Multi-language Support** - Support for Hindi and English PDFs

## 📁 **Project Structure**
```
CheckPLG1/
├── backend/                    # Python Flask backend
│   ├── app.py                 # Main Flask application
│   ├── config.py              # Configuration settings
│   ├── requirements.txt       # Python dependencies
│   ├── services/              # Core business logic
│   │   ├── llm_service.py     # Google Gemini AI integration
│   │   ├── arxiv_service.py   # ArXiv paper scraping
│   │   ├── tavily_service.py  # Web content extraction
│   │   └── data_collection_service.py  # Main orchestrator
│   ├── utils/                 # Utility functions
│   │   ├── pdf_extractor.py   # PDF text extraction
│   │   └── text_processor.py  # Text processing utilities
│   ├── uploads/               # User uploaded PDFs
│   ├── results/               # Generated chunks and keywords (JSON)
│   └── scraped_data/          # Downloaded content (ArXiv + Web)
└── frontend/                   # React frontend (separate)
```

## 🛠️ **Setup Instructions**

### **1. Prerequisites**
- Python 3.8+ installed
- Git installed
- API Keys for:
  - Google Gemini AI (for keyword generation)
  - Tavily API (for web scraping)

### **2. Clone Repository**
```bash
git clone https://github.com/Soumaditya-Kashyap/PLG-Check.git
cd PLG-Check/backend
```

### **3. Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### **4. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **5. Environment Configuration**
Create `.env` file in `backend/` folder:
```env
# Google Gemini AI Configuration
GOOGLE_API_KEY=your_google_gemini_api_key_here

# Tavily Web Scraping API
TAVILY_API_KEY=your_tavily_api_key_here

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
```

### **6. Run the Application**
```bash
python app.py
```

Server will start at: `http://127.0.0.1:5000`

## 📊 **API Endpoints**

### **POST /upload**
Upload and process a PDF file for plagiarism checking preparation.

**Request:**
- `file`: PDF file (multipart/form-data)

**Response:**
```json
{
  "success": true,
  "message": "File processing completed successfully",
  "file_id": "unique_document_id",
  "results": {
    "chunks_created": 45,
    "keywords_extracted": 15,
    "arxiv_papers_found": 8,
    "web_content_scraped": 12
  }
}
```

### **GET /health**
Check if the server is running properly.

## 🗂️ **Data Storage Structure**

### **Results Folder** (`results/DOCUMENT_ID/`)
- `chunks.json` - Text chunks from uploaded PDF
- `keywords.json` - AI-generated search keywords

### **Scraped Data Folder** (`scraped_data/DOCUMENT_ID/`)
```
arxiv/
├── arxiv_results.json          # Search results metadata
└── pdfs/                       # Downloaded ArXiv PDFs
    ├── paper1.pdf
    ├── paper2.pdf
    └── metadata/               # Paper metadata (JSON)
        ├── paper1.json
        └── paper2.json

web/
├── web_results.json            # Search results metadata
└── content/                    # Extracted web content
    ├── content1.json
    ├── content2.json
    └── ...
```

## 🔧 **Configuration Options**

### **Key Settings** (`config.py`)
```python
# File Processing
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'pdf'}

# ArXiv Scraping
ARXIV_MAX_RESULTS = 10
ARXIV_MAX_DOWNLOADS = 5

# Web Scraping
WEB_SEARCH_MAX_RESULTS = 10
MAX_CONTENT_EXTRACTIONS = 8

# Text Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

## 🚦 **Usage Workflow**

1. **📤 Upload PDF** - User uploads academic paper via web interface
2. **📝 Text Extraction** - System extracts text and creates chunks
3. **🧠 Keyword Generation** - AI generates relevant search terms
4. **📚 ArXiv Search** - Download related academic papers
5. **🌐 Web Search** - Extract relevant web content
6. **💾 Data Storage** - All content saved in structured JSON format
7. **✅ Ready for Analysis** - Data prepared for similarity detection

## 📋 **Dependencies Explanation**

### **Core Framework**
- `Flask` - Web framework for API endpoints
- `Flask-CORS` - Cross-origin request handling

### **PDF Processing**
- `PyPDF2` - Extract text from PDF files

### **Web & API Integration**
- `requests` - HTTP requests for ArXiv and web APIs
- `trafilatura` - Clean web content extraction
- `google-generativeai` - Google Gemini AI integration

### **Utilities**
- `python-dotenv` - Environment variable management
- `numpy` - Numerical operations for text processing

## 🔍 **Testing**

### **Test PDF Upload**
```bash
curl -X POST -F "file=@test_paper.pdf" http://127.0.0.1:5000/upload
```

### **Check Server Health**
```bash
curl http://127.0.0.1:5000/health
```

## 📈 **Future Enhancements**
- 🔄 Vector embeddings creation (FAISS integration)
- 🎯 Similarity analysis and scoring
- 📊 Plagiarism report generation
- 🎨 Enhanced web interface
- 📱 Mobile application support

## 🆘 **Troubleshooting**

### **Common Issues**

1. **PDF Text Extraction Fails**
   - Ensure PDF is text-based (not scanned image)
   - Check file size under 16MB limit

2. **API Key Errors**
   - Verify `.env` file exists and contains valid API keys
   - Check API key permissions and usage limits

3. **Missing Dependencies**
   - Run `pip install -r requirements.txt` again
   - Check Python version compatibility

### **Log Files**
Check console output for detailed error messages and processing status.

## 👥 **Contributors**
- **Development Team** - Academic plagiarism detection system
- **AI Integration** - Google Gemini AI for intelligent keyword extraction
- **Web Scraping** - Tavily API for comprehensive content collection

## 📄 **License**
This project is developed for academic purposes and research use.

---

**🎯 Status: Data Pipeline Complete ✅**  
**🔄 Next Phase: Vector Embeddings & Similarity Analysis**
