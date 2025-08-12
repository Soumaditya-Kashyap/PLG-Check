# 📄 Plagiarism Checker - Data Scraping System

## 🎯 **What This Does**
Processes uploaded PDFs, extracts relevant research content from ArXiv and web sources, and stores them in organized JSON format for analysis.

## 🚀 **Current Features**
- ✅ PDF text extraction and chunking
- ✅ AI keyword generation (Google Gemini)
- ✅ ArXiv research paper downloading
- ✅ Web content scraping (Tavily API)
- ✅ Structured JSON data storage

## 📁 **Backend File Structure**
```
backend/
├── app.py                          # Main Flask server & API endpoints
├── config.py                       # Configuration settings & constants
├── requirements.txt                # Python dependencies
├── services/
│   ├── data_collection_service.py  # Main orchestrator - coordinates all services
│   ├── llm_service.py             # Google Gemini AI integration
│   ├── arxiv_service.py           # ArXiv paper search & download
│   └── tavily_service.py          # Web content scraping & extraction
└── utils/
    └── smart_text_processor.py    # PDF processing & text chunking
```

## 🛠️ **Setup**

### **1. Install Dependencies**
```bash
# Windows
cd backend
pip install -r requirements.txt

# Linux/Mac
cd backend
pip3 install -r requirements.txt
```

### **2. API Keys Setup**
Create `backend/.env` file:
```env
GOOGLE_API_KEY=your_google_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### **3. Run Backend Server**
```bash
# Windows
python app.py

# Linux/Mac
python3 app.py
```
Backend runs at: `http://127.0.0.1:5000`

### **4. Run Frontend (Optional)**
```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```
Frontend runs at: `http://localhost:5173`

## 📊 **API Usage**

### **Upload PDF**
```bash
curl -X POST -F "file=@document.pdf" http://127.0.0.1:5000/upload
```

### **Response**
```json
{
  "success": true,
  "message": "File processing completed successfully",
  "results": {
    "chunks_created": 45,
    "keywords_extracted": 15,
    "arxiv_papers_found": 8,
    "web_content_scraped": 12
  }
}
```

## 📁 **Generated Data Structure**
```
results/DOCUMENT_ID/
├── chunks.json     # PDF text chunks
└── keywords.json   # AI-generated keywords

scraped_data/DOCUMENT_ID/
├── arxiv/          # Downloaded research papers
│   ├── arxiv_results.json
│   └── pdfs/
└── web/            # Scraped web content  
    ├── web_results.json
    └── content/
```

## 🔧 **Required API Keys**
- **Google Gemini AI** - For keyword generation
- **Tavily API** - For web content scraping

## ✅ **System Status**
**Phase 1 Complete**: Data Collection & Storage  
**Ready For**: Vector Embeddings & Similarity Analysis

## 🔄 **How It Works (Workflow)**

### **Step 1: PDF Upload**
User uploads academic PDF → `app.py` receives file → saves to `uploads/` folder

### **Step 2: Text Processing**
`smart_text_processor.py` → extracts text → creates chunks → saves to `results/DOCUMENT_ID/chunks.json`

### **Step 3: AI Keyword Generation**
`llm_service.py` → sends text to Google Gemini → generates search keywords → saves to `results/DOCUMENT_ID/keywords.json`

### **Step 4: Research Paper Collection**
`arxiv_service.py` → searches ArXiv using keywords → downloads relevant PDFs → saves to `scraped_data/DOCUMENT_ID/arxiv/`

### **Step 5: Web Content Scraping**
`tavily_service.py` → searches web using keywords → extracts clean content → saves to `scraped_data/DOCUMENT_ID/web/`

### **Step 6: Data Ready**
All content organized in JSON format → ready for similarity analysis → plagiarism detection can begin

**🎯 Result: Complete academic content database for uploaded PDF**
