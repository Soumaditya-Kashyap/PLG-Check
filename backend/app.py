from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import uuid
import json
from datetime import datetime
from werkzeug.utils import secure_filename

from config import Config
from services.plagiarism_service import PlagiarismService
from services.data_collection_service import DataCollectionService
from utils.pdf_extractor import PDFExtractor
from utils.smart_text_processor import SmartTextProcessor
from utils.text_processor import TextProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),  # Console output only for clean terminal
    ]
)

# Reduce noise from other libraries
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load configuration from Config class
app.config.from_object(Config)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'pdf'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize services
plagiarism_service = PlagiarismService()
data_collection_service = DataCollectionService()
pdf_extractor = PDFExtractor()
smart_text_processor = SmartTextProcessor()
text_processor = TextProcessor()

def extract_fallback_keywords(text):
    """Extract keywords using simple text analysis"""
    import re
    # Remove special characters and get words
    words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())
    # Filter common words
    stop_words = {'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'they', 'have', 'been', 'will', 'can', 'but', 'not', 'use', 'used', 'also', 'such', 'which', 'more', 'than', 'may', 'all', 'our', 'their', 'these', 'into', 'data', 'system', 'work', 'paper', 'show', 'using'}
    keywords = [word for word in words if word not in stop_words and len(word) > 3]
    # Return unique keywords, limited to 5
    return list(dict.fromkeys(keywords))[:5]

def extract_fallback_phrases(text):
    """Extract key phrases using simple pattern matching"""
    import re
    # Look for capitalized phrases (potential technical terms)
    phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    # Look for acronyms
    acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
    all_phrases = phrases + acronyms
    # Return unique phrases, limited to 5
    return list(dict.fromkeys(all_phrases))[:5]

def extract_fallback_key_points(text):
    """Extract key points by finding sentences with important indicators"""
    import re
    sentences = re.split(r'[.!?]+', text)
    key_indicators = ['proposed', 'present', 'design', 'implement', 'achieve', 'results', 'conclusion', 'contribution', 'novel', 'first', 'performance', 'security', 'mechanism']
    key_points = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20 and any(indicator in sentence.lower() for indicator in key_indicators):
            key_points.append(sentence[:100] + "..." if len(sentence) > 100 else sentence)
    return key_points[:5] if key_points else ["Key technical content identified in this section"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Plagiarism Checker API'
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload PDF file and create chunks.json and keywords.json in results folder"""
    logger.info("=" * 50)
    logger.info("NEW FILE UPLOAD REQUEST")
    logger.info("=" * 50)
    
    try:
        # Check if file is present
        if 'file' not in request.files:
            logger.warning("ERROR: No file provided in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            logger.warning("ERROR: No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            logger.warning(f"ERROR: Invalid file type - {file.filename}")
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Generate unique document ID and save file
        document_id = str(uuid.uuid4())
        pdf_filename = f"{document_id}.pdf"
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
        
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save uploaded file
        file.save(pdf_path)
        
        # Get original filename without extension for folder name
        original_filename = secure_filename(file.filename)
        file_basename = os.path.splitext(original_filename)[0]
        
        logger.info(f"Processing file: {file.filename}")
        logger.info(f"Document ID: {document_id}")
        logger.info(f"Saved to: {pdf_path}")
        
        # Create results folder for this file
        results_base_dir = "results"
        os.makedirs(results_base_dir, exist_ok=True)
        
        file_results_dir = os.path.join(results_base_dir, file_basename)
        os.makedirs(file_results_dir, exist_ok=True)
        
        logger.info(f"Created results folder: {file_results_dir}")
        
        # Extract text from PDF using smart processor
        try:
            logger.info("\nSTEP 1: EXTRACTING TEXT FROM PDF WITH SMART CHUNKING")
            raw_text = smart_text_processor.extract_text_from_pdf(pdf_path)
            
            if not raw_text or len(raw_text.strip()) < 100:
                logger.warning("ERROR: PDF appears to be empty or contains insufficient text")
                return jsonify({'error': 'PDF appears to be empty or contains insufficient text'}), 400
            
            # Clean and chunk the text using SMART CHUNKING
            logger.info(f"Text extracted: {len(raw_text):,} characters")
            cleaned_text = pdf_extractor.clean_text(raw_text)
            
            logger.info("\nSTEP 2: CREATING SMART SEMANTIC CHUNKS")
            logger.info("🧠 Using Advanced Semantic Chunking...")
            
            # Use smart chunker for semantic chunking
            smart_chunks = smart_text_processor.create_semantic_chunks(pdf_path)
            
            logger.info(f"✅ Created {len(smart_chunks)} SMART SEMANTIC chunks")
            logger.info(f"📊 Sections identified: {len(set(chunk['section'] for chunk in smart_chunks))}")
            
            # Convert to simple format like leader's test.chunks.json (with section info)
            simple_chunks = []
            for chunk in smart_chunks:
                simple_chunks.append({
                    "text": chunk["text"],
                    "section": chunk.get("section", "Unknown")
                })
            
            logger.info(f"📄 Converted to {len(simple_chunks)} simple text chunks with section info")
            
            # Save chunks.json in simple format
            chunks_json_path = os.path.join(file_results_dir, "chunks.json")
            with open(chunks_json_path, 'w', encoding='utf-8') as f:
                json.dump(simple_chunks, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved chunks.json: {chunks_json_path}")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to extract text from PDF - {str(e)}")
            return jsonify({'error': f'Failed to extract text from PDF: {str(e)}'}), 400
        
        # Extract keywords using LLM
        try:
            logger.info("\nSTEP 3: EXTRACTING KEYWORDS USING LLM")
            
            from services.llm_service import LLMService
            llm_service = LLMService()
            
            if llm_service.is_available():
                logger.info("🤖 Using LLM for keyword extraction...")
                
                # Prepare sections for LLM processing (use original smart_chunks for section info)
                sections_for_llm = []
                for i, chunk in enumerate(smart_chunks):
                    sections_for_llm.append({
                        'id': f"chunk_{i}",
                        'text': chunk['text'],
                        'type': chunk.get('section', 'unknown'),
                        'section': chunk.get('section', 'unknown')
                    })
                
                # Extract keywords using LLM batch processing
                keyword_results = []
                
                # Process sections in batches of 3 (smaller batches for faster processing)
                batch_size = 3
                for i in range(0, len(sections_for_llm), batch_size):
                    batch = sections_for_llm[i:i+batch_size]
                    
                    # Only process first 20 sections to avoid overwhelming the LLM
                    if i >= 60:  # Stop after 20 batches (60 sections)
                        logger.info(f"⏱️ Limiting processing to first 60 sections for faster response")
                        break
                    
                    # Create batch text for LLM
                    batch_text = ""
                    for sec in batch:
                        batch_text += f"Section: {sec['section']}\nText:\n{sec['text']}\n\n"
                    
                    # Use LLM to extract keywords and key phrases
                    prompt = (
                        "You are a research assistant. For each section below, extract:\n"
                        "1. **Top 3 most relevant keywords** (single words, directly tied to the section context)\n"
                        "2. **Top 3 most relevant key phrases** (multi-word, directly tied to the section context)\n"
                        "3. **All important key points** (bullet points, capturing every crucial idea or fact in the section)\n\n"
                        "Return ONLY valid JSON in the format:\n"
                        "[\n"
                        "  {\n"
                        '    "section": "section name",\n'
                        '    "keywords": ["word1", "word2", "word3"],\n'
                        '    "key_phrases": ["phrase1", "phrase2", "phrase3"],\n'
                        '    "key_points": ["point 1", "point 2", "point 3"]\n'
                        "  }\n"
                        "]\n"
                        f"Sections:\n{batch_text}"
                    )
                    
                    try:
                        response = llm_service.model.generate_content(prompt)
                        
                        if response and response.text:
                            # Try to parse JSON response
                            response_text = response.text.strip()
                            
                            # Remove code fences if present
                            import re
                            response_text = re.sub(r"^```(json)?", "", response_text.strip(), flags=re.IGNORECASE)
                            response_text = re.sub(r"```$", "", response_text.strip())
                            
                            # Try to extract JSON
                            match = re.search(r"\[.*\]|\{.*\}", response_text, re.DOTALL)
                            if match:
                                response_text = match.group(0)
                            
                            try:
                                parsed_results = json.loads(response_text)
                                if isinstance(parsed_results, list):
                                    keyword_results.extend(parsed_results)
                                else:
                                    # Handle single object response
                                    keyword_results.append(parsed_results)
                            except json.JSONDecodeError:
                                # Fallback: create entries for each section in batch
                                for sec in batch:
                                    keyword_results.append({
                                        "section": sec["section"],
                                        "error": "Failed to parse LLM output for this section",
                                        "raw_output": response.text
                                    })
                        else:
                            # Empty response fallback
                            for sec in batch:
                                keyword_results.append({
                                    "section": sec["section"],
                                    "error": "Empty response from LLM"
                                })
                    
                    except Exception as batch_error:
                        logger.error(f"Error processing batch {i//batch_size + 1}: {batch_error}")
                        # Check if it's a quota error
                        if "quota" in str(batch_error).lower() or "429" in str(batch_error):
                            logger.warning("⚠️ LLM quota exceeded, switching to fallback extraction")
                            # Use fallback for this batch
                            for sec in batch:
                                section_name = sec.get('section', 'unknown')
                                text_content = sec.get('text', '')
                                
                                fallback_keywords = extract_fallback_keywords(text_content)
                                fallback_phrases = extract_fallback_phrases(text_content)
                                fallback_points = extract_fallback_key_points(text_content)
                                
                                keyword_results.append({
                                    "section": section_name,
                                    "keywords": fallback_keywords[:3],
                                    "key_phrases": fallback_phrases[:3],
                                    "key_points": fallback_points[:5]
                                })
                        else:
                            # Add error entries for this batch
                            for sec in batch:
                                keyword_results.append({
                                    "section": sec["section"],
                                    "error": f"LLM processing error: {str(batch_error)}"
                                })
                
                logger.info(f"🏷️ Extracted keywords for {len(keyword_results)} sections")
                
            else:
                logger.warning("⚠️ LLM service not available, using fallback keyword extraction")
                # Enhanced fallback keyword extraction
                keyword_results = []
                for i, chunk in enumerate(smart_chunks):
                    section_name = chunk.get('section', 'unknown')
                    text_content = chunk.get('text', '')
                    
                    # Extract keywords using basic text analysis
                    fallback_keywords = extract_fallback_keywords(text_content)
                    fallback_phrases = extract_fallback_phrases(text_content)
                    fallback_points = extract_fallback_key_points(text_content)
                    
                    keyword_results.append({
                        "section": section_name,
                        "keywords": fallback_keywords[:3],
                        "key_phrases": fallback_phrases[:3],
                        "key_points": fallback_points[:5]
                    })
            
            # Save keywords.json
            keywords_json_path = os.path.join(file_results_dir, "keywords.json")
            with open(keywords_json_path, 'w', encoding='utf-8') as f:
                json.dump(keyword_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved keywords.json: {keywords_json_path}")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to extract keywords - {str(e)}")
            return jsonify({'error': f'Failed to extract keywords: {str(e)}'}), 400
        
        # STEP 4: ArXiv Data Collection
        try:
            logger.info("\nSTEP 4: COLLECTING ARXIV DATA")
            
            # Create scraped_data folder structure
            scraped_base_dir = "scraped_data"
            os.makedirs(scraped_base_dir, exist_ok=True)
            
            document_scraped_dir = os.path.join(scraped_base_dir, file_basename)
            os.makedirs(document_scraped_dir, exist_ok=True)
            
            arxiv_dir = os.path.join(document_scraped_dir, "arxiv")
            web_dir = os.path.join(document_scraped_dir, "web")
            os.makedirs(arxiv_dir, exist_ok=True)
            os.makedirs(web_dir, exist_ok=True)
            
            # Extract and prioritize keywords from the generated keywords.json for ArXiv search
            extracted_keywords = []
            extracted_phrases = []
            
            # Prioritize keywords from different sections
            section_priorities = {
                'title': 3.0,
                'abstract': 2.5, 
                'introduction': 2.0,
                'conclusion': 2.0,
                'methodology': 1.5,
                'results': 1.5
            }
            
            for result in keyword_results:
                if isinstance(result, dict) and not result.get('error'):
                    section_name = result.get('section', '').lower()
                    
                    # Get priority multiplier for this section
                    priority = 1.0
                    for key, mult in section_priorities.items():
                        if key in section_name:
                            priority = mult
                            break
                    
                    # Extract keywords and phrases with priority weighting
                    keywords = result.get('keywords', [])
                    key_phrases = result.get('key_phrases', [])
                    
                    # Add keywords with priority
                    for kw in keywords:
                        if kw and len(kw.strip()) > 2:
                            extracted_keywords.extend([kw.strip()] * int(priority))
                    
                    # Add phrases with priority  
                    for phrase in key_phrases:
                        if phrase and len(phrase.strip()) > 3:
                            extracted_phrases.extend([phrase.strip()] * int(priority))
            
            # Get most frequent (prioritized) keywords and phrases
            from collections import Counter
            keyword_counts = Counter(extracted_keywords)
            phrase_counts = Counter(extracted_phrases)
            
            # Get top keywords and phrases
            top_keywords = [kw for kw, count in keyword_counts.most_common(10) 
                           if not kw.lower() in ['error', 'fallback', 'section', 'unknown']]
            top_phrases = [phrase for phrase, count in phrase_counts.most_common(8)
                          if not phrase.lower() in ['error', 'fallback', 'section', 'unknown']]
            
            # Combine and create final search terms (prioritize phrases over single keywords)
            final_search_terms = top_phrases[:5] + top_keywords[:5]
            
            logger.info(f"🔍 Prioritized search terms: {len(final_search_terms)} terms")
            logger.info(f"🏷️ Top phrases: {', '.join(top_phrases[:5])}")
            logger.info(f"🏷️ Top keywords: {', '.join(top_keywords[:5])}")
            
            if final_search_terms:
                # Initialize ArXiv service
                from services.arxiv_service import ArxivService
                arxiv_service = ArxivService()
                
                # Set custom directory for this document
                arxiv_pdf_dir = os.path.join(arxiv_dir, "pdfs")
                os.makedirs(arxiv_pdf_dir, exist_ok=True)
                
                # Search and download ArXiv papers with improved search terms
                logger.info(f"📚 Searching ArXiv with prioritized terms...")
                arxiv_results = arxiv_service.search_and_download(
                    keywords=final_search_terms[:8],  # Use top 8 prioritized search terms
                    max_papers=8,  # Download up to 8 relevant papers (flexible range 5-10)
                    custom_pdf_dir=arxiv_pdf_dir
                )
                
                # Save ArXiv results metadata
                arxiv_metadata_path = os.path.join(arxiv_dir, "arxiv_results.json")
                with open(arxiv_metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(arxiv_results, f, indent=2, ensure_ascii=False)
                
                logger.info(f"📚 ArXiv Collection Results:")
                logger.info(f"   Papers Found: {arxiv_results.get('papers_found', 0)}")
                logger.info(f"   Papers Downloaded: {arxiv_results.get('papers_downloaded', 0)}")
                logger.info(f"   Storage: {arxiv_pdf_dir}")
                
                arxiv_success = True
                arxiv_papers_downloaded = arxiv_results.get('papers_downloaded', 0)
            else:
                logger.warning("⚠️ No valid keywords extracted for ArXiv search")
                arxiv_success = False
                arxiv_papers_downloaded = 0
                arxiv_results = {"error": "No valid keywords for search"}
            
        except Exception as e:
            logger.error(f"ERROR: ArXiv data collection failed - {str(e)}")
            arxiv_success = False
            arxiv_papers_downloaded = 0
            arxiv_results = {"error": str(e)}
        
        # Prepare response
        results = {
            'success': True,
            'message': 'File uploaded and processed successfully',
            'document_info': {
                'document_id': document_id,
                'filename': original_filename,
                'upload_time': datetime.now().isoformat(),
                'text_length': len(cleaned_text),
                'chunk_count': len(simple_chunks)
            },
            'file_paths': {
                'pdf_path': pdf_path,
                'results_folder': file_results_dir,
                'chunks_json': chunks_json_path,
                'keywords_json': keywords_json_path,
                'scraped_data_folder': document_scraped_dir if 'document_scraped_dir' in locals() else None
            },
            'arxiv_collection': {
                'success': arxiv_success if 'arxiv_success' in locals() else False,
                'papers_downloaded': arxiv_papers_downloaded if 'arxiv_papers_downloaded' in locals() else 0,
                'storage_path': arxiv_pdf_dir if 'arxiv_pdf_dir' in locals() else None,
                'results': arxiv_results if 'arxiv_results' in locals() else {}
            }
        }
        
        # Log results summary
        logger.info("\n" + "=" * 50)
        logger.info("FILE PROCESSING COMPLETED")
        logger.info("=" * 50)
        logger.info(f"Document: {file.filename}")
        logger.info(f"Results Folder: {file_results_dir}")
        logger.info(f"Chunks Created: {len(simple_chunks)}")
        logger.info(f"Keywords Extracted: {len(keyword_results)}")
        
        if 'document_scraped_dir' in locals():
            logger.info(f"Scraped Data Folder: {document_scraped_dir}")
        
        if 'arxiv_success' in locals() and arxiv_success:
            logger.info(f"ArXiv Papers Downloaded: {arxiv_papers_downloaded}")
            logger.info(f"ArXiv Storage: {arxiv_pdf_dir}")
        
        logger.info(f"Files Created:")
        logger.info(f"  - {chunks_json_path}")
        logger.info(f"  - {keywords_json_path}")
        
        if 'arxiv_metadata_path' in locals():
            logger.info(f"  - {arxiv_metadata_path}")
        
        logger.info("=" * 50)
        
        return jsonify(results)
            
    except Exception as e:
        logger.error(f"❌ Unexpected error in upload endpoint: {str(e)}")
        logger.error("🔧 Full error traceback:", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/collect-data', methods=['POST'])
def collect_data():
    """New endpoint for comprehensive data collection (LLM + ArXiv + Web)"""
    logger.info("=" * 50)
    logger.info("NEW DATA COLLECTION REQUEST")
    logger.info("=" * 50)
    
    try:
        # Check if file is present
        if 'file' not in request.files:
            logger.warning("ERROR: No file provided in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            logger.warning("ERROR: No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            logger.warning(f"ERROR: Invalid file type - {file.filename}")
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Generate unique document ID and save file
        document_id = str(uuid.uuid4())
        pdf_filename = f"{document_id}.pdf"
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
        
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save uploaded file
        file.save(pdf_path)
        
        logger.info(f"Processing file for data collection: {file.filename}")
        logger.info(f"Document ID: {document_id}")
        logger.info(f"Saved to: {pdf_path}")
        
        # Get optional parameters
        max_arxiv_papers = request.form.get('max_arxiv_papers', 5, type=int)
        max_web_results = request.form.get('max_web_results', 2, type=int)
        processing_mode = request.form.get('mode', 'comprehensive')  # 'comprehensive' or 'section_by_section'
        
        # Perform comprehensive data collection
        if processing_mode == 'section_by_section':
            logger.info("Using section-by-section processing mode")
            results = data_collection_service.process_sections_individually(
                pdf_path, max_arxiv_papers, max_web_results
            )
        else:
            logger.info("Using comprehensive processing mode")
            results = data_collection_service.process_pdf_comprehensive(
                pdf_path, max_arxiv_papers, max_web_results
            )
        
        # Add document metadata
        results['document_info'] = {
            'document_id': document_id,
            'filename': secure_filename(file.filename),
            'upload_time': datetime.now().isoformat(),
            'processing_mode': processing_mode
        }
        
        # Log results summary
        logger.info("\n" + "=" * 50)
        logger.info("DATA COLLECTION COMPLETED")
        logger.info("=" * 50)
        logger.info(f"Document: {file.filename}")
        logger.info(f"Processing Mode: {processing_mode}")
        
        if results.get('success'):
            logger.info(f"✅ Collection successful!")
            
            # Log ArXiv results
            arxiv_data = results.get('arxiv_collection', {})
            if arxiv_data:
                logger.info(f"📚 ArXiv Papers Downloaded: {arxiv_data.get('papers_downloaded', 0)}")
                logger.info(f"🔍 ArXiv Papers Found: {arxiv_data.get('papers_found', 0)}")
            
            # Log web results
            web_data = results.get('web_collection', {})
            if web_data:
                logger.info(f"🌐 Web Content Extracted: {web_data.get('total_content_extracted', 0)}")
                logger.info(f"🔗 Web Results Found: {web_data.get('total_results', 0)}")
            
            # Log keywords
            keywords_data = results.get('keywords_extraction', {})
            if keywords_data:
                combined_keywords = keywords_data.get('combined_keywords', [])
                logger.info(f"🏷️ Keywords Extracted: {len(combined_keywords)}")
                if combined_keywords:
                    logger.info(f"📝 Sample Keywords: {', '.join(combined_keywords[:5])}")
            
            # Log chunks
            chunks_data = results.get('chunks_analysis', {})
            if chunks_data:
                logger.info(f"📄 Text Chunks Created: {chunks_data.get('total_chunks', 0)}")
        else:
            logger.error(f"❌ Collection failed: {results.get('error', 'Unknown error')}")
        
        logger.info("=" * 50)
        logger.info("DATA COLLECTION COMPLETE - Results sent to frontend")
        logger.info("=" * 50)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in data collection endpoint: {str(e)}")
        logger.error("🔧 Full error traceback:", exc_info=True)
        return jsonify({'error': 'Internal server error during data collection'}), 500

@app.route('/collection-summary', methods=['GET'])
def collection_summary():
    """Get summary of all data collection activities"""
    try:
        summary = data_collection_service.get_collection_summary()
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error getting collection summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/check-status/<document_id>', methods=['GET'])
def check_status(document_id):
    """Check status of a plagiarism check (for future async implementation)"""
    return jsonify({
        'document_id': document_id,
        'status': 'completed',
        'message': 'This endpoint is reserved for future async implementation'
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("PLAGIARISM CHECKER API")
    print("=" * 50)
    logger.info("Starting server...")
    logger.info(f"Tavily API Key: {'Configured' if Config.TAVILY_API_KEY else 'Missing'}")
    logger.info(f"ChromaDB path: {Config.CHROMA_DB_PATH}")
    logger.info("Server ready at http://127.0.0.1:5000")
    print("=" * 50)
    
    app.run(debug=False, host='0.0.0.0', port=5000)
