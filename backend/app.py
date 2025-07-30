from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename

from config import Config
from services.plagiarism_service import PlagiarismService
from utils.pdf_extractor import PDFExtractor

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

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize services
plagiarism_service = PlagiarismService()
pdf_extractor = PDFExtractor()

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
    """Upload PDF file and perform plagiarism check"""
    logger.info("=" * 50)
    logger.info("NEW PLAGIARISM CHECK REQUEST")
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
        
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        logger.info(f"Processing file: {file.filename}")
        logger.info(f"Document ID: {document_id}")
        logger.info(f"File size: {len(file.read())} bytes")
        file.seek(0)  # Reset file pointer
        
        # Extract text from PDF
        try:
            logger.info("\nSTEP 1: EXTRACTING TEXT FROM PDF")
            raw_text = pdf_extractor.extract_text_from_pdf(file)
            
            if not raw_text or len(raw_text.strip()) < 100:
                logger.warning("ERROR: PDF appears to be empty or contains insufficient text")
                return jsonify({'error': 'PDF appears to be empty or contains insufficient text'}), 400
            
            # Clean and chunk the text
            logger.info(f"Text extracted: {len(raw_text):,} characters")
            cleaned_text = pdf_extractor.clean_text(raw_text)
            
            logger.info("\nSTEP 2: PREPARING TEXT FOR ANALYSIS")
            text_chunks = pdf_extractor.chunk_text(cleaned_text)
            
            logger.info(f"Created {len(text_chunks)} text chunks for analysis")
            logger.info(f"Average chunk size: {len(cleaned_text) // len(text_chunks) if text_chunks else 0} characters")
        except Exception as e:
            logger.error(f"ERROR: Failed to extract text from PDF - {str(e)}")
            return jsonify({'error': f'Failed to extract text from PDF: {str(e)}'}), 400
        
        # Perform plagiarism check
        try:
            logger.info("\nSTEP 3: STARTING PLAGIARISM ANALYSIS")
            logger.info("Searching academic papers and web content...")
            
            results = plagiarism_service.check_plagiarism(text_chunks, document_id)
            
            # Add document metadata
            results['document_info'] = {
                'document_id': document_id,
                'filename': secure_filename(file.filename),
                'upload_time': datetime.now().isoformat(),
                'text_length': len(cleaned_text),
                'chunk_count': len(text_chunks)
            }
            
            # Log results summary
            logger.info("\n" + "=" * 50)
            logger.info("PLAGIARISM ANALYSIS COMPLETED")
            logger.info("=" * 50)
            logger.info(f"Document: {file.filename}")
            logger.info(f"Plagiarism Score: {results.get('plagiarism_percentage', 0)}%")
            logger.info(f"Total Chunks Analyzed: {results.get('total_chunks', 0)}")
            logger.info(f"Flagged Chunks: {results.get('flagged_chunks', 0)}")
            logger.info(f"ArXiv Sources Found: {results.get('sources_searched', {}).get('arxiv_papers', 0)}")
            logger.info(f"Web Sources Found: {results.get('sources_searched', {}).get('web_pages', 0)}")
            logger.info(f"Similar Matches Found: {len(results.get('matches', []))}")
            
            if results.get('matches'):
                logger.info("\nTOP SIMILARITY MATCHES:")
                for i, match in enumerate(results['matches'][:5], 1):
                    similarity = match.get('similarity', 0) * 100
                    source_type = match.get('source_type', 'unknown')
                    source_title = match.get('source_title', 'No title')[:60]
                    logger.info(f"{i}. {similarity:.1f}% similar - {source_type}")
                    logger.info(f"   Source: {source_title}")
                    
                    if match.get('source_url'):
                        logger.info(f"   URL: {match.get('source_url')}")
                    
                    # Show snippet of your text vs matched content
                    your_text = match.get('query_chunk', '')[:80]
                    matched_text = match.get('matched_content', '')[:80]
                    if your_text and matched_text:
                        logger.info(f"   Your text: {your_text}...")
                        logger.info(f"   Found text: {matched_text}...")
                    logger.info("")
            
            logger.info("=" * 50)
            logger.info("ANALYSIS COMPLETE - Results sent to frontend")
            logger.info("=" * 50)
            
            return jsonify(results)
            
        except Exception as e:
            logger.error(f"ERROR: Plagiarism check failed - {str(e)}")
            return jsonify({'error': f'Plagiarism check failed: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"❌ Unexpected error in upload endpoint: {str(e)}")
        logger.error("🔧 Full error traceback:", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

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
