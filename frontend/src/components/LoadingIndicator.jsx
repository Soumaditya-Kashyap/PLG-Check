import './LoadingIndicator.css'

function LoadingIndicator() {
  return (
    <div className="loading-container">
      <div className="loading-spinner">
        <div className="spinner"></div>
      </div>
      
      <div className="loading-content">
        <h3>🔍 Analyzing Your Document</h3>
        <div className="loading-steps">
          <div className="step">
            <div className="step-icon">📄</div>
            <span>Extracting text from PDF...</span>
          </div>
          <div className="step">
            <div className="step-icon">📚</div>
            <span>Searching academic papers on arXiv...</span>
          </div>
          <div className="step">
            <div className="step-icon">🌐</div>
            <span>Searching web content...</span>
          </div>
          <div className="step">
            <div className="step-icon">🤖</div>
            <span>Analyzing similarities with AI...</span>
          </div>
          <div className="step">
            <div className="step-icon">📊</div>
            <span>Generating plagiarism report...</span>
          </div>
        </div>
        
        <div className="loading-message">
          <p>This may take a few minutes depending on document size and content complexity.</p>
          <small>Please don't close this tab while processing.</small>
        </div>
      </div>
    </div>
  )
}

export default LoadingIndicator
