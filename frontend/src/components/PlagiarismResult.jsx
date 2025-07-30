import './PlagiarismResult.css'

function PlagiarismResult({ results, onReset }) {
  const {
    plagiarism_percentage,
    total_chunks,
    flagged_chunks,
    matches,
    sources_searched,
    document_info,
    arxiv_acknowledgment
  } = results

  const getPercentageColor = (percentage) => {
    if (percentage < 15) return '#4caf50' // Green
    if (percentage < 30) return '#ff9800' // Orange
    return '#f44336' // Red
  }

  const getPercentageLevel = (percentage) => {
    if (percentage < 15) return 'Low Risk'
    if (percentage < 30) return 'Medium Risk'
    return 'High Risk'
  }

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString()
  }

  const truncateText = (text, maxLength = 200) => {
    if (text.length <= maxLength) return text
    return text.substring(0, maxLength) + '...'
  }

  return (
    <div className="plagiarism-result">
      <div className="result-header">
        <h2>📊 Plagiarism Analysis Results</h2>
        <button className="reset-button" onClick={onReset}>
          🔄 Check Another Document
        </button>
      </div>

      {/* Overall Score */}
      <div className="score-container">
        <div className="score-circle">
          <div 
            className="score-text"
            style={{ color: getPercentageColor(plagiarism_percentage) }}
          >
            <span className="percentage">{plagiarism_percentage}%</span>
            <span className="risk-level">{getPercentageLevel(plagiarism_percentage)}</span>
          </div>
        </div>
        <div className="score-details">
          <h3>Similarity Detected</h3>
          <p>{flagged_chunks} out of {total_chunks} text chunks showed potential similarities</p>
          <div className="sources-info">
            <span>📚 ArXiv Papers: {sources_searched?.arxiv_papers || 0}</span>
            <span>🌐 Web Sources: {sources_searched?.web_pages || 0}</span>
          </div>
        </div>
      </div>

      {/* Document Information */}
      <div className="document-info">
        <h3>📄 Document Information</h3>
        <div className="info-grid">
          <div className="info-item">
            <strong>Filename:</strong> {document_info?.filename}
          </div>
          <div className="info-item">
            <strong>Upload Time:</strong> {document_info?.upload_time ? formatDate(document_info.upload_time) : 'N/A'}
          </div>
          <div className="info-item">
            <strong>Text Length:</strong> {document_info?.text_length?.toLocaleString()} characters
          </div>
          <div className="info-item">
            <strong>Analyzed Chunks:</strong> {total_chunks}
          </div>
        </div>
      </div>

      {/* Matches */}
      {matches && matches.length > 0 && (
        <div className="matches-container">
          <h3>🔍 Similar Content Found</h3>
          <div className="matches-list">
            {matches.slice(0, 10).map((match, index) => (
              <div key={index} className="match-item">
                <div className="match-header">
                  <div className="match-source">
                    <span className={`source-badge ${match.source_type}`}>
                      {match.source_type === 'arxiv' ? '📚 ArXiv' : '🌐 Web'}
                    </span>
                    <span className="similarity-score">
                      {(match.similarity * 100).toFixed(1)}% similar
                    </span>
                  </div>
                </div>
                
                <div className="match-content">
                  <div className="your-text">
                    <strong>Your text:</strong>
                    <p>"{truncateText(match.query_chunk)}"</p>
                  </div>
                  
                  <div className="similar-text">
                    <strong>Similar content found:</strong>
                    <p>"{truncateText(match.matched_content)}"</p>
                  </div>
                </div>

                {match.metadata && (
                  <div className="match-metadata">
                    {match.source_type === 'arxiv' && (
                      <>
                        {match.metadata.title && (
                          <div><strong>Paper:</strong> {match.metadata.title}</div>
                        )}
                        {match.metadata.authors && (
                          <div><strong>Authors:</strong> {match.metadata.authors}</div>
                        )}
                        {match.metadata.url && (
                          <div>
                            <strong>URL:</strong> 
                            <a href={match.metadata.url} target="_blank" rel="noopener noreferrer">
                              {match.metadata.url}
                            </a>
                          </div>
                        )}
                      </>
                    )}
                    
                    {match.source_type === 'web' && (
                      <>
                        {match.metadata.title && (
                          <div><strong>Page:</strong> {match.metadata.title}</div>
                        )}
                        {match.metadata.url && (
                          <div>
                            <strong>URL:</strong> 
                            <a href={match.metadata.url} target="_blank" rel="noopener noreferrer">
                              {match.metadata.url}
                            </a>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
          
          {matches.length > 10 && (
            <div className="matches-summary">
              <p>Showing top 10 matches out of {matches.length} total matches found.</p>
            </div>
          )}
        </div>
      )}

      {/* No matches found */}
      {(!matches || matches.length === 0) && (
        <div className="no-matches">
          <h3>✅ No Significant Similarities Found</h3>
          <p>Your document appears to be original. No substantial similarities were detected in our academic and web databases.</p>
        </div>
      )}

      {/* Acknowledgment */}
      {arxiv_acknowledgment && (
        <div className="acknowledgment">
          <p><em>{arxiv_acknowledgment}</em></p>
        </div>
      )}
    </div>
  )
}

export default PlagiarismResult
