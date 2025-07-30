import { useState } from 'react'
import './App.css'
import FileUpload from './components/FileUpload'
import PlagiarismResult from './components/PlagiarismResult'
import LoadingIndicator from './components/LoadingIndicator'

function App() {
  const [isLoading, setIsLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  const handleFileUpload = async (file) => {
    setIsLoading(true)
    setError(null)
    setResults(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Failed to check plagiarism')
      }

      setResults(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleReset = () => {
    setResults(null)
    setError(null)
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>📄 Plagiarism Checker</h1>
        <p>Upload a PDF document to check for plagiarism against academic papers and web content</p>
      </header>

      <main className="app-main">
        {!results && !isLoading && (
          <FileUpload onFileUpload={handleFileUpload} />
        )}

        {isLoading && <LoadingIndicator />}

        {error && (
          <div className="error-container">
            <h3>❌ Error</h3>
            <p>{error}</p>
            <button onClick={handleReset} className="retry-button">
              Try Again
            </button>
          </div>
        )}

        {results && (
          <PlagiarismResult results={results} onReset={handleReset} />
        )}
      </main>

      <footer className="app-footer">
        <p>Powered by arXiv API and Tavily Search</p>
        <small>Thank you to arXiv for use of its open access interoperability.</small>
      </footer>
    </div>
  )
}

export default App
