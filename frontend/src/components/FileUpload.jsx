import { useState, useRef } from 'react'
import './FileUpload.css'

function FileUpload({ onFileUpload }) {
  const [isDragActive, setIsDragActive] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const fileInputRef = useRef(null)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragActive(true)
    } else if (e.type === 'dragleave') {
      setIsDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)

    const files = e.dataTransfer.files
    if (files && files[0]) {
      handleFileSelection(files[0])
    }
  }

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0])
    }
  }

  const handleFileSelection = (file) => {
    if (file.type !== 'application/pdf') {
      alert('Please select a PDF file')
      return
    }
    
    if (file.size > 16 * 1024 * 1024) { // 16MB
      alert('File size must be less than 16MB')
      return
    }

    setSelectedFile(file)
  }

  const handleUpload = () => {
    if (selectedFile) {
      onFileUpload(selectedFile)
    }
  }

  const handleBrowseClick = () => {
    fileInputRef.current?.click()
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div className="file-upload-container">
      <div
        className={`file-upload-area ${isDragActive ? 'drag-active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <div className="upload-icon">📄</div>
        <h3>Upload PDF Document</h3>
        <p>Drag and drop your PDF file here, or</p>
        <button 
          type="button" 
          className="browse-button"
          onClick={handleBrowseClick}
        >
          Browse Files
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          onChange={handleFileInput}
          style={{ display: 'none' }}
        />
        <small>Maximum file size: 16MB • Supported format: PDF</small>
      </div>

      {selectedFile && (
        <div className="selected-file">
          <div className="file-info">
            <div className="file-details">
              <strong>{selectedFile.name}</strong>
              <span>{formatFileSize(selectedFile.size)}</span>
            </div>
            <button 
              className="remove-file"
              onClick={() => setSelectedFile(null)}
            >
              ❌
            </button>
          </div>
          <button 
            className="check-plagiarism-button"
            onClick={handleUpload}
          >
            🔍 Check Plagiarism
          </button>
        </div>
      )}
    </div>
  )
}

export default FileUpload
