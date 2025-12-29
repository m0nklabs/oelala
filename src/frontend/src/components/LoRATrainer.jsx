import React, { useState, useRef } from 'react'
import BACKEND_BASE from '../config'
import { postForm } from '../api'
import { Upload, Play, Download, Loader, BookOpen } from 'lucide-react'
import './LoRATrainer.css'

function LoRATrainer() {
  const [selectedFiles, setSelectedFiles] = useState([])
  const [previews, setPreviews] = useState([])
  const [modelName, setModelName] = useState('')
  const [numEpochs, setNumEpochs] = useState(10)
  const [learningRate, setLearningRate] = useState(1e-4)
  const [isTraining, setIsTraining] = useState(false)
  const [trainingStatus, setTrainingStatus] = useState(null)
  const [error, setError] = useState('')
  const fileInputRef = useRef(null)

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files)
    if (files.length > 0) {
      setSelectedFiles(files)
      setError('')

      // Create previews
      const newPreviews = files.map(file => {
        return new Promise((resolve) => {
          const reader = new FileReader()
          reader.onload = (e) => resolve(e.target.result)
          reader.readAsDataURL(file)
        })
      })

      Promise.all(newPreviews).then(setPreviews)
    }
  }

  const handleTrain = async () => {
    if (selectedFiles.length === 0) {
      setError('Please select at least one image')
      return
    }

    if (!modelName.trim()) {
      setError('Please enter a model name')
      return
    }

    setIsTraining(true)
    setError('')
    setTrainingStatus(null)

    const formData = new FormData()
    selectedFiles.forEach((file, index) => {
      formData.append(`files`, file)
    })
    formData.append('model_name', modelName)
    formData.append('num_epochs', numEpochs.toString())
    formData.append('learning_rate', learningRate.toString())

    try {
      const result = await postForm(`${BACKEND_BASE}/train-lora`, formData)
      if (!result.ok) {
        setError(result.data?.detail || `Training failed (status ${result.status})`)
      } else {
        setTrainingStatus(result.data)
      }
    } catch (err) {
      setError(err.message || 'Failed to start LoRA training')
    } finally {
      setIsTraining(false)
    }
  }

  const removeFile = (index) => {
    const newFiles = selectedFiles.filter((_, i) => i !== index)
    const newPreviews = previews.filter((_, i) => i !== index)
    setSelectedFiles(newFiles)
    setPreviews(newPreviews)
  }

  return (
    <div className="lora-trainer">
      <div className="trainer-header">
        <BookOpen size={24} />
        <h2>LoRA Model Training</h2>
        <p>Train a personalized LoRA model on your images for consistent avatar generation</p>
      </div>

      <div className="upload-section">
        <div className="upload-area" onClick={() => fileInputRef.current?.click()}>
          {previews.length > 0 ? (
            <div className="previews-grid">
              {previews.map((preview, index) => (
                <div key={index} className="preview-item">
                  <img src={preview} alt={`Preview ${index + 1}`} className="image-preview-small" />
                  <button
                    className="remove-btn"
                    onClick={(e) => {
                      e.stopPropagation()
                      removeFile(index)
                    }}
                  >
                    ×
                  </button>
                </div>
              ))}
              <div className="add-more">
                <Upload size={24} />
                <span>Add More</span>
              </div>
            </div>
          ) : (
            <div className="upload-placeholder">
              <Upload size={48} />
              <h3>Upload Training Images</h3>
              <p>Select multiple images to train your LoRA model</p>
              <p className="file-types">Supports: JPG, PNG, WebP (5-20 images recommended)</p>
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            multiple
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
        </div>
      </div>

      <div className="controls-section">
        <div className="control-group">
          <label htmlFor="modelName">Model Name</label>
          <input
            id="modelName"
            type="text"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder="e.g., my-avatar-style"
            required
          />
        </div>

        <div className="control-row">
          <div className="control-group">
            <label htmlFor="epochs">Training Epochs: {numEpochs}</label>
            <input
              id="epochs"
              type="range"
              min="5"
              max="50"
              value={numEpochs}
              onChange={(e) => setNumEpochs(parseInt(e.target.value))}
              step="5"
            />
            <div className="range-labels">
              <span>5</span>
              <span>25</span>
              <span>50</span>
            </div>
          </div>

          <div className="control-group">
            <label htmlFor="learningRate">Learning Rate: {learningRate}</label>
            <input
              id="learningRate"
              type="range"
              min="1e-5"
              max="1e-3"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
              step="1e-5"
            />
            <div className="range-labels">
              <span>1e-5</span>
              <span>5e-4</span>
              <span>1e-3</span>
            </div>
          </div>
        </div>

        <button
          className="train-btn"
          onClick={handleTrain}
          disabled={selectedFiles.length === 0 || !modelName.trim() || isTraining}
        >
          {isTraining ? (
            <>
              <Loader size={20} className="spinning" />
              Training LoRA Model...
            </>
          ) : (
            <>
              <Play size={20} />
              Start Training
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="error-message">
          ❌ {error}
        </div>
      )}

      {trainingStatus && (
        <div className="training-result">
          <h3>🎉 Training Started Successfully!</h3>
          <div className="training-info">
            <p><strong>Model Name:</strong> {trainingStatus.model_name}</p>
            <p><strong>Images:</strong> {trainingStatus.num_images}</p>
            <p><strong>Epochs:</strong> {trainingStatus.num_epochs}</p>
            <p><strong>Learning Rate:</strong> {trainingStatus.learning_rate}</p>
            <p><strong>Status:</strong> {trainingStatus.status}</p>
            <p><strong>Started:</strong> {new Date(trainingStatus.timestamp).toLocaleString()}</p>
          </div>

          {trainingStatus.model_path && (
            <div className="model-download">
              <p>Model will be saved to: {trainingStatus.model_path}</p>
              <p>You can use this model in future video generations for consistent results.</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default LoRATrainer
