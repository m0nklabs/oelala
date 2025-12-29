import React, { useMemo, useRef, useState } from 'react'
import { Upload, Layers, Settings, Zap, X } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm } from '../../api'
import { sendClientLog } from '../../logging'

export default function LoRATrainingTool({ onOutput }) {
  const fileInputRef = useRef(null)

  const [files, setFiles] = useState([])
  const [modelName, setModelName] = useState('')
  const [numEpochs, setNumEpochs] = useState(10)
  const [learningRate, setLearningRate] = useState(1e-4)

  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const canSubmit = useMemo(() => files.length > 0 && modelName.trim().length > 0 && !busy, [files, modelName, busy])

  const handlePickFiles = (picked) => {
    const arr = Array.from(picked || [])
    setFiles(arr)
    setError('')
  }

  const clearFiles = () => {
    setFiles([])
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const handleSubmit = async () => {
    if (files.length === 0) {
      setError('At least one image is required')
      return
    }
    if (!modelName.trim()) {
      setError('Model name is required')
      return
    }

    setBusy(true)
    setError('')

    const formData = new FormData()
    files.forEach((f) => formData.append('files', f))
    formData.append('model_name', modelName.trim())
    formData.append('num_epochs', String(numEpochs))
    formData.append('learning_rate', String(learningRate))

    try {
      if (DEBUG) console.debug('🐛 submit train-lora', { files: files.length, numEpochs, learningRate })
      const result = await postForm(`${BACKEND_BASE}/train-lora`, formData)
      if (!result.ok) {
        setError(result.data?.detail || `Training failed (status ${result.status})`)
        return
      }

      onOutput({ kind: 'lora', ...result.data })
    } catch (e) {
      const message = e?.message || 'Failed to start LoRA training'
      setError(message)
      await sendClientLog({
        level: 'error',
        message: 'LoRA training failed',
        timestamp: new Date().toISOString(),
        meta: { message },
      })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="tool-container">
      {/* Dataset Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Training Dataset</div>
          <Layers size={16} className="text-muted" />
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          onChange={(e) => handlePickFiles(e.target.files)}
          style={{ display: 'none' }}
        />

        {files.length === 0 ? (
          <div className="upload-box" onClick={() => fileInputRef.current?.click()} style={{ cursor: 'pointer' }}>
            <Upload size={32} className="text-muted" />
            <div className="text-muted">Upload training images (5-20 recommended)</div>
            <button className="upload-btn">
              <Upload size={16} />
              Select Images
            </button>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{files.length} images selected</span>
              <button 
                onClick={clearFiles}
                className="upload-btn secondary"
                style={{ padding: '4px 8px', fontSize: '0.8rem' }}
              >
                <X size={14} /> Clear
              </button>
            </div>
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fill, minmax(60px, 1fr))', 
              gap: '8px',
              maxHeight: '200px',
              overflowY: 'auto',
              padding: '8px',
              backgroundColor: '#0f0f0f',
              borderRadius: '8px',
              border: '1px solid var(--border-color)'
            }}>
              {files.map((f, i) => (
                <div key={i} style={{ aspectRatio: '1/1', backgroundColor: '#222', borderRadius: '4px', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <span style={{ fontSize: '0.6rem', color: '#666' }}>IMG</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Configuration Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Configuration</div>
          <Settings size={16} className="text-muted" />
        </div>

        <div className="form-group">
          <label className="grok-section-label">Model Name</label>
          <input 
            className="form-input" 
            value={modelName} 
            onChange={(e) => setModelName(e.target.value)} 
            placeholder="e.g. my-style-v1"
            style={{ backgroundColor: '#0f0f0f' }}
          />
        </div>

        <div className="form-group">
          <label className="grok-section-label">Training Epochs ({numEpochs})</label>
          <input 
            type="range" 
            min="5" 
            max="50" 
            step="5" 
            value={numEpochs} 
            onChange={(e) => setNumEpochs(parseInt(e.target.value, 10))}
            style={{ width: '100%', accentColor: 'var(--text-primary)' }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            <span>Fast (5)</span>
            <span>Quality (50)</span>
          </div>
        </div>

        <div className="form-group">
          <label className="grok-section-label">Learning Rate</label>
          <input
            className="form-input"
            type="number"
            step="0.00001"
            value={learningRate}
            onChange={(e) => setLearningRate(parseFloat(e.target.value || '0'))}
            style={{ backgroundColor: '#0f0f0f' }}
          />
        </div>
      </div>

      {error && (
        <div style={{ 
          padding: '12px', 
          backgroundColor: 'rgba(239, 68, 68, 0.1)', 
          border: '1px solid rgba(239, 68, 68, 0.2)', 
          borderRadius: '8px', 
          color: '#ef4444',
          marginBottom: '16px',
          fontSize: '0.9rem'
        }}>
          {error}
        </div>
      )}

      <button 
        className="primary-btn" 
        disabled={!canSubmit} 
        onClick={handleSubmit}
        style={{ 
          height: '48px', 
          fontSize: '1rem', 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center', 
          gap: '8px' 
        }}
      >
        {busy ? (
          <>Training...</>
        ) : (
          <>
            <Zap size={18} />
            Start Training
          </>
        )}
      </button>
    </div>
  )
}
