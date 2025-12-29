import React, { useRef, useState } from 'react'
import { Sparkles, Settings2, Image as ImageIcon, Info } from 'lucide-react'
import { BACKEND_BASE } from '../../config'
import { postForm } from '../../api'

const MODEL_OPTIONS = [
  { value: 'sd3.5-large-int8', label: 'SD3.5 Large (INT8, multi-GPU)' },
  { value: 'realvisxl-v5.0', label: 'RealVisXL V5.0 (FP16, 1xGPU)' },
]

export default function TextToImageTool({ onOutput }) {
  const [prompt, setPrompt] = useState('')
  const [aspectRatio, setAspectRatio] = useState('1:1')
  const [mode, setMode] = useState('normal')
  const [model, setModel] = useState('sd3.5-large-int8')
  const [batchCount, setBatchCount] = useState(1)
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState('')
  const [progress, setProgress] = useState(0)
  const pollerRef = useRef(null)

  const handleGenerate = async () => {
    if (!prompt.trim()) return
    setIsGenerating(true)
    setProgress(0)
    setError('')

    try {
      for (let i = 0; i < batchCount; i++) {
        const jobId = `t2i-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
        const formData = new FormData()
        formData.append('prompt', prompt)
        formData.append('aspect_ratio', aspectRatio)
        formData.append('mode', mode)
        formData.append('model', model)
        formData.append('job_id', jobId)

        // Start polling backend progress for this job
        if (pollerRef.current) {
          clearInterval(pollerRef.current)
          pollerRef.current = null
        }
        pollerRef.current = setInterval(async () => {
          try {
            const res = await fetch(`${BACKEND_BASE}/progress/${jobId}`)
            if (!res.ok) return
            const data = await res.json()
            if (typeof data.progress === 'number') setProgress(data.progress)
            if (data.status === 'done' || data.status === 'failed') {
              clearInterval(pollerRef.current)
              pollerRef.current = null
            }
          } catch (e) {
            // Swallow polling errors
          }
        }, 1000)
        
        // Add batch index to filename hint if needed, but backend handles timestamps
        
        const result = await postForm(`${BACKEND_BASE}/generate-image`, formData)
        if (!result.ok) {
          throw new Error(result.data?.detail || `Generation failed (status ${result.status})`)
        }

        console.log(`Batch ${i+1}/${batchCount} success:`, result.data)

        const imageUrl = result.data?.url
        const url = imageUrl ? `${BACKEND_BASE}${imageUrl}` : ''

        setProgress(100)

        onOutput({
          kind: 'image',
          url,
          backendUrl: url,
          filename: result.data?.filename,
          meta: result.data?.meta,
        })
      }
    } catch (e) {
      console.error('Generation error:', e)
      setError(e.message || 'Failed to generate image')
    } finally {
      if (pollerRef.current) {
        clearInterval(pollerRef.current)
        pollerRef.current = null
      }
      setIsGenerating(false)
      setTimeout(() => setProgress(0), 500)
    }
  }

  return (
    <div className="tool-container">
      {/* Mode Selection */}
      <div className="grok-card">
        <div className="form-group">
          <label className="grok-section-label">Mode</label>
          <div className="form-select" style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <Sparkles size={16} className="text-primary" />
            <span>Normal</span>
          </div>
          <div className="info-badge">
            <span style={{ color: '#93c5fd' }}>Standard Quality</span>
            <div style={{ marginTop: '4px', opacity: 0.8 }}>Fast and efficient image generation (1 credit per image)</div>
          </div>
        </div>
      </div>

      {/* Prompt Input */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Enter Image Prompt</div>
          <div style={{ display: 'flex', gap: '4px' }}>
             {/* Mock icons for prompt tools */}
             <button className="icon-btn" style={{ width: '24px', height: '24px', fontSize: '10px' }}>T</button>
             <button className="icon-btn" style={{ width: '24px', height: '24px', fontSize: '10px' }}>✨</button>
          </div>
        </div>
        
        <div style={{ position: 'relative' }}>
          <textarea
            className="form-textarea"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={4}
            placeholder="A attractive blonde woman with cup f, tattoos, looking at me defiantly."
            style={{ 
              backgroundColor: '#0f0f0f', 
              border: 'none', 
              resize: 'none',
              paddingBottom: '24px'
            }}
          />
        </div>
      </div>

      {/* Model Selection */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Model</div>
          <Info size={16} className="text-muted" />
        </div>
        <div className="grok-toggle-group" style={{ flexWrap: 'wrap' }}>
          {MODEL_OPTIONS.map((option) => (
            <button
              key={option.value}
              className={`grok-toggle-btn ${model === option.value ? 'active' : ''}`}
              onClick={() => setModel(option.value)}
              style={{ minWidth: '180px', marginBottom: '8px' }}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      {/* Aspect Ratio */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Aspect Ratio</div>
        </div>
        <div className="aspect-grid" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>
          {[
            { label: '1:1', icon: <div style={{ width: '18px', height: '18px', border: '1px solid currentColor' }} /> },
            { label: '16:9', icon: <div style={{ width: '24px', height: '14px', border: '1px solid currentColor' }} /> },
            { label: '9:16', icon: <div style={{ width: '14px', height: '24px', border: '1px solid currentColor' }} /> },
            { label: '4:3', icon: <div style={{ width: '20px', height: '15px', border: '1px solid currentColor' }} /> },
            { label: '3:4', icon: <div style={{ width: '15px', height: '20px', border: '1px solid currentColor' }} /> },
            { label: '2:3', icon: <div style={{ width: '16px', height: '24px', border: '1px solid currentColor' }} /> },
            { label: '3:2', icon: <div style={{ width: '24px', height: '16px', border: '1px solid currentColor' }} /> },
            { label: '4:5', icon: <div style={{ width: '16px', height: '20px', border: '1px solid currentColor' }} /> },
            { label: '5:4', icon: <div style={{ width: '20px', height: '16px', border: '1px solid currentColor' }} /> },
            { label: '9:21', icon: <div style={{ width: '10px', height: '24px', border: '1px solid currentColor' }} /> },
            { label: '21:9', icon: <div style={{ width: '24px', height: '10px', border: '1px solid currentColor' }} /> },
          ].map((ratio) => (
            <button
              key={ratio.label}
              className={`aspect-btn ${aspectRatio === ratio.label ? 'active' : ''}`}
              onClick={() => setAspectRatio(ratio.label)}
              style={{ height: '60px' }}
            >
              <div className="aspect-icon" style={{ background: 'transparent', display: 'flex', alignItems: 'center', justifyContent: 'center', border: 'none', marginBottom: '4px' }}>
                {ratio.icon}
              </div>
              <span className="aspect-label" style={{ fontSize: '0.65rem' }}>{ratio.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Advanced Settings */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Advanced Settings</div>
          <Settings2 size={16} className="text-muted" />
        </div>
        
        <div className="form-group">
          <label className="grok-section-label">Batch Count (Images)</label>
          <div className="grok-toggle-group">
            {[1, 2, 3, 4].map((num) => (
              <button
                key={num}
                className={`grok-toggle-btn ${batchCount === num ? 'active' : ''}`}
                onClick={() => setBatchCount(num)}
              >
                {num}
              </button>
            ))}
          </div>
        </div>
      </div>

      {error && (
        <div style={{ color: '#ef4444', marginBottom: '12px', fontSize: '0.9rem' }}>
          {error}
        </div>
      )}

      <button 
        className="primary-btn" 
        onClick={handleGenerate}
        disabled={isGenerating || !prompt.trim()}
        style={{ 
          height: '48px', 
          fontSize: '1rem', 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center', 
          gap: '8px',
          backgroundColor: 'white',
          color: 'black'
        }}
      >
        {isGenerating ? (
          <>Generating...</>
        ) : (
          <>
            <Sparkles size={18} />
            Generate {batchCount > 1 ? `${batchCount} Images` : 'Image'} ({batchCount})
          </>
        )}
      </button>

      {isGenerating && (
        <div className="progress-container">
          <div className="progress-fill" style={{ width: `${Math.min(progress, 100)}%` }}></div>
        </div>
      )}
    </div>
  )
}
