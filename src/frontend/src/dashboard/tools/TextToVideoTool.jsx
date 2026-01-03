import React, { useMemo, useState } from 'react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm } from '../../api'
import { sendClientLog } from '../../logging'
import { Settings, Wand2 } from 'lucide-react'

// Resolution presets with pixel dimensions
const RESOLUTION_PRESETS = [
  { value: '480p', label: '480p', width: 854, height: 480 },
  { value: '720p', label: '720p', width: 1280, height: 720 },
  { value: '1080p', label: '1080p', width: 1920, height: 1080 },
]

const FPS_OPTIONS = [8, 12, 16, 24]

export default function TextToVideoTool({ onOutput, onRefreshHistory }) {
  const [prompt, setPrompt] = useState('')
  const [numFrames, setNumFrames] = useState(16)
  const [modelType, setModelType] = useState('light')
  const [outputFilename, setOutputFilename] = useState('')
  const [aspectRatio, setAspectRatio] = useState('16:9')
  const [resolution, setResolution] = useState('720p')
  const [fps, setFps] = useState(16)

  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const canSubmit = useMemo(() => prompt.trim().length > 0 && !busy, [prompt, busy])

  const handleSubmit = async () => {
    if (!prompt.trim()) {
      setError('Prompt is required')
      return
    }

    setBusy(true)
    setError('')

    const formData = new FormData()
    formData.append('prompt', prompt)
    formData.append('num_frames', String(numFrames))
    formData.append('model_type', modelType)
    formData.append('aspect_ratio', aspectRatio)
    formData.append('resolution', resolution)
    formData.append('fps', String(fps))
    if (outputFilename.trim()) formData.append('output_filename', outputFilename.trim())

    try {
      if (DEBUG) console.debug('🐛 submit text-to-video', { numFrames, modelType, resolution, fps })
      const result = await postForm(`${BACKEND_BASE}/generate-text`, formData)
      if (!result.ok) {
        setError(result.data?.detail || `Generation failed (status ${result.status})`)
        return
      }

      const videoUrl = result.data?.video_url
      const outputVideo = result.data?.output_video
      const url = videoUrl ? `${BACKEND_BASE}${videoUrl}` : ''

      onOutput({
        kind: 'video',
        url,
        backendUrl: url,
        filename: outputVideo,
        meta: result.data,
      })
      onRefreshHistory()
    } catch (e) {
      const message = e?.message || 'Failed to generate video'
      // Check for network/fetch errors vs backend errors
      if (message.includes('NetworkError') || message.includes('fetch')) {
        setError('Connection error - backend may be down or endpoint not available')
      } else {
        setError(message)
      }
      await sendClientLog({
        level: 'error',
        message: 'Text-to-video failed',
        timestamp: new Date().toISOString(),
        meta: { message },
      })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="tool-container">
      {/* Mode Selection Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <span className="grok-card-title">Mode Selection</span>
          <button className="btn ghost" style={{ padding: '6px 12px', fontSize: '0.8rem' }}>
            <Settings size={14} style={{ marginRight: 6 }} /> Advanced Settings
          </button>
        </div>
        
        <div className="field">
          <label className="grok-section-label">Mode</label>
          <select className="select" value={modelType} onChange={(e) => setModelType(e.target.value)}>
            <option value="light">🚀 Turbo (Light)</option>
            <option value="svd">⚡ Standard (SVD)</option>
            <option value="wan2.2">🌟 Quality (Wan2.2)</option>
          </select>
          <div className="muted" style={{ marginTop: 8, fontSize: '0.8rem' }}>
            {modelType === 'light' && 'Fastest generation speed. Good for quick previews.'}
            {modelType === 'svd' && 'Balanced quality and speed using Stable Video Diffusion.'}
            {modelType === 'wan2.2' && 'Highest quality video generation with Wan2.2 model.'}
          </div>
        </div>
      </div>

      {/* Prompt Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <span className="grok-card-title">Enter Video Prompt</span>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="icon-btn" title="Enhance Prompt"><Wand2 size={14} /></button>
          </div>
        </div>
        <textarea
          className="textarea"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={5}
          placeholder="Describe the video you want to generate..."
          style={{ backgroundColor: '#0f0f0f', border: 'none', resize: 'none' }}
        />
      </div>

      {/* Settings Card */}
      <div className="grok-card">
        {/* Resolution Presets */}
        <div className="field" style={{ marginBottom: 24 }}>
          <div className="grok-card-header">
            <span className="grok-card-title">Resolution</span>
          </div>
          <div className="grok-toggle-group" style={{ marginBottom: 16 }}>
            {RESOLUTION_PRESETS.map((preset) => (
              <button
                key={preset.value}
                className={`grok-toggle-btn ${resolution === preset.value ? 'active' : ''}`}
                onClick={() => setResolution(preset.value)}
                type="button"
              >
                {preset.label}
              </button>
            ))}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {RESOLUTION_PRESETS.find(p => p.value === resolution)?.width}×{RESOLUTION_PRESETS.find(p => p.value === resolution)?.height}px
          </div>
        </div>

        {/* Aspect Ratio */}
        <div className="field" style={{ marginBottom: 24 }}>
          <div className="grok-card-header">
            <span className="grok-card-title">Aspect Ratio</span>
          </div>
          <div className="aspect-grid">
            {['16:9', '9:16', '1:1', '4:3', '3:4', '21:9'].map((ratio) => (
              <button
                key={ratio}
                className={`aspect-btn ${aspectRatio === ratio ? 'active' : ''}`}
                onClick={() => setAspectRatio(ratio)}
                type="button"
              >
                <div className="aspect-icon" style={{ 
                  width: ratio === '16:9' ? 32 : ratio === '9:16' ? 18 : ratio === '21:9' ? 36 : 24,
                  height: ratio === '16:9' ? 18 : ratio === '9:16' ? 32 : ratio === '21:9' ? 15 : 24
                }} />
                <span className="aspect-label">{ratio}</span>
              </button>
            ))}
          </div>
        </div>

        {/* FPS Control */}
        <div className="field" style={{ marginBottom: 24 }}>
          <div className="grok-card-header">
            <span className="grok-card-title">Frame Rate (FPS)</span>
            <span className="status-pill">{fps} fps</span>
          </div>
          <div className="grok-toggle-group">
            {FPS_OPTIONS.map((f) => (
              <button
                key={f}
                className={`grok-toggle-btn ${fps === f ? 'active' : ''}`}
                onClick={() => setFps(f)}
                type="button"
              >
                {f}
              </button>
            ))}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 8 }}>
            Higher FPS = smoother motion, more VRAM required
          </div>
        </div>

        {/* Duration */}
        <div className="field">
          <div className="grok-card-header">
            <span className="grok-card-title">Duration</span>
            <span className="status-pill">{(numFrames / fps).toFixed(1)}s ({numFrames}f)</span>
          </div>
          <input
            className="range"
            type="range"
            min="8"
            max="48"
            step="4"
            value={numFrames}
            onChange={(e) => setNumFrames(parseInt(e.target.value, 10))}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8, fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            <span>{(8 / fps).toFixed(1)}s (8f)</span>
            <span>{(48 / fps).toFixed(1)}s (48f)</span>
          </div>
        </div>
      </div>

      <div className="field">
        <label className="label">Output filename (optional)</label>
        <input
          className="input"
          value={outputFilename}
          onChange={(e) => setOutputFilename(e.target.value)}
          placeholder="my_video.mp4"
        />
      </div>

      {error && <div className="error">{error}</div>}

      <button className="primary-btn" type="button" disabled={!canSubmit} onClick={handleSubmit}>
        {busy ? 'Generating Video...' : 'Generate Video'}
      </button>
    </div>
  )
}
