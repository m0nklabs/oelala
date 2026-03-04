import React, { useState, useCallback, useMemo, useEffect } from 'react'
import { Upload, ZoomIn, Loader2, Settings, ChevronDown } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm } from '../../api'
import { useAuth } from '../../contexts/AuthContext'
import { useToolSettings } from '../../hooks/useToolSettings'
import ResetDefaultsButton from '../../components/ResetDefaultsButton'

// Video upscaling models
const UPSCALE_MODELS = [
  { value: 'realesrgan-video', label: 'Real-ESRGAN Video', desc: 'AI-enhanced video upscaling', scale: [2, 4] },
  { value: 'basic-lanczos', label: 'Basic Lanczos', desc: 'Fast traditional upscaling', scale: [2, 4] },
]

const VU_DEFAULTS = { model: 'realesrgan-video' }

export default function VideoUpscalerTool({ onOutput, onJobSubmitted }) {
  const { user, requestLogin } = useAuth()
  const { initial, save: saveSettings, resetDefaults } = useToolSettings('video_upscaler', VU_DEFAULTS)

  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [videoInfo, setVideoInfo] = useState(null)

  const [model, setModel] = useState(initial.model)

  // Auto-save settings
  const settingsSnapshot = useMemo(() => ({ model }), [model])
  useEffect(() => { saveSettings(settingsSnapshot) }, [settingsSnapshot, saveSettings])

  const handleResetDefaults = useCallback(() => {
    const d = resetDefaults()
    setModel(d.model)
  }, [resetDefaults])

  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)
  const [lastQueued, setLastQueued] = useState(null)

  const handleFileChange = useCallback((e) => {
    const f = e.target.files?.[0]
    if (f) {
      setFile(f)
      const url = URL.createObjectURL(f)
      setPreview(url)
      setResult(null)
      setError(null)
      setLastQueued(null)

      // Get video info
      const video = document.createElement('video')
      video.onloadedmetadata = () => {
        setVideoInfo({
          duration: video.duration.toFixed(1),
          width: video.videoWidth,
          height: video.videoHeight,
        })
      }
      video.src = url
    }
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0]
    if (f && f.type.startsWith('video/')) {
      setFile(f)
      const url = URL.createObjectURL(f)
      setPreview(url)
      setError(null)
      setLastQueued(null)

      const video = document.createElement('video')
      video.onloadedmetadata = () => {
        setVideoInfo({
          duration: video.duration.toFixed(1),
          width: video.videoWidth,
          height: video.videoHeight,
        })
      }
      video.src = url
    }
  }, [])

  const handleUpscale = async () => {
    // Check if user is logged in
    if (!user) {
      requestLogin('Log in om te genereren')
      return
    }

    if (!file) return

    setSubmitting(true)
    setError(null)
    setLastQueued(null)

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('model', model)

      if (DEBUG) console.debug('🔍 Video upscale request:', { model })

      const res = await postForm(`${BACKEND_BASE}/upscale-video`, formData)

      if (!res.ok) {
        throw new Error(res.data?.detail || 'Video upscaling failed')
      }

      const promptId = res.data?.prompt_id
      if (!promptId) {
        throw new Error('No prompt_id returned')
      }

      // Show queued confirmation
      setLastQueued({
        promptId,
        model: UPSCALE_MODELS.find(m => m.value === model)?.label || model,
      })

      if (DEBUG) console.debug('📋 Video upscale queued:', promptId)

      if (onJobSubmitted) onJobSubmitted(promptId)
    } catch (err) {
      console.error('Video upscale error:', err)
      setError(err.message || 'Failed to upscale video')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="tool-container">
      {/* Upload Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <ZoomIn size={16} />
            Video Upscaler
          </div>
          <ResetDefaultsButton onReset={handleResetDefaults} />
        </div>

        <div
          className="upload-box"
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => document.getElementById('video-upscale-file')?.click()}
          style={{ cursor: 'pointer' }}
        >
          {preview ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px', width: '100%' }}>
              <video
                src={preview}
                controls
                muted
                style={{ maxHeight: '180px', borderRadius: '8px', maxWidth: '100%' }}
              />
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{file?.name}</span>
              {videoInfo && (
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', gap: '12px' }}>
                  <span>📐 {videoInfo.width}×{videoInfo.height}</span>
                  <span>⏱️ {videoInfo.duration}s</span>
                </div>
              )}
            </div>
          ) : (
            <>
              <Upload size={32} className="text-muted" />
              <div className="text-muted">Drop video here, or click to upload</div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>MP4, WebM, MOV</span>
            </>
          )}
          <input
            id="video-upscale-file"
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
        </div>
      </div>

      {/* Settings Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Settings</div>
        </div>

        <div className="form-group">
          <label className="grok-section-label">Upscale Model</label>
          <div className="grok-toggle-group" style={{ flexDirection: 'column' }}>
            {UPSCALE_MODELS.map(m => (
              <button
                key={m.value}
                onClick={() => setModel(m.value)}
                className={`grok-toggle-btn ${model === m.value ? 'active' : ''}`}
                style={{ textAlign: 'left', padding: '10px 12px' }}
              >
                <div style={{ fontWeight: 500 }}>{m.label}</div>
                <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>{m.desc}</div>
              </button>
            ))}
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '8px' }}>
            Currently uses fixed 4x upscaling. Custom resolution settings coming soon.
          </div>
        </div>
      </div>

      {error && <div className="status-banner error">{error}</div>}

      {/* Generate Button */}
      <button
        className="primary-btn"
        onClick={handleUpscale}
        disabled={!file || submitting}
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', height: '48px', fontSize: '1rem' }}
      >
        {submitting ? (
          <>
            <Loader2 size={18} className="animate-spin" />
            Upscaling...
          </>
        ) : (
          <>
            <ZoomIn size={18} />
            Upscale Video
          </>
        )}
      </button>

      {lastQueued && (
        <div className="status-banner success">
          ✅ Video upscale queued! ({lastQueued.model}) — Check queue panel for progress
        </div>
      )}
    </div>
  )
}
