import React, { useState, useCallback } from 'react'
import { Upload, ZoomIn, Loader2, Settings, ChevronDown } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm } from '../../api'

// Video upscaling models
const UPSCALE_MODELS = [
  { value: 'realesrgan-video', label: 'Real-ESRGAN Video', desc: 'AI-enhanced video upscaling', scale: [2, 4] },
  { value: 'basic-lanczos', label: 'Basic Lanczos', desc: 'Fast traditional upscaling', scale: [2, 4] },
]

// Resolution presets
const RESOLUTION_PRESETS = [
  { from: '480p', to: '720p', scale: 1.5, label: '480p → 720p' },
  { from: '480p', to: '1080p', scale: 2.25, label: '480p → 1080p' },
  { from: '720p', to: '1080p', scale: 1.5, label: '720p → 1080p' },
  { from: '720p', to: '4K', scale: 3, label: '720p → 4K' },
  { from: '1080p', to: '4K', scale: 2, label: '1080p → 4K' },
]

// Quality vs Speed presets
const QUALITY_PRESETS = [
  { value: 'fast', label: 'Fast', desc: 'Lower quality, faster processing', denoise: 0.3 },
  { value: 'balanced', label: 'Balanced', desc: 'Good quality & speed', denoise: 0.5 },
  { value: 'quality', label: 'Quality', desc: 'Best quality, slower', denoise: 0.7 },
]

export default function VideoUpscalerTool({ onOutput, onJobSubmitted }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [videoInfo, setVideoInfo] = useState(null)

  const [model, setModel] = useState('realesrgan-video')

  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)
  const [lastQueued, setLastQueued] = useState(null)
  const [result, setResult] = useState(null)

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
      setResult(null)
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
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', padding: '20px' }}>
      <div style={{ marginBottom: '8px' }}>
        <h2 style={{ fontSize: '1.3rem', fontWeight: 600, marginBottom: '4px' }}>Video Upscaler</h2>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          AI-enhanced video upscaling • 480p → 720p → 1080p → 4K
        </p>
      </div>

      {/* Upload Section */}
      <div
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        style={{
          border: '2px dashed var(--border-color)',
          borderRadius: '8px',
          padding: '24px',
          textAlign: 'center',
          cursor: 'pointer',
          transition: 'all 0.2s',
        }}
        onClick={() => document.getElementById('video-upscale-file')?.click()}
      >
        <Upload size={32} style={{ margin: '0 auto 12px', color: 'var(--text-muted)' }} />
        <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>
          {file ? file.name : 'Drop video or click to upload'}
        </p>
        {videoInfo && (
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {videoInfo.width}×{videoInfo.height} • {videoInfo.duration}s
          </p>
        )}
        <input
          id="video-upscale-file"
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
      </div>

      {/* Preview */}
      {preview && (
        <div style={{ borderRadius: '8px', overflow: 'hidden', maxWidth: '100%' }}>
          <video
            src={preview}
            controls
            style={{ width: '100%', maxHeight: '400px', display: 'block' }}
          />
        </div>
      )}

      {/* Settings */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {/* Model Selection */}
        <div>
          <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '6px', color: 'var(--text-secondary)' }}>
            Upscale Model
          </label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            style={{
              width: '100%',
              padding: '8px 12px',
              borderRadius: '6px',
              border: '1px solid var(--border-color)',
              background: 'var(--bg-secondary)',
              color: 'var(--text-primary)',
              fontSize: '0.9rem',
            }}
          >
            {UPSCALE_MODELS.map(m => (
              <option key={m.value} value={m.value}>
                {m.label} - {m.desc}
              </option>
            ))}
          </select>
          <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Note: Currently uses fixed 4x upscaling with RealESRGAN. Custom resolution and quality settings coming soon.
          </p>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div style={{ padding: '12px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '6px' }}>
          <p style={{ fontSize: '0.85rem', color: '#ef4444' }}>{error}</p>
        </div>
      )}

      {/* Queued Confirmation */}
      {lastQueued && (
        <div style={{ padding: '12px', background: 'rgba(34, 197, 94, 0.1)', border: '1px solid rgba(34, 197, 94, 0.3)', borderRadius: '6px' }}>
          <p style={{ fontSize: '0.85rem', color: '#22c55e' }}>
            ✓ Video upscale queued! ({lastQueued.model}, {lastQueued.preset})
          </p>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Job ID: {lastQueued.promptId}
          </p>
        </div>
      )}

      {/* Generate Button */}
      <button
        onClick={handleUpscale}
        disabled={!file || submitting}
        style={{
          padding: '14px',
          borderRadius: '8px',
          border: 'none',
          background: !file || submitting ? 'var(--bg-tertiary)' : 'var(--accent-color)',
          color: !file || submitting ? 'var(--text-muted)' : 'white',
          fontSize: '1rem',
          fontWeight: 600,
          cursor: !file || submitting ? 'not-allowed' : 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '8px',
          transition: 'all 0.2s',
        }}
      >
        {submitting ? (
          <>
            <Loader2 size={20} style={{ animation: 'spin 1s linear infinite' }} />
            Upscaling...
          </>
        ) : (
          <>
            <ZoomIn size={20} />
            Upscale Video
          </>
        )}
      </button>

      {/* Result */}
      {result && (
        <div>
          <h3 style={{ fontSize: '1rem', marginBottom: '8px' }}>
            Result ({RESOLUTION_PRESETS.find(p => p.label === resolutionPreset)?.label})
          </h3>
          <div style={{ borderRadius: '8px', overflow: 'hidden' }}>
            <video src={result} controls style={{ width: '100%', display: 'block' }} />
          </div>
        </div>
      )}
    </div>
  )
}
