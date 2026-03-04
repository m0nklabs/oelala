import React, { useState, useCallback, useRef, useMemo, useEffect } from 'react'
import { Upload, ZoomIn, Zap, Film, Loader2, Trash2, Plus, GripVertical, Play, Settings } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm } from '../../api'
import { useAuth } from '../../contexts/AuthContext'
import MyMediaTool from './MyMediaTool'
import { useToolSettings } from '../../hooks/useToolSettings'
import ResetDefaultsButton from '../../components/ResetDefaultsButton'

// Processing modes
const PROCESSING_MODES = {
  upscale: {
    id: 'upscale',
    label: '📈 Upscale Video',
    desc: 'Increase video resolution using AI',
    icon: ZoomIn,
  },
  interpolate: {
    id: 'interpolate',
    label: '🔄 Frame Interpolation',
    desc: 'Smooth motion by adding frames (RIFE)',
    icon: Zap,
  },
  concat: {
    id: 'concat',
    label: '🎬 Video Montage',
    desc: 'Join multiple videos together',
    icon: Film,
  },
}

// Upscale models
const UPSCALE_MODELS = [
  { value: 'realesrgan-x4plus', label: 'Real-ESRGAN x4+', desc: 'Best quality, slower' },
  { value: 'realesrgan-video', label: 'Real-ESRGAN Video', desc: 'Optimized for video' },
]

// FPS presets for interpolation
const FPS_PRESETS = [
  { value: 30, label: '30 fps', multiplier: 2 },
  { value: 60, label: '60 fps', multiplier: 4 },
  { value: 120, label: '120 fps (slow-mo)', multiplier: 8 },
]

const PP_DEFAULTS = { mode: 'upscale', upscaleModel: 'realesrgan-x4plus', upscaleScale: 2, targetFps: 60 }

export default function PostProcessingTool({ onOutput, onJobSubmitted }) {
  const { user, requestLogin } = useAuth()
  const { initial, save: saveSettings, resetDefaults } = useToolSettings('post_processing', PP_DEFAULTS)

  // Mode selection
  const [mode, setMode] = useState(initial.mode)

  // File management
  const [files, setFiles] = useState([])
  const [showMediaPicker, setShowMediaPicker] = useState(false)
  const fileInputRef = useRef(null)

  // Upscale settings
  const [upscaleModel, setUpscaleModel] = useState(initial.upscaleModel)
  const [upscaleScale, setUpscaleScale] = useState(initial.upscaleScale)

  // Interpolation settings
  const [targetFps, setTargetFps] = useState(initial.targetFps)

  // Auto-save settings
  const settingsSnapshot = useMemo(() => ({ mode, upscaleModel, upscaleScale, targetFps }), [mode, upscaleModel, upscaleScale, targetFps])
  useEffect(() => { saveSettings(settingsSnapshot) }, [settingsSnapshot, saveSettings])

  const handleResetDefaults = useCallback(() => {
    const d = resetDefaults()
    setMode(d.mode); setUpscaleModel(d.upscaleModel); setUpscaleScale(d.upscaleScale); setTargetFps(d.targetFps)
  }, [resetDefaults])

  // Processing state
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)
  const [lastQueued, setLastQueued] = useState(null)

  // Handle file selection
  const handleFileChange = useCallback((e) => {
    const newFiles = Array.from(e.target.files || [])
    addFiles(newFiles)
  }, [])

  const addFiles = useCallback((newFiles) => {
    const videoFiles = newFiles.filter(f => f.type.startsWith('video/'))

    const fileEntries = videoFiles.map(file => {
      const url = URL.createObjectURL(file)
      return {
        id: Date.now() + Math.random(),
        file,
        preview: url,
        videoInfo: null,
      }
    })

    // Get video info for each file
    fileEntries.forEach(entry => {
      const video = document.createElement('video')
      video.onloadedmetadata = () => {
        entry.videoInfo = {
          duration: video.duration.toFixed(1),
          width: video.videoWidth,
          height: video.videoHeight,
        }
        setFiles(prev => [...prev]) // Force re-render
      }
      video.src = entry.preview
    })

    setFiles(prev => [...prev, ...fileEntries])
    setError(null)
    setLastQueued(null)
  }, [])

  // Handle drag and drop
  const handleDrop = useCallback((e) => {
    e.preventDefault()
    const droppedFiles = Array.from(e.dataTransfer.files || [])
    addFiles(droppedFiles)
  }, [addFiles])

  // Remove file
  const removeFile = useCallback((id) => {
    setFiles(prev => {
      const file = prev.find(f => f.id === id)
      if (file?.preview) URL.revokeObjectURL(file.preview)
      return prev.filter(f => f.id !== id)
    })
  }, [])

  // Reorder files (for concat mode)
  const moveFile = useCallback((fromIndex, toIndex) => {
    setFiles(prev => {
      const newFiles = [...prev]
      const [removed] = newFiles.splice(fromIndex, 1)
      newFiles.splice(toIndex, 0, removed)
      return newFiles
    })
  }, [])

  // Handle media selection from picker
  const handleMediaSelect = useCallback((item) => {
    // Create a file entry from the media item
    const entry = {
      id: Date.now() + Math.random(),
      file: null, // Will use URL instead
      mediaUrl: item.url,
      signedUrl: item.signed_url,
      preview: item.signed_url || item.url,
      videoInfo: {
        duration: '?',
        width: '?',
        height: '?',
      },
      filename: item.filename,
    }
    setFiles(prev => [...prev, entry])
    setShowMediaPicker(false)
  }, [])

  // Submit processing job
  const handleSubmit = async () => {
    if (!user) {
      requestLogin('Log in to process media')
      return
    }

    if (files.length === 0) {
      setError('Please add at least one video')
      return
    }

    if (mode === 'concat' && files.length < 2) {
      setError('Please add at least 2 videos to concatenate')
      return
    }

    setSubmitting(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('mode', mode)

      // Add files or URLs
      for (let i = 0; i < files.length; i++) {
        const entry = files[i]
        if (entry.file) {
          formData.append('files', entry.file)
        } else if (entry.mediaUrl) {
          formData.append('media_urls', entry.mediaUrl)
        }
      }

      // Mode-specific settings
      if (mode === 'upscale') {
        formData.append('model', upscaleModel)
        formData.append('scale', String(upscaleScale))
      } else if (mode === 'interpolate') {
        formData.append('target_fps', String(targetFps))
      }

      const endpoint = `${BACKEND_BASE}/post-process`
      const result = await postForm(endpoint, formData)

      if (!result.ok) {
        setError(result.data?.detail || `Processing failed (status ${result.status})`)
        return
      }

      if (DEBUG) console.debug('🐛 Post-processing job queued:', result.data)

      setLastQueued(result.data)
      if (onJobSubmitted) {
        onJobSubmitted(result.data)
      }
    } catch (err) {
      setError(`Processing failed: ${err.message}`)
    } finally {
      setSubmitting(false)
    }
  }

  // Calculate total duration for concat mode
  const totalDuration = files.reduce((sum, f) => {
    const d = parseFloat(f.videoInfo?.duration || 0)
    return sum + (isNaN(d) ? 0 : d)
  }, 0)

  return (
    <div className="tool-container">
      {/* Mode Selection Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Processing Mode</div>
          <ResetDefaultsButton onReset={handleResetDefaults} />
        </div>
      <div className="grok-mode-selector">
        {Object.values(PROCESSING_MODES).map(m => {
          const Icon = m.icon
          return (
            <button
              key={m.id}
              onClick={() => setMode(m.id)}
              className={`grok-mode-card ${mode === m.id ? 'active' : ''}`}
            >
              <Icon size={24} className="mode-icon" />
              <div className="mode-label">
                {m.label.replace(/^. /, '')}
              </div>
              <div className="mode-desc">
                {m.desc}
              </div>
            </button>
          )
        })}
      </div>
      </div>

      {/* File Upload Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">
            {mode === 'concat' ? 'Video Files' : 'Source Video'}
          </div>
        </div>
      <div
        className="upload-box"
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        onClick={() => fileInputRef.current?.click()}
        style={{ cursor: 'pointer', marginBottom: '16px' }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          multiple={mode === 'concat'}
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        <Upload size={32} style={{ color: 'var(--text-muted)' }} />
        <div style={{ color: 'var(--text-primary)', fontWeight: 500 }}>
          {mode === 'concat' ? 'Drop videos here or click to upload' : 'Drop a video here or click to upload'}
        </div>
        <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
          Or select from your media library
        </div>
      </div>

      {/* Select from Media Library */}
      <button
        onClick={() => setShowMediaPicker(true)}
        className="btn-secondary"
        style={{ width: '100%', marginBottom: '16px' }}
      >
        📁 Select from My Media
      </button>

      {/* Media Picker Modal */}
      {showMediaPicker && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.8)',
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '20px',
        }}>
          <div style={{
            backgroundColor: 'var(--bg-primary)',
            borderRadius: '12px',
            width: '100%',
            maxWidth: '1200px',
            maxHeight: '80vh',
            overflow: 'auto',
            position: 'relative',
          }}>
            <div style={{
              position: 'sticky',
              top: 0,
              padding: '16px',
              backgroundColor: 'var(--bg-primary)',
              borderBottom: '1px solid var(--border-color)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}>
              <h3>Select Video</h3>
              <button
                onClick={() => setShowMediaPicker(false)}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                  fontSize: '1.5rem',
                }}
              >
                ×
              </button>
            </div>
            <div style={{ padding: '16px' }}>
              <MyMediaTool
                filter="video"
                selectionMode={true}
                onSelectItem={handleMediaSelect}
              />
            </div>
          </div>
        </div>
      )}

      {/* File List */}
      {files.length > 0 && (
        <div style={{ marginBottom: '16px' }}>
          <label className="grok-section-label">
            {mode === 'concat' ? `Videos to Join (${files.length})` : 'Selected Video'}
            {mode === 'concat' && totalDuration > 0 && (
              <span style={{ fontWeight: 'normal', color: 'var(--text-muted)', marginLeft: '8px' }}>
                Total: {totalDuration.toFixed(1)}s
              </span>
            )}
          </label>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {files.map((entry, idx) => (
              <div key={entry.id} className="file-item-row">
                {mode === 'concat' && (
                  <GripVertical size={16} style={{ color: 'var(--text-muted)', cursor: 'grab' }} />
                )}

                <div className="file-thumb">
                  <video src={entry.preview} muted />
                </div>

                <div className="file-info">
                  <div className="file-name">
                    {entry.file?.name || entry.filename || 'Video'}
                  </div>
                  {entry.videoInfo && (
                    <div className="file-meta">
                      {entry.videoInfo.width}×{entry.videoInfo.height} • {entry.videoInfo.duration}s
                    </div>
                  )}
                </div>

                {mode === 'concat' && (
                  <div className="file-meta">#{idx + 1}</div>
                )}

                <button onClick={() => removeFile(entry.id)} className="file-remove">
                  <Trash2 size={16} />
                </button>
              </div>
            ))}
          </div>

          {mode === 'concat' && (
            <button
              onClick={() => fileInputRef.current?.click()}
              className="btn-secondary"
              style={{ marginTop: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}
            >
              <Plus size={16} /> Add more videos
            </button>
          )}
        </div>
      )}
      </div>

      {/* Settings Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Settings size={16} /> Settings
          </div>
        </div>

        {mode === 'upscale' && (
          <>
            <div className="form-group">
              <label className="grok-section-label">Upscale Model</label>
              <div className="grok-toggle-group">
                {UPSCALE_MODELS.map(m => (
                  <button
                    key={m.value}
                    onClick={() => setUpscaleModel(m.value)}
                    className={`grok-toggle-btn ${upscaleModel === m.value ? 'active' : ''}`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label className="grok-section-label">Scale Factor</label>
              <div className="grok-toggle-group">
                {[2, 4].map(s => (
                  <button
                    key={s}
                    onClick={() => setUpscaleScale(s)}
                    className={`grok-toggle-btn ${upscaleScale === s ? 'active' : ''}`}
                  >
                    {s}x
                  </button>
                ))}
              </div>
            </div>
          </>
        )}

        {mode === 'interpolate' && (
          <div className="form-group">
            <label className="grok-section-label">Target Frame Rate</label>
            <div className="grok-toggle-group">
              {FPS_PRESETS.map(p => (
                <button
                  key={p.value}
                  onClick={() => setTargetFps(p.value)}
                  className={`grok-toggle-btn ${targetFps === p.value ? 'active' : ''}`}
                >
                  {p.label}
                </button>
              ))}
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '8px' }}>
              Uses RIFE AI to generate intermediate frames for smoother motion
            </p>
          </div>
        )}

        {mode === 'concat' && (
          <p style={{ color: 'var(--text-muted)' }}>
            Videos will be joined in the order shown above. Drag to reorder.
            <br />
            <span style={{ fontSize: '0.85rem' }}>
              Note: All videos should have the same resolution for best results.
            </span>
          </p>
        )}
      </div>

      {error && <div className="status-banner error">{error}</div>}

      {lastQueued && (
        <div className="status-banner success">
          ✅ Job queued! Check the Jobs panel for progress.
        </div>
      )}

      <button
        onClick={handleSubmit}
        disabled={submitting || files.length === 0}
        className="primary-btn"
      >
        {submitting ? (
          <>
            <Loader2 size={20} className="animate-spin" /> Processing...
          </>
        ) : (
          <>
            <Play size={20} />
            {mode === 'upscale' && ` Upscale ${upscaleScale}x`}
            {mode === 'interpolate' && ` Interpolate to ${targetFps}fps`}
            {mode === 'concat' && ` Join ${files.length} Videos`}
          </>
        )}
      </button>
    </div>
  )
}
