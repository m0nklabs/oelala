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
    <div className="post-processing-tool" style={{ padding: '20px', maxWidth: '900px', margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h2 style={{ marginBottom: '8px' }}>⚙️ Post-Processing</h2>
        <ResetDefaultsButton onReset={handleResetDefaults} />
      </div>
      <p style={{ color: 'var(--text-muted)', marginBottom: '24px' }}>
        Process existing videos: upscale resolution, smooth motion, or join multiple clips
      </p>

      {/* Mode Selection */}
      <div style={{ display: 'flex', gap: '12px', marginBottom: '24px' }}>
        {Object.values(PROCESSING_MODES).map(m => {
          const Icon = m.icon
          return (
            <button
              key={m.id}
              onClick={() => setMode(m.id)}
              style={{
                flex: 1,
                padding: '16px',
                backgroundColor: mode === m.id ? 'var(--accent-color)' : 'var(--bg-secondary)',
                border: mode === m.id ? '2px solid var(--accent-color)' : '1px solid var(--border-color)',
                borderRadius: '12px',
                cursor: 'pointer',
                textAlign: 'center',
                transition: 'all 0.2s',
              }}
            >
              <Icon size={24} style={{ marginBottom: '8px', color: mode === m.id ? '#fff' : 'var(--text-muted)' }} />
              <div style={{ fontWeight: 600, color: mode === m.id ? '#fff' : 'var(--text-primary)' }}>
                {m.label.replace(/^. /, '')}
              </div>
              <div style={{ fontSize: '0.75rem', color: mode === m.id ? 'rgba(255,255,255,0.7)' : 'var(--text-muted)', marginTop: '4px' }}>
                {m.desc}
              </div>
            </button>
          )
        })}
      </div>

      {/* File Drop Zone */}
      <div
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        style={{
          border: '2px dashed var(--border-color)',
          borderRadius: '12px',
          padding: '24px',
          textAlign: 'center',
          backgroundColor: 'var(--bg-secondary)',
          marginBottom: '16px',
          cursor: 'pointer',
        }}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          multiple={mode === 'concat'}
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        <Upload size={32} style={{ color: 'var(--text-muted)', marginBottom: '8px' }} />
        <div style={{ color: 'var(--text-primary)', fontWeight: 500 }}>
          {mode === 'concat' ? 'Drop videos here or click to upload' : 'Drop a video here or click to upload'}
        </div>
        <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '4px' }}>
          Or select from your media library
        </div>
      </div>

      {/* Select from Media Library */}
      <button
        onClick={() => setShowMediaPicker(true)}
        style={{
          width: '100%',
          padding: '12px',
          backgroundColor: 'var(--bg-tertiary)',
          border: '1px solid var(--border-color)',
          borderRadius: '8px',
          cursor: 'pointer',
          color: 'var(--text-primary)',
          marginBottom: '24px',
        }}
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
        <div style={{ marginBottom: '24px' }}>
          <h4 style={{ marginBottom: '12px' }}>
            {mode === 'concat' ? `Videos to Join (${files.length})` : 'Selected Video'}
            {mode === 'concat' && totalDuration > 0 && (
              <span style={{ fontWeight: 'normal', color: 'var(--text-muted)', marginLeft: '8px' }}>
                Total: {totalDuration.toFixed(1)}s
              </span>
            )}
          </h4>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {files.map((entry, idx) => (
              <div
                key={entry.id}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  padding: '12px',
                  backgroundColor: 'var(--bg-secondary)',
                  borderRadius: '8px',
                  border: '1px solid var(--border-color)',
                }}
              >
                {mode === 'concat' && (
                  <GripVertical size={16} style={{ color: 'var(--text-muted)', cursor: 'grab' }} />
                )}

                <div style={{
                  width: '80px',
                  height: '45px',
                  borderRadius: '4px',
                  overflow: 'hidden',
                  flexShrink: 0,
                }}>
                  <video
                    src={entry.preview}
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                    muted
                  />
                </div>

                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {entry.file?.name || entry.filename || 'Video'}
                  </div>
                  {entry.videoInfo && (
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      {entry.videoInfo.width}×{entry.videoInfo.height} • {entry.videoInfo.duration}s
                    </div>
                  )}
                </div>

                {mode === 'concat' && (
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                    #{idx + 1}
                  </div>
                )}

                <button
                  onClick={() => removeFile(entry.id)}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: 'var(--error-color)',
                    cursor: 'pointer',
                    padding: '8px',
                  }}
                >
                  <Trash2 size={16} />
                </button>
              </div>
            ))}
          </div>

          {mode === 'concat' && (
            <button
              onClick={() => fileInputRef.current?.click()}
              style={{
                marginTop: '8px',
                padding: '8px 16px',
                backgroundColor: 'var(--bg-tertiary)',
                border: '1px dashed var(--border-color)',
                borderRadius: '8px',
                cursor: 'pointer',
                color: 'var(--text-muted)',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}
            >
              <Plus size={16} /> Add more videos
            </button>
          )}
        </div>
      )}

      {/* Mode-specific Settings */}
      <div style={{
        backgroundColor: 'var(--bg-secondary)',
        padding: '16px',
        borderRadius: '12px',
        marginBottom: '24px',
      }}>
        <h4 style={{ marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Settings size={16} /> Settings
        </h4>

        {mode === 'upscale' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>
                Upscale Model
              </label>
              <div style={{ display: 'flex', gap: '8px' }}>
                {UPSCALE_MODELS.map(m => (
                  <button
                    key={m.value}
                    onClick={() => setUpscaleModel(m.value)}
                    style={{
                      flex: 1,
                      padding: '10px',
                      backgroundColor: upscaleModel === m.value ? 'var(--accent-color)' : 'var(--bg-tertiary)',
                      border: upscaleModel === m.value ? '2px solid var(--accent-color)' : '1px solid var(--border-color)',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      color: upscaleModel === m.value ? '#fff' : 'var(--text-primary)',
                    }}
                  >
                    <div style={{ fontWeight: 500 }}>{m.label}</div>
                    <div style={{ fontSize: '0.75rem', opacity: 0.7, marginTop: '4px' }}>{m.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>
                Scale Factor
              </label>
              <div style={{ display: 'flex', gap: '8px' }}>
                {[2, 4].map(s => (
                  <button
                    key={s}
                    onClick={() => setUpscaleScale(s)}
                    style={{
                      padding: '10px 24px',
                      backgroundColor: upscaleScale === s ? 'var(--accent-color)' : 'var(--bg-tertiary)',
                      border: upscaleScale === s ? '2px solid var(--accent-color)' : '1px solid var(--border-color)',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      color: upscaleScale === s ? '#fff' : 'var(--text-primary)',
                      fontWeight: 600,
                    }}
                  >
                    {s}x
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {mode === 'interpolate' && (
          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>
              Target Frame Rate
            </label>
            <div style={{ display: 'flex', gap: '8px' }}>
              {FPS_PRESETS.map(p => (
                <button
                  key={p.value}
                  onClick={() => setTargetFps(p.value)}
                  style={{
                    flex: 1,
                    padding: '12px',
                    backgroundColor: targetFps === p.value ? 'var(--accent-color)' : 'var(--bg-tertiary)',
                    border: targetFps === p.value ? '2px solid var(--accent-color)' : '1px solid var(--border-color)',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    color: targetFps === p.value ? '#fff' : 'var(--text-primary)',
                  }}
                >
                  <div style={{ fontWeight: 600 }}>{p.label}</div>
                  <div style={{ fontSize: '0.75rem', opacity: 0.7, marginTop: '4px' }}>
                    {p.multiplier}x frames
                  </div>
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

      {/* Error display */}
      {error && (
        <div style={{
          padding: '12px 16px',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid var(--error-color)',
          borderRadius: '8px',
          color: 'var(--error-color)',
          marginBottom: '16px',
        }}>
          {error}
        </div>
      )}

      {/* Success message */}
      {lastQueued && (
        <div style={{
          padding: '12px 16px',
          backgroundColor: 'rgba(34, 197, 94, 0.1)',
          border: '1px solid var(--success-color)',
          borderRadius: '8px',
          color: 'var(--success-color)',
          marginBottom: '16px',
        }}>
          ✅ Job queued! Check the Jobs panel for progress.
        </div>
      )}

      {/* Submit Button */}
      <button
        onClick={handleSubmit}
        disabled={submitting || files.length === 0}
        style={{
          width: '100%',
          padding: '16px',
          backgroundColor: submitting || files.length === 0 ? 'var(--bg-tertiary)' : 'var(--accent-color)',
          border: 'none',
          borderRadius: '12px',
          color: submitting || files.length === 0 ? 'var(--text-muted)' : '#fff',
          fontWeight: 600,
          fontSize: '1rem',
          cursor: submitting || files.length === 0 ? 'not-allowed' : 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '8px',
        }}
      >
        {submitting ? (
          <>
            <Loader2 size={20} className="animate-spin" /> Processing...
          </>
        ) : (
          <>
            <Play size={20} />
            {mode === 'upscale' && `Upscale ${upscaleScale}x`}
            {mode === 'interpolate' && `Interpolate to ${targetFps}fps`}
            {mode === 'concat' && `Join ${files.length} Videos`}
          </>
        )}
      </button>
    </div>
  )
}
