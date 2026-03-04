import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Upload, ZoomIn, X, Loader2, Image as ImageIcon, Video, Sparkles } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm } from '../../api'
import { useAuth } from '../../contexts/AuthContext'
import { useToolSettings } from '../../hooks/useToolSettings'
import ResetDefaultsButton from '../../components/ResetDefaultsButton'

// ─── Image upscale models ───────────────────────────────────────────────────
const IMAGE_MODELS = [
  { value: 'RealESRGAN_x4plus.pth', label: 'Real-ESRGAN x4+', desc: 'Best for photos' },
  { value: 'RealESRGAN_x4plus_anime_6B.pth', label: 'Real-ESRGAN Anime', desc: 'Optimized for anime/illustration' },
  { value: '4x-UltraSharp.pth', label: '4x UltraSharp', desc: 'Sharp details, textures' },
]

// ─── Video upscale presets ──────────────────────────────────────────────────
const VIDEO_PRESETS = [
  { value: 'fast', label: 'Fast', desc: 'Lanczos — instant, no GPU', credits: 5 },
  { value: 'balanced', label: 'Balanced', desc: 'RealESRGAN AI per-frame', credits: 15 },
  { value: 'quality', label: 'Quality', desc: 'SeedVR2 3B — best quality', credits: 30 },
]

// ─── Scale options ──────────────────────────────────────────────────────────
const SCALE_OPTIONS = [2, 4]

/**
 * Unified Upscale Tool — works for both images and videos.
 * Auto-detects media type from uploaded file.
 */
const UPSCALE_DEFAULTS = { imageModel: IMAGE_MODELS[0].value, videoPreset: 'balanced', scale: 2, faceEnhance: false }

export default function UpscaleTool({ onOutput, onJobSubmitted }) {
  const { user, requestLogin } = useAuth()
  const { initial, save: saveSettings, resetDefaults } = useToolSettings('upscale', UPSCALE_DEFAULTS)
  const fileInputRef = useRef(null)

  // File state
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [mediaType, setMediaType] = useState(null) // 'image' | 'video'
  const [mediaInfo, setMediaInfo] = useState(null) // { width, height, duration? }

  // Settings
  const [imageModel, setImageModel] = useState(initial.imageModel)
  const [videoPreset, setVideoPreset] = useState(initial.videoPreset)
  const [scale, setScale] = useState(initial.scale)
  const [faceEnhance, setFaceEnhance] = useState(initial.faceEnhance)

  // Auto-save settings
  const settingsSnapshot = useMemo(() => ({ imageModel, videoPreset, scale, faceEnhance }), [imageModel, videoPreset, scale, faceEnhance])
  useEffect(() => { saveSettings(settingsSnapshot) }, [settingsSnapshot, saveSettings])

  const handleResetDefaults = useCallback(() => {
    const d = resetDefaults()
    setImageModel(d.imageModel); setVideoPreset(d.videoPreset); setScale(d.scale); setFaceEnhance(d.faceEnhance)
  }, [resetDefaults])

  // Processing
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)
  const [lastQueued, setLastQueued] = useState(null)

  // Detect media type from file
  const processFile = useCallback((f) => {
    if (!f) return

    const isVideo = f.type.startsWith('video/')
    const isImage = f.type.startsWith('image/')

    if (!isVideo && !isImage) {
      setError('Unsupported file type. Use images (PNG, JPG) or videos (MP4, WebM).')
      return
    }

    setFile(f)
    setMediaType(isVideo ? 'video' : 'image')
    setError(null)
    setLastQueued(null)

    const url = URL.createObjectURL(f)
    setPreview(url)

    // Get resolution info
    if (isVideo) {
      const vid = document.createElement('video')
      vid.onloadedmetadata = () => {
        setMediaInfo({
          width: vid.videoWidth,
          height: vid.videoHeight,
          duration: vid.duration,
        })
      }
      vid.src = url
    } else {
      const img = new window.Image()
      img.onload = () => {
        setMediaInfo({ width: img.naturalWidth, height: img.naturalHeight })
      }
      img.src = url
    }
  }, [])

  const handleFileChange = useCallback((e) => {
    const f = e.target.files?.[0]
    if (f) processFile(f)
  }, [processFile])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    const f = e.dataTransfer?.files?.[0]
    if (f) processFile(f)
  }, [processFile])

  const clearFile = useCallback(() => {
    if (preview) URL.revokeObjectURL(preview)
    setFile(null)
    setPreview(null)
    setMediaType(null)
    setMediaInfo(null)
    setError(null)
    setLastQueued(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }, [preview])

  // Compute output resolution
  const outputRes = useMemo(() => {
    if (!mediaInfo) return null
    return { width: mediaInfo.width * scale, height: mediaInfo.height * scale }
  }, [mediaInfo, scale])

  // Credits estimate
  const creditsNeeded = useMemo(() => {
    if (mediaType === 'image') return 5
    const preset = VIDEO_PRESETS.find(p => p.value === videoPreset)
    return preset?.credits || 15
  }, [mediaType, videoPreset])

  // Submit upscale
  const handleSubmit = async () => {
    if (!user) {
      requestLogin('Log in to upscale')
      return
    }
    if (!file) return

    setSubmitting(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      let endpoint
      if (mediaType === 'image') {
        endpoint = '/upscale'
        formData.append('model', imageModel)
        formData.append('scale', String(scale))
        formData.append('face_enhance', String(faceEnhance))
      } else {
        endpoint = '/upscale-video'
        formData.append('preset', videoPreset)
        formData.append('scale', String(scale))
      }

      if (DEBUG) console.debug('🔍 Upscale submit:', { endpoint, mediaType, scale })

      const result = await postForm(`${BACKEND_BASE}${endpoint}`, formData)

      if (!result.ok) {
        setError(result.data?.detail || `Upscale failed (${result.status})`)
        return
      }

      setLastQueued({
        promptId: result.data.prompt_id,
        credits: result.data.credits_used,
        type: mediaType,
      })

      if (onJobSubmitted) onJobSubmitted()
      if (onOutput) onOutput({ kind: 'upscale', ...result.data })
    } catch (e) {
      setError(e.message || 'Failed to start upscale')
    } finally {
      setSubmitting(false)
    }
  }

  const canSubmit = file && !submitting

  return (
    <div className="tool-container">
      {/* Upload Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <ZoomIn size={16} />
            Upscale
          </div>
          <ResetDefaultsButton onReset={handleResetDefaults} />
          {mediaType && (
            <span className="info-badge" style={{ fontSize: '0.75rem' }}>
              {mediaType === 'image' ? '🖼️ Image' : '🎬 Video'}
            </span>
          )}
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*,video/*"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />

        <div
          className="upload-box"
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          style={{ cursor: 'pointer' }}
        >
          {preview ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px', width: '100%' }}>
              {mediaType === 'video' ? (
                <video
                  src={preview}
                  controls
                  muted
                  style={{ maxHeight: '180px', borderRadius: '8px', maxWidth: '100%' }}
                />
              ) : (
                <img
                  src={preview}
                  alt="Preview"
                  style={{ maxHeight: '180px', borderRadius: '8px', maxWidth: '100%', objectFit: 'contain' }}
                />
              )}
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{file?.name}</span>
                <button
                  onClick={(e) => { e.stopPropagation(); clearFile() }}
                  className="icon-btn"
                  style={{ width: '24px', height: '24px', padding: '4px', color: '#ef4444' }}
                >
                  <X size={14} />
                </button>
              </div>
              {mediaInfo && (
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', gap: '12px' }}>
                  <span>📐 {mediaInfo.width}×{mediaInfo.height}</span>
                  {mediaInfo.duration != null && (
                    <span>⏱️ {mediaInfo.duration.toFixed(1)}s</span>
                  )}
                </div>
              )}
            </div>
          ) : (
            <>
              <Upload size={32} className="text-muted" />
              <div className="text-muted">Drop image or video here, or click to upload</div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>PNG, JPG, MP4, WebM</span>
            </>
          )}
        </div>
      </div>

      {/* Settings — depends on media type */}
      {file && (
        <div className="grok-card">
          <div className="grok-card-header">
            <div className="grok-card-title">Settings</div>
          </div>

          {/* Scale */}
          <div className="form-group">
            <label className="grok-section-label">Scale Factor</label>
            <div className="grok-toggle-group">
              {SCALE_OPTIONS.map(s => (
                <button
                  key={s}
                  onClick={() => setScale(s)}
                  className={`grok-toggle-btn ${scale === s ? 'active' : ''}`}
                >
                  {s}x
                </button>
              ))}
            </div>
            {outputRes && (
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '8px' }}>
                Output: {outputRes.width}×{outputRes.height}
              </div>
            )}
          </div>

          {/* Image-specific: model + face enhance */}
          {mediaType === 'image' && (
            <>
              <div className="form-group">
                <label className="grok-section-label">Model</label>
                <div className="grok-toggle-group" style={{ flexDirection: 'column' }}>
                  {IMAGE_MODELS.map(m => (
                    <button
                      key={m.value}
                      onClick={() => setImageModel(m.value)}
                      className={`grok-toggle-btn ${imageModel === m.value ? 'active' : ''}`}
                      style={{ textAlign: 'left', padding: '10px 12px' }}
                    >
                      <div style={{ fontWeight: 500 }}>{m.label}</div>
                      <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>{m.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Face enhance toggle */}
              <div className="form-group" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label className="grok-section-label" style={{ marginBottom: 0 }}>Face Enhancement (GFPGAN)</label>
                <label className="grok-switch">
                  <input
                    type="checkbox"
                    checked={faceEnhance}
                    onChange={(e) => setFaceEnhance(e.target.checked)}
                  />
                  <span className="grok-slider"></span>
                </label>
              </div>
            </>
          )}

          {/* Video-specific: preset */}
          {mediaType === 'video' && (
            <div className="form-group">
              <label className="grok-section-label">Quality Preset</label>
              <div className="grok-toggle-group" style={{ flexDirection: 'column' }}>
                {VIDEO_PRESETS.map(p => (
                  <button
                    key={p.value}
                    onClick={() => setVideoPreset(p.value)}
                    className={`grok-toggle-btn ${videoPreset === p.value ? 'active' : ''}`}
                    style={{ textAlign: 'left', padding: '10px 12px' }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontWeight: 500 }}>{p.label}</span>
                      <span className="nav-badge" style={{ fontSize: '0.7rem' }}>{p.credits} credits</span>
                    </div>
                    <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>{p.desc}</div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {error && <div className="status-banner error">{error}</div>}

      {/* Submit */}
      <button
        className="primary-btn"
        onClick={handleSubmit}
        disabled={!canSubmit}
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
      >
        {submitting ? (
          <>
            <Loader2 size={18} className="animate-spin" />
            Upscaling...
          </>
        ) : (
          <>
            <Sparkles size={18} />
            Upscale {mediaType === 'video' ? 'Video' : mediaType === 'image' ? 'Image' : 'Media'}
            {file && <span style={{ opacity: 0.7, fontSize: '0.8rem' }}>({creditsNeeded} credits)</span>}
          </>
        )}
      </button>

      {lastQueued && (
        <div className="status-banner success">
          ✅ Upscale queued! ({lastQueued.credits} credits) — Check queue panel for progress
        </div>
      )}

      {/* Info */}
      <div className="info-badge" style={{ marginTop: '12px', textAlign: 'center' }}>
        Upload any image or video to enhance its resolution using AI upscaling models.
      </div>
    </div>
  )
}
