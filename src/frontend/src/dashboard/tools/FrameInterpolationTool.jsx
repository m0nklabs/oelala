import React, { useState, useCallback, useMemo, useEffect } from 'react'
import { Upload, Zap, Loader2, Settings, ChevronDown } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm } from '../../api'
import { useAuth } from '../../contexts/AuthContext'
import { useToolSettings } from '../../hooks/useToolSettings'
import ResetDefaultsButton from '../../components/ResetDefaultsButton'

// Frame interpolation models
const INTERPOLATION_MODELS = [
  { value: 'rife', label: 'RIFE', desc: 'Fast & high quality', recommended: true },
  { value: 'film', label: 'FILM', desc: 'Google Research model', recommended: false },
]

// FPS conversion presets
const FPS_PRESETS = [
  { from: 15, to: 30, label: '15fps → 30fps (2x)', multiplier: 2 },
  { from: 15, to: 60, label: '15fps → 60fps (4x)', multiplier: 4 },
  { from: 24, to: 30, label: '24fps → 30fps (1.25x)', multiplier: 1.25 },
  { from: 24, to: 60, label: '24fps → 60fps (2.5x)', multiplier: 2.5 },
  { from: 30, to: 60, label: '30fps → 60fps (2x)', multiplier: 2 },
]

// Slow motion presets
const SLOW_MOTION_PRESETS = [
  { value: '2x', label: '2x Slower', multiplier: 2, desc: 'Double frame count' },
  { value: '4x', label: '4x Slower', multiplier: 4, desc: 'Quadruple frame count' },
  { value: '8x', label: '8x Slower', multiplier: 8, desc: 'Epic slow motion' },
]

const INTERP_DEFAULTS = { model: 'rife', mode: 'fps', fpsPreset: '30fps → 60fps (2x)', slowMoPreset: '2x' }

export default function FrameInterpolationTool({ onOutput, onJobSubmitted }) {
  const { user, requestLogin } = useAuth()
  const { initial, save: saveSettings, resetDefaults } = useToolSettings('frame_interpolation', INTERP_DEFAULTS)

  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [videoInfo, setVideoInfo] = useState(null)

  const [model, setModel] = useState(initial.model)
  const [mode, setMode] = useState(initial.mode)
  const [fpsPreset, setFpsPreset] = useState(initial.fpsPreset)
  const [slowMoPreset, setSlowMoPreset] = useState(initial.slowMoPreset)

  // Auto-save settings
  const settingsSnapshot = useMemo(() => ({ model, mode, fpsPreset, slowMoPreset }), [model, mode, fpsPreset, slowMoPreset])
  useEffect(() => { saveSettings(settingsSnapshot) }, [settingsSnapshot, saveSettings])

  const handleResetDefaults = useCallback(() => {
    const d = resetDefaults()
    setModel(d.model); setMode(d.mode); setFpsPreset(d.fpsPreset); setSlowMoPreset(d.slowMoPreset)
  }, [resetDefaults])

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
          fps: 30, // Approximate, actual FPS detection requires server-side
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
          fps: 30,
        })
      }
      video.src = url
    }
  }, [])

  const handleInterpolate = async () => {
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
      formData.append('mode', mode)

      if (mode === 'fps') {
        const preset = FPS_PRESETS.find(p => p.label === fpsPreset)
        formData.append('target_fps', String(preset?.to || 60))
        formData.append('multiplier', String(preset?.multiplier || 2))
      } else {
        const preset = SLOW_MOTION_PRESETS.find(p => p.value === slowMoPreset)
        formData.append('multiplier', String(preset?.multiplier || 2))
      }

      if (DEBUG) console.debug('🔍 Interpolation request:', { model, mode, fpsPreset, slowMoPreset })

      const res = await postForm(`${BACKEND_BASE}/interpolate-video`, formData)

      if (!res.ok) {
        throw new Error(res.data?.detail || 'Frame interpolation failed')
      }

      const promptId = res.data?.prompt_id
      if (!promptId) {
        throw new Error('No prompt_id returned')
      }

      // Show queued confirmation
      setLastQueued({
        promptId,
        model: INTERPOLATION_MODELS.find(m => m.value === model)?.label || model,
        preset: mode === 'fps' ? fpsPreset : `${slowMoPreset} Slow Motion`
      })

      if (DEBUG) console.debug('📋 Interpolation queued:', promptId)

      if (onJobSubmitted) onJobSubmitted(promptId)
    } catch (err) {
      console.error('Interpolation error:', err)
      setError(err.message || 'Failed to interpolate video')
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
            <Zap size={16} />
            Frame Interpolation
          </div>
          <ResetDefaultsButton onReset={handleResetDefaults} />
        </div>

        <div
          className="upload-box"
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => document.getElementById('interpolate-file')?.click()}
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
                  <span>~{videoInfo.fps}fps</span>
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
            id="interpolate-file"
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
        </div>
      </div>

      {/* Model Selection Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Interpolation Model</div>
        </div>
        <div className="grok-toggle-group" style={{ flexDirection: 'column' }}>
          {INTERPOLATION_MODELS.map(m => (
            <button
              key={m.value}
              onClick={() => setModel(m.value)}
              className={`grok-toggle-btn ${model === m.value ? 'active' : ''}`}
              style={{ textAlign: 'left', padding: '10px 12px' }}
            >
              <div style={{ fontWeight: 500 }}>
                {m.label} {m.recommended && '⭐'}
              </div>
              <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>{m.desc}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Mode & Preset Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Mode</div>
        </div>
        <div className="grok-toggle-group" style={{ marginBottom: '16px' }}>
          <button
            onClick={() => setMode('fps')}
            className={`grok-toggle-btn ${mode === 'fps' ? 'active' : ''}`}
          >
            FPS Conversion
          </button>
          <button
            onClick={() => setMode('slowmo')}
            className={`grok-toggle-btn ${mode === 'slowmo' ? 'active' : ''}`}
          >
            Slow Motion
          </button>
        </div>

        {/* FPS Presets */}
        {mode === 'fps' && (
          <div className="form-group">
            <label className="grok-section-label">Target FPS</label>
            <div className="grok-toggle-group" style={{ flexWrap: 'wrap', gap: '4px' }}>
              {FPS_PRESETS.map(preset => (
                <button
                  key={preset.label}
                  onClick={() => setFpsPreset(preset.label)}
                  className={`grok-toggle-btn ${fpsPreset === preset.label ? 'active' : ''}`}
                  style={{ fontSize: '0.8rem', padding: '6px 10px' }}
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Slow Motion Presets */}
        {mode === 'slowmo' && (
          <div className="form-group">
            <label className="grok-section-label">Slow Motion Speed</label>
            <div className="grok-toggle-group" style={{ flexDirection: 'column' }}>
              {SLOW_MOTION_PRESETS.map(preset => (
                <button
                  key={preset.value}
                  onClick={() => setSlowMoPreset(preset.value)}
                  className={`grok-toggle-btn ${slowMoPreset === preset.value ? 'active' : ''}`}
                  style={{ textAlign: 'left', padding: '10px 12px' }}
                >
                  <div style={{ fontWeight: 500 }}>{preset.label}</div>
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>{preset.desc}</div>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {error && <div className="status-banner error">{error}</div>}

      {/* Generate Button */}
      <button
        className="primary-btn"
        onClick={handleInterpolate}
        disabled={!file || submitting}
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', height: '48px', fontSize: '1rem' }}
      >
        {submitting ? (
          <>
            <Loader2 size={18} className="animate-spin" />
            Interpolating...
          </>
        ) : (
          <>
            <Zap size={18} />
            Interpolate Frames
          </>
        )}
      </button>

      {lastQueued && (
        <div className="status-banner success">
          ✅ Interpolation queued! ({lastQueued.model}, {lastQueued.preset}) — Check queue panel for progress
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="grok-card">
          <div className="grok-card-header">
            <div className="grok-card-title">
              Result ({mode === 'fps'
                ? FPS_PRESETS.find(p => p.label === fpsPreset)?.label
                : SLOW_MOTION_PRESETS.find(p => p.value === slowMoPreset)?.label})
            </div>
          </div>
          <div style={{ borderRadius: '8px', overflow: 'hidden' }}>
            <video src={result} controls style={{ width: '100%', display: 'block' }} />
          </div>
        </div>
      )}
    </div>
  )
}
