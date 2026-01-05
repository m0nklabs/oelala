import React, { useState, useCallback } from 'react'
import { Upload, Zap, Loader2, Video, Settings, ChevronDown } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm } from '../../api'

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

export default function FrameInterpolationTool({ onOutput, onJobSubmitted }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [videoInfo, setVideoInfo] = useState(null)

  const [model, setModel] = useState('rife')
  const [mode, setMode] = useState('fps') // 'fps' or 'slowmo'
  const [fpsPreset, setFpsPreset] = useState('30fps → 60fps (2x)')
  const [slowMoPreset, setSlowMoPreset] = useState('2x')
  const [showFlowViz, setShowFlowViz] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)

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
      
      formData.append('show_flow_viz', String(showFlowViz))

      if (DEBUG) console.debug('🔍 Interpolation request:', { model, mode, fpsPreset, slowMoPreset, showFlowViz })

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
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', padding: '20px' }}>
      <div style={{ marginBottom: '8px' }}>
        <h2 style={{ fontSize: '1.3rem', fontWeight: 600, marginBottom: '4px' }}>Frame Interpolation</h2>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          Increase FPS & create smooth slow motion • RIFE/FILM integration
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
        onClick={() => document.getElementById('interpolate-file')?.click()}
      >
        <Upload size={32} style={{ margin: '0 auto 12px', color: 'var(--text-muted)' }} />
        <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>
          {file ? file.name : 'Drop video or click to upload'}
        </p>
        {videoInfo && (
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {videoInfo.width}×{videoInfo.height} • {videoInfo.duration}s • ~{videoInfo.fps}fps
          </p>
        )}
        <input
          id="interpolate-file"
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
            Interpolation Model
          </label>
          <div style={{ display: 'flex', gap: '8px' }}>
            {INTERPOLATION_MODELS.map(m => (
              <button
                key={m.value}
                onClick={() => setModel(m.value)}
                type="button"
                style={{
                  flex: 1,
                  padding: '10px',
                  borderRadius: '6px',
                  border: model === m.value ? '1px solid var(--accent-color)' : '1px solid var(--border-color)',
                  background: model === m.value ? 'rgba(59, 130, 246, 0.2)' : 'var(--bg-secondary)',
                  color: model === m.value ? 'var(--accent-color)' : 'var(--text-secondary)',
                  fontSize: '0.85rem',
                  cursor: 'pointer',
                  transition: 'all 0.15s',
                }}
              >
                <div style={{ fontWeight: 600 }}>
                  {m.label} {m.recommended && '⭐'}
                </div>
                <div style={{ fontSize: '0.7rem', marginTop: '2px', opacity: 0.8 }}>{m.desc}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Mode Selection */}
        <div>
          <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '6px', color: 'var(--text-secondary)' }}>
            Mode
          </label>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={() => setMode('fps')}
              type="button"
              style={{
                flex: 1,
                padding: '10px',
                borderRadius: '6px',
                border: mode === 'fps' ? '1px solid var(--accent-color)' : '1px solid var(--border-color)',
                background: mode === 'fps' ? 'rgba(59, 130, 246, 0.2)' : 'var(--bg-secondary)',
                color: mode === 'fps' ? 'var(--accent-color)' : 'var(--text-secondary)',
                fontSize: '0.85rem',
                cursor: 'pointer',
                transition: 'all 0.15s',
              }}
            >
              <div style={{ fontWeight: 600 }}>FPS Conversion</div>
              <div style={{ fontSize: '0.7rem', marginTop: '2px', opacity: 0.8 }}>Increase frame rate</div>
            </button>
            <button
              onClick={() => setMode('slowmo')}
              type="button"
              style={{
                flex: 1,
                padding: '10px',
                borderRadius: '6px',
                border: mode === 'slowmo' ? '1px solid var(--accent-color)' : '1px solid var(--border-color)',
                background: mode === 'slowmo' ? 'rgba(59, 130, 246, 0.2)' : 'var(--bg-secondary)',
                color: mode === 'slowmo' ? 'var(--accent-color)' : 'var(--text-secondary)',
                fontSize: '0.85rem',
                cursor: 'pointer',
                transition: 'all 0.15s',
              }}
            >
              <div style={{ fontWeight: 600 }}>Slow Motion</div>
              <div style={{ fontSize: '0.7rem', marginTop: '2px', opacity: 0.8 }}>Smooth slow-mo</div>
            </button>
          </div>
        </div>

        {/* FPS Presets (shown when mode === 'fps') */}
        {mode === 'fps' && (
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '6px', color: 'var(--text-secondary)' }}>
              Target FPS
            </label>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {FPS_PRESETS.map(preset => (
                <button
                  key={preset.label}
                  onClick={() => setFpsPreset(preset.label)}
                  type="button"
                  style={{
                    padding: '8px 14px',
                    borderRadius: '6px',
                    border: fpsPreset === preset.label ? '1px solid var(--accent-color)' : '1px solid var(--border-color)',
                    background: fpsPreset === preset.label ? 'rgba(59, 130, 246, 0.2)' : 'var(--bg-secondary)',
                    color: fpsPreset === preset.label ? 'var(--accent-color)' : 'var(--text-secondary)',
                    fontSize: '0.85rem',
                    cursor: 'pointer',
                    transition: 'all 0.15s',
                  }}
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Slow Motion Presets (shown when mode === 'slowmo') */}
        {mode === 'slowmo' && (
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '6px', color: 'var(--text-secondary)' }}>
              Slow Motion Speed
            </label>
            <div style={{ display: 'flex', gap: '8px' }}>
              {SLOW_MOTION_PRESETS.map(preset => (
                <button
                  key={preset.value}
                  onClick={() => setSlowMoPreset(preset.value)}
                  type="button"
                  style={{
                    flex: 1,
                    padding: '10px',
                    borderRadius: '6px',
                    border: slowMoPreset === preset.value ? '1px solid var(--accent-color)' : '1px solid var(--border-color)',
                    background: slowMoPreset === preset.value ? 'rgba(59, 130, 246, 0.2)' : 'var(--bg-secondary)',
                    color: slowMoPreset === preset.value ? 'var(--accent-color)' : 'var(--text-secondary)',
                    fontSize: '0.85rem',
                    cursor: 'pointer',
                    transition: 'all 0.15s',
                  }}
                  title={preset.desc}
                >
                  <div style={{ fontWeight: 600 }}>{preset.label}</div>
                  <div style={{ fontSize: '0.7rem', marginTop: '2px', opacity: 0.8 }}>{preset.desc}</div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Advanced Settings */}
        <div>
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            type="button"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 0',
              background: 'none',
              border: 'none',
              color: 'var(--text-secondary)',
              fontSize: '0.85rem',
              cursor: 'pointer',
            }}
          >
            <Settings size={14} />
            Advanced Settings
            <ChevronDown
              size={14}
              style={{
                transform: showAdvanced ? 'rotate(180deg)' : 'rotate(0deg)',
                transition: 'transform 0.2s',
              }}
            />
          </button>
          {showAdvanced && (
            <div style={{ marginTop: '12px', padding: '12px', background: 'var(--bg-secondary)', borderRadius: '6px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={showFlowViz}
                  onChange={(e) => setShowFlowViz(e.target.checked)}
                  style={{ cursor: 'pointer' }}
                />
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  Show optical flow visualization
                </span>
              </label>
              <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px', marginLeft: '24px' }}>
                Generates a side-by-side view showing motion vectors (for debugging)
              </p>
            </div>
          )}
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
            ✓ Interpolation queued! ({lastQueued.model}, {lastQueued.preset})
          </p>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Job ID: {lastQueued.promptId}
          </p>
        </div>
      )}

      {/* Generate Button */}
      <button
        onClick={handleInterpolate}
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
            Interpolating...
          </>
        ) : (
          <>
            <Zap size={20} />
            Interpolate Frames
          </>
        )}
      </button>

      {/* Result */}
      {result && (
        <div>
          <h3 style={{ fontSize: '1rem', marginBottom: '8px' }}>
            Result ({mode === 'fps' 
              ? FPS_PRESETS.find(p => p.label === fpsPreset)?.label 
              : SLOW_MOTION_PRESETS.find(p => p.value === slowMoPreset)?.label})
          </h3>
          <div style={{ borderRadius: '8px', overflow: 'hidden' }}>
            <video src={result} controls style={{ width: '100%', display: 'block' }} />
          </div>
        </div>
      )}
    </div>
  )
}
