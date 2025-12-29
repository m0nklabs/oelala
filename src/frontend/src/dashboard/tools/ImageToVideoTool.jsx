import React, { useMemo, useRef, useState } from 'react'
import { Upload, X, Film, Type, Settings2, Image as ImageIcon, Link, FolderOpen, Sparkles, Info, ChevronDown } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm } from '../../api'
import { sendClientLog } from '../../logging'

const FPS_OPTIONS = [8, 12, 16, 24]

// Model mode options for I2V
const MODEL_MODES = [
  { value: 'light', label: '⚡ Light (Fast)', desc: 'Quick preview, lower quality' },
  { value: 'wan2.2', label: '🎬 Wan2.2 14B (Q5)', desc: 'High quality via ComfyUI' },
]

// Resolution presets with dimensions
const RESOLUTION_PRESETS = {
  '480p': { label: '480p', desc: '480×480', maxPixels: 480 * 480 },
  '720p': { label: '720p', desc: '720×720', maxPixels: 720 * 720 },
  '1080p': { label: '1080p', desc: '1080×1080', maxPixels: 1080 * 1080 },
}

// Aspect ratio options
const ASPECT_RATIOS = ['16:9', '9:16', '1:1', '4:3', '3:4']

export default function ImageToVideoTool({ onOutput, onRefreshHistory }) {
  const fileInputRef = useRef(null)

  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [uploadTab, setUploadTab] = useState('file') // 'file', 'url', 'creations'

  const [prompt, setPrompt] = useState('')
  const [numFrames, setNumFrames] = useState(16) // 5s approx
  const [resolution, setResolution] = useState('480p')
  const [modelMode, setModelMode] = useState('wan2.2')  // default to Wan2.2 for quality
  const [modelVersion, setModelVersion] = useState('v2')
  const [usePose, setUsePose] = useState(false)
  const [aspectRatio, setAspectRatio] = useState('1:1')
  const [fps, setFps] = useState(16)
  const [steps, setSteps] = useState(6)
  const [cfg, setCfg] = useState(5.0)
  const [seed, setSeed] = useState(-1)
  const [showAdvanced, setShowAdvanced] = useState(false)
  
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const canSubmit = useMemo(() => !!file && !busy, [file, busy])

  const onPickFile = (picked) => {
    if (!picked) return
    setFile(picked)
    setError('')

    const url = URL.createObjectURL(picked)
    setPreviewUrl(url)
  }

  const clearFile = () => {
    setFile(null)
    setPreviewUrl('')
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const handleSubmit = async () => {
    if (!file) {
      setError('Image is required')
      return
    }

    setBusy(true)
    setError('')

    const formData = new FormData()
    formData.append('file', file)
    formData.append('num_frames', String(numFrames))
    formData.append('resolution', resolution)
    formData.append('fps', String(fps))
    formData.append('aspect_ratio', aspectRatio)
    if (!usePose) formData.append('prompt', prompt || 'Motion, subject moving naturally')

    // Choose endpoint based on model mode
    let endpoint
    if (usePose) {
      endpoint = `${BACKEND_BASE}/generate-pose`
    } else if (modelMode === 'wan2.2') {
      // Use ComfyUI endpoint for Wan2.2
      endpoint = `${BACKEND_BASE}/generate-wan22-comfyui`
      formData.append('steps', String(steps))
      formData.append('cfg', String(cfg))
      formData.append('seed', String(seed))
    } else {
      // Default endpoint for light/other modes
      endpoint = `${BACKEND_BASE}/generate`
      formData.append('model_version', modelVersion)
    }

    try {
      if (DEBUG) console.debug('🐛 submit image-to-video', { numFrames, usePose, resolution, fps, modelMode })
      const result = await postForm(endpoint, formData)
      if (!result.ok) {
        setError(result.data?.detail || `Generation failed (status ${result.status})`)
        return
      }

      const videoUrl = result.data?.video_url || result.data?.url
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
      setError(message)
      await sendClientLog({
        level: 'error',
        message: 'Image-to-video failed',
        timestamp: new Date().toISOString(),
        meta: { message, modelMode },
      })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="tool-container">
      {/* Mode Selection */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Model Selection</div>
          <button 
            className="btn ghost sm" 
            style={{ fontSize: '0.8rem' }}
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            <Settings2 size={14} style={{ marginRight: '4px' }} /> 
            {showAdvanced ? 'Hide' : 'Show'} Advanced
          </button>
        </div>
        
        <div className="form-group">
          <label className="grok-section-label">Generation Mode</label>
          <div style={{ position: 'relative' }}>
            <select
              value={modelMode}
              onChange={(e) => {
                setModelMode(e.target.value)
                // Adjust defaults for Wan2.2
                if (e.target.value === 'wan2.2') {
                  setResolution('480p')
                  setAspectRatio('1:1')
                  setNumFrames(41)
                }
              }}
              style={{
                width: '100%',
                padding: '12px 40px 12px 16px',
                backgroundColor: 'var(--bg-secondary)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                fontSize: '1rem',
                appearance: 'none',
                cursor: 'pointer',
              }}
            >
              {MODEL_MODES.map((mode) => (
                <option key={mode.value} value={mode.value}>
                  {mode.label}
                </option>
              ))}
            </select>
            <ChevronDown 
              size={20} 
              style={{ 
                position: 'absolute', 
                right: '12px', 
                top: '50%', 
                transform: 'translateY(-50%)', 
                pointerEvents: 'none',
                color: 'var(--text-muted)'
              }} 
            />
          </div>
          <div className="info-badge" style={{ marginTop: '8px' }}>
            {modelMode === 'wan2.2' ? (
              <>
                <span style={{ fontWeight: 600 }}>🎬 Wan2.2 14B Q5</span> • <span style={{ color: '#93c5fd' }}>ComfyUI Backend</span>
                <div style={{ marginTop: '4px', opacity: 0.8 }}>
                  High-quality I2V with DisTorch2 + SageAttention (10GB VRAM)
                </div>
              </>
            ) : (
              <>
                <span style={{ fontWeight: 600 }}>⚡ Light Mode</span> • <span style={{ color: '#93c5fd' }}>Quick Preview</span>
                <div style={{ marginTop: '4px', opacity: 0.8 }}>
                  Fast generation for testing, lower quality output
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Upload Photo */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Upload Photo</div>
        </div>

        <div className="grok-tabs">
          <button 
            className={`grok-tab ${uploadTab === 'file' ? 'active' : ''}`}
            onClick={() => setUploadTab('file')}
          >
            <Upload size={14} /> Upload File
          </button>
          <button 
            className={`grok-tab ${uploadTab === 'url' ? 'active' : ''}`}
            onClick={() => setUploadTab('url')}
          >
            <Link size={14} /> From URL
          </button>
          <button 
            className={`grok-tab ${uploadTab === 'creations' ? 'active' : ''}`}
            onClick={() => setUploadTab('creations')}
          >
            <FolderOpen size={14} /> From My Creations
          </button>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={(e) => onPickFile(e.target.files?.[0])}
          style={{ display: 'none' }}
        />

        {!file ? (
          <div className="upload-box" onClick={() => fileInputRef.current?.click()} style={{ cursor: 'pointer', borderStyle: 'dashed', minHeight: '200px', justifyContent: 'center' }}>
            <Upload size={48} className="text-muted" style={{ opacity: 0.2 }} />
            <div style={{ fontSize: '1rem', fontWeight: 500, color: 'var(--text-secondary)' }}>
              Drag & drop an image here, or click to browse
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              JPEG, PNG, WebP, Max 20MB
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Minimum size: 300x300px
            </div>
          </div>
        ) : (
          <div className="relative" style={{ position: 'relative' }}>
            <img 
              src={previewUrl} 
              alt="Preview" 
              style={{ 
                width: '100%', 
                maxHeight: '400px', 
                objectFit: 'contain', 
                borderRadius: '8px',
                border: '1px solid var(--border-color)'
              }} 
            />
            <button 
              onClick={(e) => { e.stopPropagation(); clearFile(); }}
              style={{
                position: 'absolute',
                top: '12px',
                right: '12px',
                background: 'rgba(0,0,0,0.7)',
                border: 'none',
                color: 'white',
                borderRadius: '50%',
                width: '32px',
                height: '32px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                backdropFilter: 'blur(4px)'
              }}
            >
              <X size={18} />
            </button>
          </div>
        )}
      </div>

      {/* Settings */}
      <div className="grok-card">
        <div className="form-group">
          <label className="grok-section-label">
            Resolution 
            <span className="text-muted" style={{ fontWeight: 400 }}>
              {modelMode === 'wan2.2' ? ' (480p recommended for 16GB VRAM)' : ' (Higher = Better Quality)'}
            </span>
          </label>
          <div className="grok-toggle-group">
            {Object.entries(RESOLUTION_PRESETS).map(([key, preset]) => (
              <button 
                key={key}
                className={`grok-toggle-btn ${resolution === key ? 'active' : ''}`}
                onClick={() => setResolution(key)}
                disabled={modelMode === 'wan2.2' && key === '1080p'}
                style={{
                  opacity: modelMode === 'wan2.2' && key === '1080p' ? 0.5 : 1,
                  cursor: modelMode === 'wan2.2' && key === '1080p' ? 'not-allowed' : 'pointer'
                }}
              >
                {preset.label}
                <span style={{ fontSize: '0.7rem', opacity: 0.7, display: 'block' }}>
                  {preset.desc}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Aspect Ratio */}
        <div className="form-group">
          <label className="grok-section-label">Aspect Ratio</label>
          <div className="grok-toggle-group">
            {ASPECT_RATIOS.map((ar) => (
              <button 
                key={ar}
                className={`grok-toggle-btn ${aspectRatio === ar ? 'active' : ''}`}
                onClick={() => setAspectRatio(ar)}
              >
                {ar}
              </button>
            ))}
          </div>
        </div>

        <div className="form-group">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <label className="grok-section-label">Duration</label>
            <span className="nav-badge" style={{ fontSize: '0.8rem' }}>{(numFrames / fps).toFixed(1)}s ({numFrames}f)</span>
          </div>
          {/* Different ranges for different modes */}
          {modelMode === 'wan2.2' ? (
            <>
              <div style={{ position: 'relative', height: '24px', marginBottom: '8px' }}>
                <input
                  type="range"
                  min="21"
                  max="81"
                  step="4"
                  value={numFrames}
                  onChange={(e) => setNumFrames(parseInt(e.target.value, 10))}
                  style={{ width: '100%', opacity: 0, position: 'absolute', zIndex: 2, cursor: 'pointer' }}
                />
                <div style={{ 
                  position: 'absolute', 
                  top: '10px', 
                  left: 0, 
                  right: 0, 
                  height: '4px', 
                  backgroundColor: '#333', 
                  borderRadius: '2px' 
                }}>
                  <div style={{ 
                    width: `${((numFrames - 21) / (81 - 21)) * 100}%`, 
                    height: '100%', 
                    backgroundColor: 'var(--text-primary)', 
                    borderRadius: '2px' 
                  }} />
                </div>
                <div style={{ 
                  position: 'absolute', 
                  top: '2px', 
                  left: `calc(${((numFrames - 21) / (81 - 21)) * 100}% - 10px)`, 
                  width: '20px', 
                  height: '20px', 
                  backgroundColor: 'white', 
                  borderRadius: '50%', 
                  boxShadow: '0 2px 4px rgba(0,0,0,0.3)' 
                }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                <span>{(21 / fps).toFixed(1)}s</span>
                <span>41f (rec)</span>
                <span>{(81 / fps).toFixed(1)}s</span>
              </div>
            </>
          ) : (
            <>
              <div style={{ position: 'relative', height: '24px', marginBottom: '8px' }}>
                <input
                  type="range"
                  min="12"
                  max="48"
                  step="4"
                  value={numFrames}
                  onChange={(e) => setNumFrames(parseInt(e.target.value, 10))}
                  style={{ width: '100%', opacity: 0, position: 'absolute', zIndex: 2, cursor: 'pointer' }}
                />
                <div style={{ 
                  position: 'absolute', 
                  top: '10px', 
                  left: 0, 
                  right: 0, 
                  height: '4px', 
                  backgroundColor: '#333', 
                  borderRadius: '2px' 
                }}>
                  <div style={{ 
                    width: `${((numFrames - 12) / (48 - 12)) * 100}%`, 
                    height: '100%', 
                    backgroundColor: 'var(--text-primary)', 
                    borderRadius: '2px' 
                  }} />
                </div>
                <div style={{ 
                  position: 'absolute', 
                  top: '2px', 
                  left: `calc(${((numFrames - 12) / (48 - 12)) * 100}% - 10px)`, 
                  width: '20px', 
                  height: '20px', 
                  backgroundColor: 'white', 
                  borderRadius: '50%', 
                  boxShadow: '0 2px 4px rgba(0,0,0,0.3)' 
                }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                <span>{(12 / fps).toFixed(1)}s</span>
                <span>{(48 / fps).toFixed(1)}s</span>
              </div>
            </>
          )}
        </div>

        {/* FPS Control */}
        <div className="form-group">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <label className="grok-section-label">Frame Rate (FPS)</label>
            <span className="nav-badge" style={{ fontSize: '0.8rem' }}>{fps} fps</span>
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
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '8px' }}>
            Higher FPS = smoother motion, more VRAM required
          </div>
        </div>

        {/* Model Version - only for non-Wan2.2 modes */}
        {modelMode !== 'wan2.2' && (
          <div className="form-group">
            <label className="grok-section-label">Model Version</label>
            <div className="grok-toggle-group">
              <button 
                className={`grok-toggle-btn ${modelVersion === 'v1' ? 'active' : ''}`}
                onClick={() => setModelVersion('v1')}
              >
                V1
              </button>
              <button 
                className={`grok-toggle-btn ${modelVersion === 'v2' ? 'active' : ''}`}
                onClick={() => setModelVersion('v2')}
              >
                V2 (Enhanced)
              </button>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '8px' }}>
              V2 features improved video quality, motion, and optional audio generation
            </div>
          </div>
        )}

        {/* Advanced Settings for Wan2.2 */}
        {modelMode === 'wan2.2' && showAdvanced && (
          <div style={{ 
            backgroundColor: 'var(--bg-tertiary)', 
            padding: '16px', 
            borderRadius: '8px',
            marginTop: '8px'
          }}>
            <div style={{ 
              fontSize: '0.9rem', 
              fontWeight: 600, 
              marginBottom: '12px',
              color: 'var(--text-primary)'
            }}>
              ⚙️ Wan2.2 Advanced Settings
            </div>
            
            {/* Steps */}
            <div className="form-group" style={{ marginBottom: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                <label className="grok-section-label">Sampling Steps</label>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{steps}</span>
              </div>
              <input
                type="range"
                min="4"
                max="20"
                step="1"
                value={steps}
                onChange={(e) => setSteps(parseInt(e.target.value, 10))}
                style={{ width: '100%', cursor: 'pointer' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                <span>4 (fast)</span>
                <span>6 (rec)</span>
                <span>20 (quality)</span>
              </div>
            </div>
            
            {/* CFG */}
            <div className="form-group" style={{ marginBottom: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                <label className="grok-section-label">CFG Guidance</label>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{cfg.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="1.0"
                max="10.0"
                step="0.5"
                value={cfg}
                onChange={(e) => setCfg(parseFloat(e.target.value))}
                style={{ width: '100%', cursor: 'pointer' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                <span>1.0</span>
                <span>5.0 (rec)</span>
                <span>10.0</span>
              </div>
            </div>
            
            {/* Seed */}
            <div className="form-group">
              <label className="grok-section-label">Seed</label>
              <div style={{ display: 'flex', gap: '8px' }}>
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(parseInt(e.target.value, 10))}
                  placeholder="-1 for random"
                  style={{
                    flex: 1,
                    padding: '8px 12px',
                    backgroundColor: 'var(--bg-secondary)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '6px',
                    color: 'var(--text-primary)',
                    fontSize: '0.9rem'
                  }}
                />
                <button
                  className="btn ghost sm"
                  onClick={() => setSeed(-1)}
                  style={{ whiteSpace: 'nowrap' }}
                >
                  Random
                </button>
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                -1 = random seed each generation
              </div>
            </div>
          </div>
        )}

        <div className="form-group" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div className="grok-section-label" style={{ marginBottom: '4px' }}>Generate Audio</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Enable audio generation (increases credits)</div>
          </div>
          <label className="grok-switch">
            <input type="checkbox" />
            <span className="grok-slider"></span>
          </label>
        </div>

        <div className="form-group" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div className="grok-section-label" style={{ marginBottom: '4px' }}>Camera Fixed</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Whether to fix the camera position</div>
          </div>
          <label className="grok-switch">
            <input type="checkbox" />
            <span className="grok-slider"></span>
          </label>
        </div>

        <div className="grok-tip-box">
          <div className="grok-tip-header">
            <Info size={14} /> Prompt Tips
          </div>
          <div className="grok-tip-content">
            <ul>
              <li>Structure: [subject + motion] + [scene] + [camera]</li>
              <li>Focus on motion - "walking slowly", "hair blowing in wind"</li>
              <li>Add intensity - "quickly", "gently", "dramatically"</li>
              <li>Camera moves - "slow zoom in", "pan left", "follow shot"</li>
              <li>No negatives - describe what you want, not what to avoid</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Aspect Ratio */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Aspect Ratio</div>
        </div>
        <div className="aspect-grid">
          {[
            { label: 'Auto', icon: <Sparkles size={16} /> },
            { label: '21:9', icon: <div style={{ width: '24px', height: '10px', border: '1px solid currentColor' }} /> },
            { label: '16:9', icon: <div style={{ width: '24px', height: '14px', border: '1px solid currentColor' }} /> },
            { label: '4:3', icon: <div style={{ width: '20px', height: '15px', border: '1px solid currentColor' }} /> },
            { label: '1:1', icon: <div style={{ width: '18px', height: '18px', border: '1px solid currentColor' }} /> },
            { label: '3:4', icon: <div style={{ width: '15px', height: '20px', border: '1px solid currentColor' }} /> },
            { label: '9:16', icon: <div style={{ width: '14px', height: '24px', border: '1px solid currentColor' }} /> },
          ].map((ratio) => (
            <button
              key={ratio.label}
              className={`aspect-btn ${aspectRatio === ratio.label ? 'active' : ''}`}
              onClick={() => setAspectRatio(ratio.label)}
            >
              <div className="aspect-icon" style={{ background: 'transparent', display: 'flex', alignItems: 'center', justifyContent: 'center', border: 'none' }}>
                {ratio.icon}
              </div>
              <span className="aspect-label">{ratio.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Describe Motion */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Describe the Motion</div>
          <div style={{ display: 'flex', gap: '4px' }}>
            <button className="icon-btn" style={{ width: '24px', height: '24px' }}><Type size={12} /></button>
            <button className="icon-btn" style={{ width: '24px', height: '24px' }}><Sparkles size={12} /></button>
          </div>
        </div>
        
        <div style={{ position: 'relative' }}>
          <textarea
            className="form-textarea"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={4}
            placeholder="Describe how you want the image to move or animate... (Optional for image-to-video)"
            style={{ 
              backgroundColor: '#0f0f0f', 
              border: 'none', 
              resize: 'none',
              paddingBottom: '24px'
            }}
          />
          <div style={{ position: 'absolute', bottom: '8px', right: '8px', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            {prompt.length}/2048
          </div>
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
          gap: '8px',
          backgroundColor: '#e5e5e5',
          color: 'black'
        }}
      >
        {busy ? (
          <>Generating...</>
        ) : (
          <>
            <Sparkles size={18} />
            from Image (20)
          </>
        )}
      </button>

      {busy && (
        <div className="progress-container">
          <div className="progress-indeterminate"></div>
        </div>
      )}
    </div>
  )
}
