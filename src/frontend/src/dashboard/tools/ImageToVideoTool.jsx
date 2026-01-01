import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Upload, X, Film, Type, Settings2, Image as ImageIcon, Link, FolderOpen, Sparkles, Info, ChevronDown, Play, RefreshCw, Grid, List } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm } from '../../api'
import { sendClientLog } from '../../logging'

// Spin animation for refresh button
const spinKeyframes = `
@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
`

const FPS_OPTIONS = [8, 12, 16, 24]

// Model mode options for I2V
const MODEL_MODES = [
  { value: 'light', label: '⚡ Light (Fast)', desc: 'Quick preview, lower quality' },
  { value: 'wan2.2', label: '🎬 Wan2.2 14B (Q5)', desc: 'High quality via ComfyUI' },
  { value: 'distorch2', label: '🚀 DisTorch2 Dual-Noise (Q6)', desc: 'Best quality, dual GPU + LoRAs' },
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
  
  // DisTorch2 LoRA settings
  const [loraStrength, setLoraStrength] = useState(1.5)
  const [enableNsfwLora, setEnableNsfwLora] = useState(true)
  const [enableDreamlayLora, setEnableDreamlayLora] = useState(true)
  const [enableLightx2vLora, setEnableLightx2vLora] = useState(true)
  const [enableCumshotLora, setEnableCumshotLora] = useState(true)
  const [negativePrompt, setNegativePrompt] = useState('low quality, blurry, out of focus, unstable camera, artifacts, distortion')
  
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  // My Media state
  const [showMyMedia, setShowMyMedia] = useState(false)
  const [mediaFilter, setMediaFilter] = useState('video')  // 'all', 'video', 'image'
  const [mediaList, setMediaList] = useState([])
  const [mediaLoading, setMediaLoading] = useState(false)
  const [mediaError, setMediaError] = useState('')
  const [mediaStats, setMediaStats] = useState({ videos: 0, images: 0 })
  const [mediaViewMode, setMediaViewMode] = useState('grid')  // 'grid' or 'list'
  const [previewMedia, setPreviewMedia] = useState(null)

  // Fetch ComfyUI media
  const fetchMedia = useCallback(async () => {
    setMediaLoading(true)
    setMediaError('')
    try {
      const res = await fetch(`${BACKEND_BASE}/list-comfyui-media?type=${mediaFilter}`)
      const data = await res.json()
      if (!res.ok) throw new Error(data?.detail || `Failed to load media`)
      setMediaList(data?.media || [])
      setMediaStats({ videos: data?.videos || 0, images: data?.images || 0 })
    } catch (e) {
      setMediaError(e.message)
    } finally {
      setMediaLoading(false)
    }
  }, [mediaFilter])

  // Load media when My Media is opened or filter changes
  useEffect(() => {
    if (showMyMedia) {
      fetchMedia()
    }
  }, [showMyMedia, mediaFilter, fetchMedia])

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
    } else if (modelMode === 'distorch2') {
      // Use DisTorch2 dual-noise endpoint (best quality)
      endpoint = `${BACKEND_BASE}/generate-wan22-distorch2`
      formData.append('steps', String(steps))
      formData.append('cfg', String(cfg))
      formData.append('seed', String(seed))
      formData.append('negative_prompt', negativePrompt)
      formData.append('lora_strength', String(loraStrength))
      formData.append('enable_nsfw_lora', String(enableNsfwLora))
      formData.append('enable_dreamlay_lora', String(enableDreamlayLora))
      formData.append('enable_lightx2v_lora', String(enableLightx2vLora))
      formData.append('enable_cumshot_lora', String(enableCumshotLora))
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
      {/* Inject spin animation */}
      <style>{spinKeyframes}</style>
      
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
                // Adjust defaults for different modes
                if (e.target.value === 'wan2.2') {
                  setResolution('480p')
                  setAspectRatio('1:1')
                  setNumFrames(41)
                  setCfg(5.0)
                } else if (e.target.value === 'distorch2') {
                  setResolution('480p')
                  setAspectRatio('1:1')
                  setNumFrames(41)
                  setCfg(1.0)  // Lightning LoRA works best with cfg=1
                  setSteps(6)
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
            {modelMode === 'distorch2' ? (
              <>
                <span style={{ fontWeight: 600 }}>🚀 DisTorch2 Dual-Noise Q6</span> • <span style={{ color: '#86efac' }}>Best Quality</span>
                <div style={{ marginTop: '4px', opacity: 0.8 }}>
                  Dual GPU (RTX 3060 + 5060 Ti) + Power LoRAs + 2-stage sampling
                </div>
              </>
            ) : modelMode === 'wan2.2' ? (
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
          {(modelMode === 'wan2.2' || modelMode === 'distorch2') ? (
            <>
              <div style={{ position: 'relative', height: '24px', marginBottom: '8px' }}>
                <input
                  type="range"
                  min="21"
                  max="101"
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
                    width: `${((numFrames - 21) / (101 - 21)) * 100}%`, 
                    height: '100%', 
                    backgroundColor: 'var(--text-primary)', 
                    borderRadius: '2px' 
                  }} />
                </div>
                <div style={{ 
                  position: 'absolute', 
                  top: '2px', 
                  left: `calc(${((numFrames - 21) / (101 - 21)) * 100}% - 10px)`, 
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
                <span>97f (6s)</span>
                <span>{(101 / fps).toFixed(1)}s</span>
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
        {modelMode !== 'wan2.2' && modelMode !== 'distorch2' && (
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

        {/* Advanced Settings for DisTorch2 */}
        {modelMode === 'distorch2' && showAdvanced && (
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
              🚀 DisTorch2 Advanced Settings
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
                max="12"
                step="1"
                value={steps}
                onChange={(e) => setSteps(parseInt(e.target.value, 10))}
                style={{ width: '100%', cursor: 'pointer' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                <span>4 (fast)</span>
                <span>6 (rec)</span>
                <span>12</span>
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
                max="6.0"
                step="0.5"
                value={cfg}
                onChange={(e) => setCfg(parseFloat(e.target.value))}
                style={{ width: '100%', cursor: 'pointer' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                <span>1.0 (rec)</span>
                <span>3.0</span>
                <span>6.0</span>
              </div>
            </div>
            
            {/* LoRA Strength */}
            <div className="form-group" style={{ marginBottom: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                <label className="grok-section-label">LoRA Strength</label>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{loraStrength.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="2.5"
                step="0.1"
                value={loraStrength}
                onChange={(e) => setLoraStrength(parseFloat(e.target.value))}
                style={{ width: '100%', cursor: 'pointer' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                <span>0.5</span>
                <span>1.5 (rec)</span>
                <span>2.5</span>
              </div>
            </div>
            
            {/* LoRA Toggles */}
            <div style={{ 
              fontSize: '0.85rem', 
              fontWeight: 500, 
              marginBottom: '8px',
              color: 'var(--text-secondary)'
            }}>
              🎨 LoRA Selection
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '12px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={enableDreamlayLora}
                  onChange={(e) => setEnableDreamlayLora(e.target.checked)}
                  style={{ width: '16px', height: '16px' }}
                />
                <span style={{ fontSize: '0.8rem' }}>DR34ML4Y Style</span>
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={enableNsfwLora}
                  onChange={(e) => setEnableNsfwLora(e.target.checked)}
                  style={{ width: '16px', height: '16px' }}
                />
                <span style={{ fontSize: '0.8rem' }}>NSFW</span>
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={enableLightx2vLora}
                  onChange={(e) => setEnableLightx2vLora(e.target.checked)}
                  style={{ width: '16px', height: '16px' }}
                />
                <span style={{ fontSize: '0.8rem' }}>LightX2V Speed</span>
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={enableCumshotLora}
                  onChange={(e) => setEnableCumshotLora(e.target.checked)}
                  style={{ width: '16px', height: '16px' }}
                />
                <span style={{ fontSize: '0.8rem' }}>Cumshot</span>
              </label>
            </div>
            
            {/* Negative Prompt */}
            <div className="form-group" style={{ marginBottom: '12px' }}>
              <label className="grok-section-label">Negative Prompt</label>
              <textarea
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
                placeholder="What to avoid..."
                rows={2}
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  backgroundColor: 'var(--bg-secondary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  color: 'var(--text-primary)',
                  fontSize: '0.85rem',
                  resize: 'vertical'
                }}
              />
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

        {/* My Media Section - Browse ComfyUI Output */}
        <div style={{ marginTop: '16px' }}>
          <button
            onClick={() => setShowMyMedia(!showMyMedia)}
            style={{
              width: '100%',
              padding: '12px 16px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              backgroundColor: 'var(--bg-tertiary)',
              border: '1px solid var(--border-color)',
              borderRadius: '8px',
              cursor: 'pointer',
              color: 'var(--text-primary)'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span>📁</span>
              <span style={{ fontWeight: 600 }}>My Media</span>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                ({mediaStats.videos} videos, {mediaStats.images} images)
              </span>
            </div>
            <ChevronDown 
              size={16} 
              style={{ 
                transform: showMyMedia ? 'rotate(180deg)' : 'rotate(0deg)',
                transition: 'transform 0.2s'
              }} 
            />
          </button>

          {showMyMedia && (
            <div style={{ 
              marginTop: '8px',
              backgroundColor: 'var(--bg-tertiary)', 
              padding: '16px', 
              borderRadius: '8px',
              border: '1px solid var(--border-color)'
            }}>
              {/* Filter tabs and controls */}
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center',
                marginBottom: '12px',
                flexWrap: 'wrap',
                gap: '8px'
              }}>
                <div style={{ display: 'flex', gap: '4px' }}>
                  {[
                    { key: 'video', label: 'Videos', count: mediaStats.videos },
                    { key: 'image', label: 'Images', count: mediaStats.images },
                    { key: 'all', label: 'All', count: mediaStats.videos + mediaStats.images }
                  ].map(tab => (
                    <button
                      key={tab.key}
                      onClick={() => setMediaFilter(tab.key)}
                      style={{
                        padding: '6px 12px',
                        fontSize: '0.8rem',
                        borderRadius: '6px',
                        border: 'none',
                        cursor: 'pointer',
                        backgroundColor: mediaFilter === tab.key ? 'var(--primary-color)' : 'var(--bg-secondary)',
                        color: mediaFilter === tab.key ? '#000' : 'var(--text-secondary)',
                        fontWeight: mediaFilter === tab.key ? 600 : 400
                      }}
                    >
                      {tab.label} ({tab.count})
                    </button>
                  ))}
                </div>
                <div style={{ display: 'flex', gap: '4px' }}>
                  <button
                    onClick={() => setMediaViewMode(mediaViewMode === 'grid' ? 'list' : 'grid')}
                    style={{
                      padding: '6px 8px',
                      borderRadius: '6px',
                      border: '1px solid var(--border-color)',
                      backgroundColor: 'var(--bg-secondary)',
                      cursor: 'pointer',
                      color: 'var(--text-secondary)'
                    }}
                    title={mediaViewMode === 'grid' ? 'Switch to list' : 'Switch to grid'}
                  >
                    {mediaViewMode === 'grid' ? <List size={14} /> : <Grid size={14} />}
                  </button>
                  <button
                    onClick={fetchMedia}
                    disabled={mediaLoading}
                    style={{
                      padding: '6px 8px',
                      borderRadius: '6px',
                      border: '1px solid var(--border-color)',
                      backgroundColor: 'var(--bg-secondary)',
                      cursor: mediaLoading ? 'wait' : 'pointer',
                      color: 'var(--text-secondary)'
                    }}
                    title="Refresh"
                  >
                    <RefreshCw size={14} style={{ animation: mediaLoading ? 'spin 1s linear infinite' : 'none' }} />
                  </button>
                </div>
              </div>

              {/* Error message */}
              {mediaError && (
                <div style={{ 
                  padding: '8px 12px', 
                  backgroundColor: 'rgba(239, 68, 68, 0.1)', 
                  borderRadius: '6px', 
                  color: '#ef4444',
                  fontSize: '0.8rem',
                  marginBottom: '12px'
                }}>
                  {mediaError}
                </div>
              )}

              {/* Loading state */}
              {mediaLoading && (
                <div style={{ 
                  textAlign: 'center', 
                  padding: '20px', 
                  color: 'var(--text-muted)' 
                }}>
                  Loading media...
                </div>
              )}

              {/* Media Grid/List */}
              {!mediaLoading && mediaList.length === 0 && (
                <div style={{ 
                  textAlign: 'center', 
                  padding: '20px', 
                  color: 'var(--text-muted)',
                  fontSize: '0.9rem'
                }}>
                  No {mediaFilter === 'all' ? 'media' : mediaFilter + 's'} found
                </div>
              )}

              {!mediaLoading && mediaList.length > 0 && (
                <div style={{ 
                  display: mediaViewMode === 'grid' ? 'grid' : 'flex',
                  gridTemplateColumns: mediaViewMode === 'grid' ? 'repeat(auto-fill, minmax(120px, 1fr))' : undefined,
                  flexDirection: mediaViewMode === 'list' ? 'column' : undefined,
                  gap: '8px',
                  maxHeight: '400px',
                  overflowY: 'auto',
                  padding: '4px'
                }}>
                  {mediaList.slice(0, 50).map((item, idx) => (
                    <div
                      key={idx}
                      onClick={() => {
                        if (item.type === 'video') {
                          setPreviewMedia(item)
                        } else {
                          // Use image as source
                          const img = new Image()
                          img.onload = () => {
                            setSourceImage(`${import.meta.env.VITE_API_URL || 'http://192.168.1.2:7998'}${item.url}`)
                          }
                          img.src = `${import.meta.env.VITE_API_URL || 'http://192.168.1.2:7998'}${item.url}`
                        }
                      }}
                      style={{
                        position: 'relative',
                        borderRadius: '8px',
                        overflow: 'hidden',
                        cursor: 'pointer',
                        border: '1px solid var(--border-color)',
                        backgroundColor: 'var(--bg-secondary)',
                        aspectRatio: mediaViewMode === 'grid' ? '1' : undefined,
                        display: mediaViewMode === 'list' ? 'flex' : 'block',
                        alignItems: mediaViewMode === 'list' ? 'center' : undefined,
                        gap: mediaViewMode === 'list' ? '12px' : undefined,
                        padding: mediaViewMode === 'list' ? '8px' : undefined
                      }}
                    >
                      {item.type === 'video' ? (
                        <>
                          <div style={{
                            width: mediaViewMode === 'grid' ? '100%' : '80px',
                            height: mediaViewMode === 'grid' ? '100%' : '60px',
                            backgroundColor: '#1a1a1a',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            flexShrink: 0,
                            borderRadius: mediaViewMode === 'list' ? '4px' : undefined
                          }}>
                            <Play size={24} style={{ color: 'var(--text-muted)' }} />
                          </div>
                          {mediaViewMode === 'grid' && (
                            <div style={{
                              position: 'absolute',
                              bottom: 0,
                              left: 0,
                              right: 0,
                              padding: '4px 6px',
                              backgroundColor: 'rgba(0,0,0,0.8)',
                              fontSize: '0.65rem',
                              color: 'var(--text-muted)',
                              whiteSpace: 'nowrap',
                              overflow: 'hidden',
                              textOverflow: 'ellipsis'
                            }}>
                              {item.filename}
                            </div>
                          )}
                        </>
                      ) : (
                        <img
                          src={`${import.meta.env.VITE_API_URL || 'http://192.168.1.2:7998'}${item.url}`}
                          alt={item.filename}
                          style={{
                            width: mediaViewMode === 'grid' ? '100%' : '80px',
                            height: mediaViewMode === 'grid' ? '100%' : '60px',
                            objectFit: 'cover',
                            flexShrink: 0,
                            borderRadius: mediaViewMode === 'list' ? '4px' : undefined
                          }}
                        />
                      )}
                      {mediaViewMode === 'list' && (
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ 
                            fontSize: '0.85rem', 
                            color: 'var(--text-primary)',
                            whiteSpace: 'nowrap',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis'
                          }}>
                            {item.filename}
                          </div>
                          <div style={{ 
                            fontSize: '0.75rem', 
                            color: 'var(--text-muted)' 
                          }}>
                            {item.type === 'video' ? '🎬' : '🖼️'} • {(item.size / 1024 / 1024).toFixed(1)} MB
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Show more indicator */}
              {mediaList.length > 50 && (
                <div style={{ 
                  textAlign: 'center', 
                  marginTop: '8px', 
                  fontSize: '0.8rem', 
                  color: 'var(--text-muted)' 
                }}>
                  Showing 50 of {mediaList.length} items
                </div>
              )}
            </div>
          )}
        </div>

        {/* Video Preview Modal */}
        {previewMedia && (
          <div
            onClick={() => setPreviewMedia(null)}
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: 'rgba(0,0,0,0.9)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 1000,
              cursor: 'pointer'
            }}
          >
            <div onClick={e => e.stopPropagation()} style={{ maxWidth: '90%', maxHeight: '90%' }}>
              <video
                src={`${import.meta.env.VITE_API_URL || 'http://192.168.1.2:7998'}${previewMedia.url}`}
                controls
                autoPlay
                style={{ maxWidth: '100%', maxHeight: '80vh', borderRadius: '8px' }}
              />
              <div style={{ 
                textAlign: 'center', 
                marginTop: '12px', 
                color: 'var(--text-muted)',
                fontSize: '0.9rem'
              }}>
                {previewMedia.filename}
                <button
                  onClick={() => setPreviewMedia(null)}
                  style={{
                    marginLeft: '16px',
                    padding: '6px 16px',
                    borderRadius: '6px',
                    border: 'none',
                    backgroundColor: 'var(--bg-secondary)',
                    color: 'var(--text-primary)',
                    cursor: 'pointer'
                  }}
                >
                  Close
                </button>
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
