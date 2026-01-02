import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Upload, X, Film, Type, Settings2, Image as ImageIcon, Link, FolderOpen, Sparkles, Info, ChevronDown, Layers } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm } from '../../api'
import { sendClientLog } from '../../logging'

const FPS_OPTIONS = [8, 12, 16, 24]

// Model mode options for I2V (only Q6 DisTorch2 supported)
const MODEL_MODES = [
  { value: 'wan2.2', label: '🎬 Wan2.2 14B Q6 DisTorch2', desc: 'High quality via ComfyUI' },
]

// Resolution presets with dimensions per aspect ratio
const RESOLUTION_PRESETS = {
  '480p': { 
    label: '480p', 
    dimensions: {
      '16:9': '848×480',
      '9:16': '480×848',
      '1:1': '480×480',
      '4:3': '640×480',
      '3:4': '480×640',
    }
  },
  '576p': { 
    label: '576p', 
    dimensions: {
      '16:9': '1024×576',
      '9:16': '576×1024',
      '1:1': '576×576',
      '4:3': '768×576',
      '3:4': '576×768',
    }
  },
  '720p': { 
    label: '720p', 
    dimensions: {
      '16:9': '1280×720',
      '9:16': '720×1280',
      '1:1': '720×720',
      '4:3': '960×720',
      '3:4': '720×960',
    }
  },
}

// Aspect ratio options
const ASPECT_RATIOS = ['16:9', '9:16', '1:1', '4:3', '3:4']

export default function ImageToVideoTool({ onOutput, onRefreshHistory, onCreationsModeChange, onParamsChange }) {
  const fileInputRef = useRef(null)

  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [uploadTab, setUploadTab] = useState('file') // 'file', 'url', 'creations'

  const [prompt, setPrompt] = useState('')
  const [duration, setDuration] = useState(6) // seconds, 3-15 range
  const [resolution, setResolution] = useState('480p')
  const [modelMode, setModelMode] = useState('wan2.2')  // default to Wan2.2 for quality
  const [modelVersion, setModelVersion] = useState('v2')
  const [usePose, setUsePose] = useState(false)
  const [aspectRatio, setAspectRatio] = useState('9:16')
  const [fps, setFps] = useState(16)
  const [steps, setSteps] = useState(6)
  const [cfg, setCfg] = useState(1.0)
  const [seed, setSeed] = useState(-1)
  const [showAdvanced, setShowAdvanced] = useState(false)
  
  // LoRA state
  const [availableLoras, setAvailableLoras] = useState({ high_noise: [], low_noise: [], general: [] })
  const [loraHighNoise, setLoraHighNoise] = useState('')
  const [loraLowNoise, setLoraLowNoise] = useState('')
  const [loraStrength, setLoraStrength] = useState(1.0)
  const [showLoraPanel, setShowLoraPanel] = useState(false)
  
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  // Selected creation from MyMediaTool picker
  const [selectedCreation, setSelectedCreation] = useState(null)

  const canSubmit = useMemo(() => !!file && !busy, [file, busy])

  // Fetch available LoRAs on mount
  useEffect(() => {
    const fetchLoras = async () => {
      try {
        const res = await fetch(`${BACKEND_BASE}/loras`)
        if (res.ok) {
          const data = await res.json()
          setAvailableLoras(data)
          if (DEBUG) console.debug('🐛 loaded LoRAs:', data.count)
        }
      } catch (e) {
        console.error('Failed to fetch LoRAs:', e)
      }
    }
    fetchLoras()
  }, [])

  // Expose current params to parent for JSON download
  useEffect(() => {
    if (onParamsChange) {
      onParamsChange({
        tool: 'ImageToVideo',
        prompt,
        duration,
        resolution,
        modelMode,
        modelVersion,
        aspectRatio,
        fps,
        steps,
        cfg,
        seed,
        usePose,
        loraHighNoise,
        loraLowNoise,
        loraStrength,
        filename: file?.name || null,
      })
    }
  }, [prompt, duration, resolution, modelMode, modelVersion, aspectRatio, fps, steps, cfg, seed, usePose, loraHighNoise, loraLowNoise, loraStrength, file, onParamsChange])

  // Select an image from My Creations (called by MyMediaTool in output panel)
  const selectCreation = useCallback(async (item) => {
    setSelectedCreation(item)
    setError('')
    
    try {
      // Fetch the image and convert to File object
      const imageUrl = `${BACKEND_BASE}${item.url}`
      const response = await fetch(imageUrl)
      const blob = await response.blob()
      const filename = item.filename || item.url.split('/').pop()
      const fileObj = new File([blob], filename, { type: blob.type || 'image/png' })
      
      setFile(fileObj)
      setPreviewUrl(imageUrl)
      setUploadTab('file') // Switch back to file tab to show the selection
      
      // Show in output panel
      onOutput({
        kind: 'image',
        url: imageUrl,
        backendUrl: imageUrl,
        filename: filename,
        meta: { source: 'my-creations', originalItem: item },
      })
      
      if (DEBUG) console.debug('🐛 selected creation:', filename)
    } catch (e) {
      setError('Failed to load selected image')
      console.error('Error selecting creation:', e)
    }
  }, [onOutput])

  // Notify Dashboard when creations tab is active/inactive
  useEffect(() => {
    if (onCreationsModeChange) {
      onCreationsModeChange(uploadTab === 'creations' && !file, selectCreation)
    }
    // Cleanup: disable creations mode when component unmounts
    return () => {
      if (onCreationsModeChange) {
        onCreationsModeChange(false, null)
      }
    }
  }, [uploadTab, file, onCreationsModeChange, selectCreation])

  const onPickFile = (picked) => {
    if (!picked) return
    setFile(picked)
    setError('')
    setSelectedCreation(null) // Clear selection when manually picking

    const url = URL.createObjectURL(picked)
    setPreviewUrl(url)
  }

  const clearFile = () => {
    setFile(null)
    setPreviewUrl('')
    setSelectedCreation(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const handleSubmit = async () => {
    if (!file) {
      setError('Image is required')
      return
    }

    setBusy(true)
    setError('')

    const numFrames = duration * fps
    const formData = new FormData()
    formData.append('file', file)
    formData.append('num_frames', String(numFrames))
    formData.append('resolution', resolution)
    formData.append('fps', String(fps))
    formData.append('aspect_ratio', aspectRatio)
    if (!usePose) formData.append('prompt', prompt || 'Motion, subject moving naturally')

    // Choose endpoint
    let endpoint
    if (usePose) {
      endpoint = `${BACKEND_BASE}/generate-pose`
    } else {
      // Use ComfyUI endpoint for Wan2.2 Q6
      endpoint = `${BACKEND_BASE}/generate-wan22-comfyui`
      formData.append('steps', String(steps))
      formData.append('cfg', String(cfg))
      formData.append('seed', String(seed))
      // LoRA parameters
      if (loraHighNoise) formData.append('lora_high_noise', loraHighNoise)
      if (loraLowNoise) formData.append('lora_low_noise', loraLowNoise)
      formData.append('lora_strength', String(loraStrength))
    }

    try {
      if (DEBUG) console.debug('🐛 submit image-to-video', { duration, numFrames, usePose, resolution, fps, modelMode })
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
      <style>{`
        .animate-spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>

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
                  setResolution('576p')
                  setAspectRatio('9:16')
                  setDuration(6)
                }
              }}
              style={{
                width: '100%',
                padding: '12px 40px 12px 16px',
                backgroundColor: 'var(--bg-secondary, #1a1a1a)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px',
                color: 'var(--text-primary, #fff)',
                fontSize: '1rem',
                appearance: 'none',
                cursor: 'pointer',
              }}
            >
              {MODEL_MODES.map((mode) => (
                <option 
                  key={mode.value} 
                  value={mode.value}
                  style={{ backgroundColor: '#1a1a1a', color: '#fff' }}
                >
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
            <span style={{ fontWeight: 600 }}>🎬 Wan2.2 14B Q6</span> • <span style={{ color: '#93c5fd' }}>ComfyUI Backend</span>
            <div style={{ marginTop: '4px', opacity: 0.8 }}>
              High-quality I2V with DisTorch2 + SageAttention (10GB VRAM)
            </div>
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

        {/* Tab Content: File Upload */}
        {uploadTab === 'file' && !file && (
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
        )}

        {/* Tab Content: URL */}
        {uploadTab === 'url' && !file && (
          <div style={{ padding: '16px 0' }}>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
              Enter image URL:
            </div>
            <input 
              type="url"
              placeholder="https://example.com/image.jpg"
              style={{
                width: '100%',
                padding: '12px',
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                fontSize: '0.9rem'
              }}
              onKeyDown={async (e) => {
                if (e.key === 'Enter' && e.target.value) {
                  try {
                    const response = await fetch(e.target.value)
                    const blob = await response.blob()
                    const filename = e.target.value.split('/').pop() || 'image.jpg'
                    const file = new File([blob], filename, { type: blob.type })
                    onPickFile(file)
                  } catch {
                    setError('Failed to load image from URL')
                  }
                }
              }}
            />
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '8px' }}>
              Press Enter to load
            </div>
          </div>
        )}

        {/* Tab Content: My Creations - show instruction, picker is in output panel */}
        {uploadTab === 'creations' && !file && (
          <div style={{ 
            padding: '24px 16px', 
            textAlign: 'center',
            color: 'var(--text-muted)',
            backgroundColor: 'var(--bg-secondary)',
            borderRadius: '8px',
            border: '1px dashed var(--border-color)'
          }}>
            <ImageIcon size={32} style={{ opacity: 0.5, marginBottom: '12px' }} />
            <div style={{ fontSize: '0.9rem', marginBottom: '8px' }}>
              Select an image from the panel on the right →
            </div>
            <div style={{ fontSize: '0.8rem', opacity: 0.7 }}>
              Browse your generated images
            </div>
          </div>
        )}

        {/* Preview when file is selected (any tab) */}
        {file && (
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
              {modelMode === 'wan2.2' ? ' (576p max for 24GB VRAM)' : ' (Higher = Better Quality)'}
            </span>
          </label>
          <div className="grok-toggle-group">
            {Object.entries(RESOLUTION_PRESETS).map(([key, preset]) => (
              <button 
                key={key}
                className={`grok-toggle-btn ${resolution === key ? 'active' : ''}`}
                onClick={() => setResolution(key)}
                disabled={modelMode === 'wan2.2' && key === '720p'}
                style={{
                  opacity: modelMode === 'wan2.2' && key === '720p' ? 0.5 : 1,
                  cursor: modelMode === 'wan2.2' && key === '720p' ? 'not-allowed' : 'pointer'
                }}
              >
                {preset.label}
                <span style={{ fontSize: '0.7rem', opacity: 0.7, display: 'block' }}>
                  {preset.dimensions[aspectRatio] || preset.dimensions['1:1']}
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
            <span className="nav-badge" style={{ fontSize: '0.8rem' }}>{duration}s ({duration * fps}f)</span>
          </div>
          <div style={{ position: 'relative', height: '24px', marginBottom: '8px' }}>
            <input
              type="range"
              min="3"
              max="15"
              step="1"
              value={duration}
              onChange={(e) => setDuration(parseInt(e.target.value, 10))}
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
                width: `${((duration - 3) / (15 - 3)) * 100}%`, 
                height: '100%', 
                backgroundColor: 'var(--accent-color, #a855f7)', 
                borderRadius: '2px' 
              }} />
            </div>
            <div style={{ 
              position: 'absolute', 
              top: '2px', 
              left: `calc(${((duration - 3) / (15 - 3)) * 100}% - 10px)`, 
              width: '20px', 
              height: '20px', 
              backgroundColor: 'white', 
              borderRadius: '50%', 
              boxShadow: '0 2px 4px rgba(0,0,0,0.3)' 
            }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            <span>3s</span>
            <span>6s (rec)</span>
            <span>15s</span>
          </div>
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
                <span>1.0 (rec)</span>
                <span>5.0</span>
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

            {/* LoRA Settings */}
            <div style={{ 
              marginTop: '16px', 
              paddingTop: '16px', 
              borderTop: '1px solid var(--border-color)' 
            }}>
              <div 
                onClick={() => setShowLoraPanel(!showLoraPanel)}
                style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'center',
                  cursor: 'pointer',
                  marginBottom: showLoraPanel ? '12px' : 0
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Layers size={16} />
                  <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>LoRA Models</span>
                  {(loraHighNoise || loraLowNoise) && (
                    <span style={{ 
                      fontSize: '0.7rem', 
                      backgroundColor: 'var(--accent-color)', 
                      color: 'white',
                      padding: '2px 6px',
                      borderRadius: '4px'
                    }}>
                      Active
                    </span>
                  )}
                </div>
                <span style={{ opacity: 0.5, fontSize: '0.8rem' }}>{showLoraPanel ? '▼' : '▶'}</span>
              </div>

              {showLoraPanel && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {/* High Noise LoRA */}
                  <div>
                    <label style={{ 
                      display: 'block', 
                      fontSize: '0.8rem', 
                      color: 'var(--text-muted)', 
                      marginBottom: '4px' 
                    }}>
                      High Noise LoRA (steps 0-3)
                    </label>
                    <select
                      value={loraHighNoise}
                      onChange={(e) => setLoraHighNoise(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '8px 12px',
                        backgroundColor: 'var(--bg-secondary)',
                        border: '1px solid var(--border-color)',
                        borderRadius: '6px',
                        color: 'var(--text-primary)',
                        fontSize: '0.85rem'
                      }}
                    >
                      <option value="">None</option>
                      {availableLoras.high_noise?.map((lora) => (
                        <option key={lora} value={lora}>{lora.replace('.safetensors', '')}</option>
                      ))}
                      <optgroup label="General LoRAs">
                        {availableLoras.general?.map((lora) => (
                          <option key={lora} value={lora}>{lora.replace('.safetensors', '')}</option>
                        ))}
                      </optgroup>
                    </select>
                  </div>

                  {/* Low Noise LoRA */}
                  <div>
                    <label style={{ 
                      display: 'block', 
                      fontSize: '0.8rem', 
                      color: 'var(--text-muted)', 
                      marginBottom: '4px' 
                    }}>
                      Low Noise LoRA (steps 3+)
                    </label>
                    <select
                      value={loraLowNoise}
                      onChange={(e) => setLoraLowNoise(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '8px 12px',
                        backgroundColor: 'var(--bg-secondary)',
                        border: '1px solid var(--border-color)',
                        borderRadius: '6px',
                        color: 'var(--text-primary)',
                        fontSize: '0.85rem'
                      }}
                    >
                      <option value="">None</option>
                      {availableLoras.low_noise?.map((lora) => (
                        <option key={lora} value={lora}>{lora.replace('.safetensors', '')}</option>
                      ))}
                      <optgroup label="General LoRAs">
                        {availableLoras.general?.map((lora) => (
                          <option key={lora} value={lora}>{lora.replace('.safetensors', '')}</option>
                        ))}
                      </optgroup>
                    </select>
                  </div>

                  {/* LoRA Strength */}
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                        LoRA Strength
                      </label>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                        {loraStrength.toFixed(2)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="2"
                      step="0.05"
                      value={loraStrength}
                      onChange={(e) => setLoraStrength(parseFloat(e.target.value))}
                      style={{ width: '100%', cursor: 'pointer' }}
                    />
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                      <span>0</span>
                      <span>1.0 (default)</span>
                      <span>2.0</span>
                    </div>
                  </div>

                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                    💡 Use high_noise for style, low_noise for details. Same LoRA on both for consistency.
                  </div>
                </div>
              )}
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
            rows={6}
            placeholder="Describe how you want the image to move or animate... (Optional for image-to-video)"
            style={{ 
              backgroundColor: '#0f0f0f', 
              border: '1px solid var(--border-color)',
              borderRadius: '8px',
              resize: 'vertical',
              minHeight: '120px',
              padding: '12px',
              paddingBottom: '28px',
              width: '100%',
              boxSizing: 'border-box'
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
