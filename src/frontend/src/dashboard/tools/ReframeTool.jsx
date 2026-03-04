import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import { Frame, Upload, Loader2, Download, Copy, Move, ChevronDown } from 'lucide-react'
import { BACKEND_BASE, DEBUG, getMediaUrl } from '../../config'
import { postForm, getJson, apiFetch } from '../../api'
import { useAuth } from '../../contexts/AuthContext'
import MediaImportModal from '../../components/MediaImportModal'
import CreationsPickerModal from '../../components/CreationsPickerModal'
import { useToolSettings } from '../../hooks/useToolSettings'
import ResetDefaultsButton from '../../components/ResetDefaultsButton'

const ASPECT_RATIOS = [
  { id: '1:1', label: '1:1 (Square)', width: 1024, height: 1024 },
  { id: '16:9', label: '16:9 (Widescreen)', width: 1280, height: 720 },
  { id: '9:16', label: '9:16 (Portrait)', width: 720, height: 1280 },
  { id: '4:3', label: '4:3 (Standard)', width: 1024, height: 768 },
  { id: '3:4', label: '3:4 (Portrait)', width: 768, height: 1024 },
  { id: '21:9', label: '21:9 (Ultrawide)', width: 1344, height: 576 },
  { id: '3:2', label: '3:2 (Photo)', width: 1152, height: 768 },
  { id: '2:3', label: '2:3 (Photo Portrait)', width: 768, height: 1152 },
]

const POSITIONS = [
  { id: 'center', label: 'Center', icon: '⊕' },
  { id: 'top', label: 'Top', icon: '⬆️' },
  { id: 'bottom', label: 'Bottom', icon: '⬇️' },
  { id: 'left', label: 'Left', icon: '⬅️' },
  { id: 'right', label: 'Right', icon: '➡️' },
  { id: 'top-left', label: 'Top Left', icon: '↖️' },
  { id: 'top-right', label: 'Top Right', icon: '↗️' },
  { id: 'bottom-left', label: 'Bottom Left', icon: '↙️' },
  { id: 'bottom-right', label: 'Bottom Right', icon: '↘️' },
]

const MODELS = [
  { id: 'sdxl', label: 'SDXL (Quality)', file: 'CyberRealisticPony_v8.safetensors' },
  { id: 'flux', label: 'Flux (Fast)', file: 'flux1-dev-bnb-nf4.safetensors' },
]

const REFRAME_DEFAULTS = {
  aspectRatioId: ASPECT_RATIOS[0].id, position: 'center', modelId: MODELS[0].id,
  prompt: '', steps: 25, cfg: 7, denoise: 0.85, feathering: 32, showAdvanced: false,
}

export default function ReframeTool({ onJobSubmitted, pendingImport, onImportConsumed }) {
  const { user, requestLogin } = useAuth()
  const { initial, save: saveSettings, resetDefaults } = useToolSettings('reframe', REFRAME_DEFAULTS)

  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [importModal, setImportModal] = useState(null)
  const [showCreationsPicker, setShowCreationsPicker] = useState(false)
  const [originalSize, setOriginalSize] = useState({ width: 0, height: 0 })
  const [aspectRatio, setAspectRatio] = useState(ASPECT_RATIOS.find(a => a.id === initial.aspectRatioId) || ASPECT_RATIOS[0])
  const [position, setPosition] = useState(initial.position)
  const [model, setModel] = useState(MODELS.find(m => m.id === initial.modelId) || MODELS[0])
  const [prompt, setPrompt] = useState(initial.prompt)
  const [steps, setSteps] = useState(initial.steps)
  const [cfg, setCfg] = useState(initial.cfg)
  const [denoise, setDenoise] = useState(initial.denoise)
  const [feathering, setFeathering] = useState(initial.feathering)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [showAdvanced, setShowAdvanced] = useState(initial.showAdvanced)
  const [lastQueued, setLastQueued] = useState(null)
  const fileInputRef = useRef(null)

  // Auto-save settings (store IDs for object states)
  const settingsSnapshot = useMemo(() => ({
    aspectRatioId: aspectRatio.id, position, modelId: model.id,
    prompt, steps, cfg, denoise, feathering, showAdvanced,
  }), [aspectRatio, position, model, prompt, steps, cfg, denoise, feathering, showAdvanced])
  useEffect(() => { saveSettings(settingsSnapshot) }, [settingsSnapshot, saveSettings])

  const handleResetDefaults = useCallback(() => {
    const d = resetDefaults()
    setAspectRatio(ASPECT_RATIOS.find(a => a.id === d.aspectRatioId) || ASPECT_RATIOS[0])
    setPosition(d.position)
    setModel(MODELS.find(m => m.id === d.modelId) || MODELS[0])
    setPrompt(d.prompt); setSteps(d.steps); setCfg(d.cfg); setDenoise(d.denoise)
    setFeathering(d.feathering); setShowAdvanced(d.showAdvanced)
  }, [resetDefaults])

  // Auto-open import modal when Dashboard sends a pendingImport
  useEffect(() => {
    if (!pendingImport) return
    setImportModal(pendingImport)
    if (onImportConsumed) onImportConsumed()
  }, [pendingImport])

  const handleApplyImport = async (selected) => {
    if (selected.image && importModal?.item) {
      const item = importModal.item

      // If item is a video, use the companion .png (first frame) instead
      let imageUrl
      if (item.type === 'video' && item.filename?.match(/\.(mp4|webm|mov)$/i)) {
        imageUrl = item.url?.replace(/\.(mp4|webm|mov)$/i, '.png')
        console.debug('🎬 Reframe: video detected, using companion image')
      } else {
        imageUrl = getMediaUrl(item.url, item.signed_url)
      }

      try {
        const response = await apiFetch(imageUrl)
        if (!response.ok) throw new Error(`Failed to fetch image: ${response.status}`)
        const blob = await response.blob()
        const filename = imageUrl.split('/').pop() || 'image.png'
        const fileObj = new File([blob], filename, { type: blob.type || 'image/png' })
        const url = URL.createObjectURL(fileObj)
        const img = new Image()
        img.onload = () => {
          setOriginalSize({ width: img.naturalWidth, height: img.naturalHeight })
          setPreview(url)
          setFile(fileObj)
          setResult(null)
          setError(null)
          setLastQueued(null)
          if (DEBUG) console.log('🖼️ Reframe imported image:', filename)
        }
        img.src = url
      } catch (e) {
        console.error('Failed to load image from import:', e)
        setError('⚠️ Failed to load image from import')
      }
    }
    if (selected.positive)  setPrompt(String(selected.positive))
    setImportModal(null)
  }

  const handleCreationsSelect = useCallback(async (item) => {
    try {
      let imageUrl
      if (item.type === 'video' && item.filename?.match(/\.(mp4|webm|mov)$/i)) {
        imageUrl = item.url?.replace(/\.(mp4|webm|mov)$/i, '.png')
      } else {
        imageUrl = getMediaUrl(item.url, item.signed_url)
      }
      const response = await apiFetch(imageUrl)
      if (!response.ok) throw new Error(`Failed to fetch: ${response.status}`)
      const blob = await response.blob()
      const filename = imageUrl.split('/').pop() || 'image.png'
      const fileObj = new File([blob], filename, { type: blob.type || 'image/png' })
      const url = URL.createObjectURL(fileObj)
      const img = new Image()
      img.onload = () => {
        setOriginalSize({ width: img.naturalWidth, height: img.naturalHeight })
        setPreview(url)
        setFile(fileObj)
        setResult(null)
        setError(null)
        setLastQueued(null)
        if (DEBUG) console.log('\ud83d\udcc1 Reframe: loaded from creations:', filename)
      }
      img.src = url
    } catch (e) {
      console.error('Failed to load from creations:', e)
      setError('\u26a0\ufe0f Failed to load image from My Creations')
    }
  }, [])

  const handleFileDrop = useCallback((e) => {
    e.preventDefault()
    const dropped = e.dataTransfer?.files?.[0] || e.target?.files?.[0]
    if (dropped && dropped.type.startsWith('image/')) {
      setFile(dropped)
      setResult(null)
      setError(null)
      setLastQueued(null)

      // Get original dimensions
      const url = URL.createObjectURL(dropped)
      const img = new Image()
      img.onload = () => {
        setOriginalSize({ width: img.naturalWidth, height: img.naturalHeight })
        setPreview(url)
      }
      img.src = url
    }
  }, [])

  const handleDragOver = (e) => e.preventDefault()

  const handleGenerate = async () => {
    // Check if user is logged in
    if (!user) {
      requestLogin('Log in om te genereren')
      return
    }

    if (!file) {
      setError('Please upload an image first')
      return
    }

    setIsLoading(true)
    setError(null)
    setResult(null)
    setLastQueued(null)

    try {
      const formData = new FormData()
      formData.append('image', file)
      formData.append('target_width', aspectRatio.width)
      formData.append('target_height', aspectRatio.height)
      formData.append('position', position)
      formData.append('prompt', prompt || 'seamless natural extension, high quality')
      formData.append('model', model.file)
      formData.append('steps', steps)
      formData.append('cfg', cfg)
      formData.append('denoise', denoise)
      formData.append('feathering', feathering)

      if (DEBUG) console.log('🖼️ Reframe request:', {
        target: `${aspectRatio.width}x${aspectRatio.height}`,
        position, model: model.id
      })

      const res = await postForm(`${BACKEND_BASE}/reframe`, formData)

      if (!res.ok) {
        throw new Error(res.data?.detail || 'Reframe request failed')
      }

      if (res.data?.prompt_id) {
        // Show queued confirmation
        setLastQueued({
          promptId: res.data.prompt_id,
          aspectRatio: aspectRatio.label
        })

        // Notify queue indicator
        if (onJobSubmitted) onJobSubmitted({ prompt_id: res.data.prompt_id })

        if (DEBUG) console.debug('📋 Reframe queued:', res.data.prompt_id)

        // Don't wait for completion - job will appear in queue/history when done
      } else if (res.data?.url) {
        setResult({ url: res.data.url })
      }

    } catch (err) {
      console.error('❌ Reframe error:', err)
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleDownload = () => {
    if (!result?.url) return
    const a = document.createElement('a')
    a.href = result.url
    a.download = `reframed_${aspectRatio.id.replace(':', 'x')}_${Date.now()}.png`
    a.click()
  }

  const calculatePreview = () => {
    if (!originalSize.width || !originalSize.height) return null

    const targetW = aspectRatio.width
    const targetH = aspectRatio.height
    const origW = originalSize.width
    const origH = originalSize.height

    // Scale to fit within target while maintaining aspect
    const scaleW = targetW / origW
    const scaleH = targetH / origH
    const scale = Math.min(scaleW, scaleH)

    const scaledW = Math.round(origW * scale)
    const scaledH = Math.round(origH * scale)

    // Calculate position offsets
    let offsetX = 0, offsetY = 0

    if (position.includes('left')) offsetX = 0
    else if (position.includes('right')) offsetX = targetW - scaledW
    else offsetX = (targetW - scaledW) / 2

    if (position.includes('top')) offsetY = 0
    else if (position.includes('bottom')) offsetY = targetH - scaledH
    else offsetY = (targetH - scaledH) / 2

    return { scaledW, scaledH, offsetX, offsetY, targetW, targetH }
  }

  const previewLayout = calculatePreview()

  return (
    <div className="tool-container">
      {importModal && (
        <MediaImportModal
          item={importModal.item}
          parsedData={importModal.workflow || {}}
          availableFields={['image', 'positive']}
          onApply={handleApplyImport}
          onClose={() => setImportModal(null)}
        />
      )}

      {/* Image Upload Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Frame size={16} />
            Reframe
          </div>
          <ResetDefaultsButton onReset={handleResetDefaults} />
        </div>

        <div
          className="upload-box"
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleFileDrop}
          onDragOver={handleDragOver}
          style={{ cursor: 'pointer' }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileDrop}
            style={{ display: 'none' }}
          />
          {preview ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px', width: '100%' }}>
              <img src={preview} alt="Preview" style={{ maxHeight: '128px', borderRadius: '8px' }} />
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Original: {originalSize.width}×{originalSize.height}
              </span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Click to change</span>
            </div>
          ) : (
            <>
              <Upload size={32} className="text-muted" />
              <div className="text-muted">Drop image here or click to upload</div>
            </>
          )}
        </div>

        <button
          onClick={() => setShowCreationsPicker(true)}
          className="btn-creations-picker"
        >
          📁 From My Creations
        </button>

        <CreationsPickerModal
          show={showCreationsPicker}
          onClose={() => setShowCreationsPicker(false)}
          onSelect={handleCreationsSelect}
          filter="image"
          title="Select Image for Reframe"
        />
      </div>

      {/* Aspect Ratio & Position Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Target Aspect Ratio</div>
          <span className="nav-badge">{aspectRatio.width}×{aspectRatio.height}</span>
        </div>
        <div className="grok-toggle-group" style={{ flexWrap: 'wrap', gap: '4px' }}>
          {ASPECT_RATIOS.map(ar => (
            <button
              key={ar.id}
              onClick={() => setAspectRatio(ar)}
              className={`grok-toggle-btn ${aspectRatio.id === ar.id ? 'active' : ''}`}
              style={{ fontSize: '0.8rem', padding: '6px 10px' }}
            >
              {ar.label}
            </button>
          ))}
        </div>

        <label className="grok-section-label" style={{ marginTop: '16px' }}>Image Position</label>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '6px', width: '160px', margin: '0 auto' }}>
          {['top-left', 'top', 'top-right', 'left', 'center', 'right', 'bottom-left', 'bottom', 'bottom-right'].map(pos => (
            <button
              key={pos}
              onClick={() => setPosition(pos)}
              className={`grok-toggle-btn ${position === pos ? 'active' : ''}`}
              style={{ padding: '8px', fontSize: '1.1rem' }}
              title={pos}
            >
              {POSITIONS.find(p => p.id === pos)?.icon || '○'}
            </button>
          ))}
        </div>
      </div>

      {/* Preview Layout */}
      {previewLayout && (
        <div className="grok-card">
          <div className="grok-card-header">
            <div className="grok-card-title">Layout Preview</div>
          </div>
          <div
            style={{
              position: 'relative', margin: '0 auto',
              border: '1px solid var(--border-color)', background: '#0a0a0a',
              width: Math.min(300, previewLayout.targetW / 3),
              height: Math.min(300, previewLayout.targetH / 3),
              aspectRatio: `${previewLayout.targetW} / ${previewLayout.targetH}`,
              borderRadius: '4px', overflow: 'hidden',
            }}
          >
            <div style={{ position: 'absolute', inset: 0, opacity: 0.3, background: 'repeating-linear-gradient(45deg, transparent, transparent 5px, rgba(255,255,255,0.05) 5px, rgba(255,255,255,0.05) 10px)' }} />
            <div
              style={{
                position: 'absolute',
                background: 'rgba(168, 85, 247, 0.4)', border: '2px solid rgba(168, 85, 247, 0.7)',
                display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.7rem',
                color: 'var(--text-secondary)',
                width: `${(previewLayout.scaledW / previewLayout.targetW) * 100}%`,
                height: `${(previewLayout.scaledH / previewLayout.targetH) * 100}%`,
                left: `${(previewLayout.offsetX / previewLayout.targetW) * 100}%`,
                top: `${(previewLayout.offsetY / previewLayout.targetH) * 100}%`,
              }}
            >
              Original
            </div>
          </div>
          <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textAlign: 'center', marginTop: '8px' }}>
            Purple = original image, striped = AI-generated fill
          </p>
        </div>
      )}

      {/* Prompt & Model Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Generation Settings</div>
        </div>

        <label className="grok-section-label">Fill Prompt (optional)</label>
        <textarea
          className="form-textarea"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe what should appear in the extended areas..."
          rows={2}
        />

        <label className="grok-section-label" style={{ marginTop: '12px' }}>Model</label>
        <div className="grok-toggle-group">
          {MODELS.map(m => (
            <button
              key={m.id}
              onClick={() => setModel(m)}
              className={`grok-toggle-btn ${model.id === m.id ? 'active' : ''}`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {/* Advanced Settings Card */}
      <div className="grok-card" style={{ padding: 0, overflow: 'hidden' }}>
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          style={{
            width: '100%', padding: '14px 20px', background: 'transparent', border: 'none',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            color: 'var(--text-secondary)', cursor: 'pointer',
          }}
        >
          <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Advanced Settings</span>
          <ChevronDown size={16} style={{ transition: 'transform 0.2s', transform: showAdvanced ? 'rotate(180deg)' : 'none' }} />
        </button>

        {showAdvanced && (
          <div style={{ padding: '0 20px 20px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div className="form-group">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label className="grok-section-label">Steps</label>
                <span className="nav-badge">{steps}</span>
              </div>
              <input type="range" className="form-range" min={10} max={50} value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
            </div>

            <div className="form-group">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label className="grok-section-label">CFG Scale</label>
                <span className="nav-badge">{cfg}</span>
              </div>
              <input type="range" className="form-range" min={1} max={15} step={0.5} value={cfg} onChange={(e) => setCfg(Number(e.target.value))} />
            </div>

            <div className="form-group">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label className="grok-section-label">Denoise</label>
                <span className="nav-badge">{denoise.toFixed(2)}</span>
              </div>
              <input type="range" className="form-range" min={0.5} max={1} step={0.05} value={denoise} onChange={(e) => setDenoise(Number(e.target.value))} />
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Higher = more creative fill</span>
            </div>

            <div className="form-group">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label className="grok-section-label">Edge Feathering</label>
                <span className="nav-badge">{feathering}px</span>
              </div>
              <input type="range" className="form-range" min={0} max={64} step={8} value={feathering} onChange={(e) => setFeathering(Number(e.target.value))} />
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Blend between original and fill</span>
            </div>
          </div>
        )}
      </div>

      {error && <div className="status-banner error">{error}</div>}

      {/* Generate Button */}
      <button
        className="primary-btn"
        onClick={handleGenerate}
        disabled={isLoading || !file}
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', height: '48px', fontSize: '1rem' }}
      >
        {isLoading ? (
          <>
            <Loader2 size={18} className="animate-spin" />
            Queueing...
          </>
        ) : (
          <>
            <Frame size={18} />
            Reframe Image
          </>
        )}
      </button>

      {lastQueued && (
        <div className="status-banner success">
          ✅ Reframe job queued! ({lastQueued.aspectRatio}) — Check queue panel for progress
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="grok-card">
          <div className="grok-card-header">
            <div className="grok-card-title">Result</div>
          </div>
          <div style={{ borderRadius: '8px', overflow: 'hidden' }}>
            <img src={result.url} alt="Reframed" style={{ width: '100%', display: 'block' }} />
          </div>
          <div style={{ display: 'flex', gap: '8px', marginTop: '12px' }}>
            <button
              className="primary-btn"
              onClick={handleDownload}
              style={{ flex: 1, height: '40px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}
            >
              <Download size={16} />
              Download
            </button>
            <button
              className="primary-btn"
              onClick={() => {
                setFile(null)
                setPreview(null)
                setResult(null)
                const blob = fetch(result.url).then(r => r.blob()).then(b => {
                  const f = new File([b], 'reframed.png', { type: 'image/png' })
                  setFile(f)
                  setPreview(result.url)
                  const img = new Image()
                  img.onload = () => setOriginalSize({ width: img.naturalWidth, height: img.naturalHeight })
                  img.src = result.url
                })
              }}
              style={{ flex: 1, height: '40px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}
            >
              <Move size={16} />
              Use as Input
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
