import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react'
import { Upload, Wand2, Loader2, Image as ImageIcon, Settings, ChevronDown, Sliders, X, Zap, Shield, User as UserIcon, Sparkles } from 'lucide-react'
import { BACKEND_BASE, DEBUG, getMediaUrl } from '../../config'
import { postForm, apiFetch } from '../../api'
import { useAuth } from '../../contexts/AuthContext'
import MediaImportModal from '../../components/MediaImportModal'
import CreationsPickerModal from '../../components/CreationsPickerModal'
import { useToolSettings } from '../../hooks/useToolSettings'
import ResetDefaultsButton from '../../components/ResetDefaultsButton'

const CHECKPOINTS = [
  { value: 'CyberRealistic_Pony_v14.1_FP16.safetensors', label: 'CyberRealistic Pony' },
  { value: 'dreamshaperXL_lightningDPMSDE.safetensors', label: 'Dreamshaper Lightning' },
  { value: 'juggernautXL_ragnarok.safetensors', label: 'Juggernaut XL' },
  { value: 'Realistic_Vision_V5.1.safetensors', label: 'Realistic Vision V5.1' },
  { value: 'waiIllustriousSDXL_v160.safetensors', label: 'Wai Illustrious (Anime)' },
  { value: 'ponyDiffusionV6XL_v6StartWithThisOne.safetensors', label: 'Pony Diffusion V6' },
  { value: 'illustriousRealismBy_v10VAE.safetensors', label: 'Illustrious Realism' },
  { value: 'reapony_v90.safetensors', label: 'ReaPony V9' },
]

const PRESETS = [
  { value: 'fast', label: 'Fast', icon: '⚡', desc: 'Quick transform, no face processing', color: '#22c55e' },
  { value: 'balanced', label: 'Balanced', icon: '⚖️', desc: 'Good quality + face refinement', color: '#3b82f6' },
  { value: 'face_preserve', label: 'Face Preserve', icon: '🛡️', desc: 'Best for keeping faces consistent', color: '#a855f7' },
  { value: 'custom', label: 'Custom', icon: '🔧', desc: 'Full manual control', color: '#f59e0b' },
]

const I2I_DEFAULTS = {
  prompt: '', negativePrompt: 'ugly, deformed, blurry, low quality, bad anatomy, watermark',
  denoise: 0.6, checkpoint: 'CyberRealistic_Pony_v14.1_FP16.safetensors', preset: 'balanced',
  faceId: false, faceDetailer: true, faceRestore: true, faceIdWeight: 0.85,
  steps: 25, cfg: 7.0, seed: -1, sampler: 'dpmpp_2m', scheduler: 'karras',
}

export default function ImageToImageTool({ onOutput, onJobSubmitted, pendingImport, onImportConsumed }) {
  const { user, requestLogin } = useAuth()
  const { initial, save: saveSettings, resetDefaults } = useToolSettings('image_to_image', I2I_DEFAULTS)

  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const fileInputKey = useRef(0)
  const fileRef = useRef(null)  // Authoritative file reference (bypasses React state batching)
  const [importModal, setImportModal] = useState(null)
  const [showCreationsPicker, setShowCreationsPicker] = useState(false)
  const [prompt, setPrompt] = useState(initial.prompt)
  const [negativePrompt, setNegativePrompt] = useState(initial.negativePrompt)
  const [denoise, setDenoise] = useState(initial.denoise)
  const [checkpoint, setCheckpoint] = useState(initial.checkpoint)

  // Preset
  const [preset, setPreset] = useState(initial.preset)

  // Face processing
  const [faceId, setFaceId] = useState(initial.faceId)
  const [faceDetailer, setFaceDetailer] = useState(initial.faceDetailer)
  const [faceRestore, setFaceRestore] = useState(initial.faceRestore)
  const [faceIdWeight, setFaceIdWeight] = useState(initial.faceIdWeight)

  // Advanced
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [steps, setSteps] = useState(initial.steps)
  const [cfg, setCfg] = useState(initial.cfg)
  const [seed, setSeed] = useState(initial.seed)
  const [sampler, setSampler] = useState(initial.sampler)
  const [scheduler, setScheduler] = useState(initial.scheduler)

  // ── Auto-save settings ──────────────────────────────────────────
  const settingsSnapshot = useMemo(() => ({
    prompt, negativePrompt, denoise, checkpoint, preset,
    faceId, faceDetailer, faceRestore, faceIdWeight,
    steps, cfg, seed, sampler, scheduler,
  }), [prompt, negativePrompt, denoise, checkpoint, preset,
    faceId, faceDetailer, faceRestore, faceIdWeight,
    steps, cfg, seed, sampler, scheduler])
  useEffect(() => { saveSettings(settingsSnapshot) }, [settingsSnapshot, saveSettings])

  const handleResetDefaults = useCallback(() => {
    const d = resetDefaults()
    setPrompt(d.prompt); setNegativePrompt(d.negativePrompt); setDenoise(d.denoise)
    setCheckpoint(d.checkpoint); setPreset(d.preset)
    setFaceId(d.faceId); setFaceDetailer(d.faceDetailer); setFaceRestore(d.faceRestore)
    setFaceIdWeight(d.faceIdWeight)
    setSteps(d.steps); setCfg(d.cfg); setSeed(d.seed); setSampler(d.sampler); setScheduler(d.scheduler)
  }, [resetDefaults])

  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)
  const [lastQueued, setLastQueued] = useState(null)
  const [result, setResult] = useState(null)

  // Update settings when preset changes
  useEffect(() => {
    if (preset === 'fast') {
      setSteps(15); setCfg(7.0); setSampler('dpmpp_2m'); setScheduler('karras')
      setFaceId(false); setFaceDetailer(false); setFaceRestore(false)
    } else if (preset === 'balanced') {
      setSteps(25); setCfg(7.0); setSampler('dpmpp_2m'); setScheduler('karras')
      setFaceId(false); setFaceDetailer(true); setFaceRestore(true)
    } else if (preset === 'face_preserve') {
      setSteps(30); setCfg(7.5); setSampler('dpmpp_2m_sde'); setScheduler('karras')
      setFaceId(true); setFaceDetailer(true); setFaceRestore(true)
    }
    // custom = no auto-change
  }, [preset])

  // Auto-open import modal when Dashboard sends a pendingImport
  useEffect(() => {
    if (!pendingImport) return
    setImportModal(pendingImport)
    if (onImportConsumed) onImportConsumed()
  }, [pendingImport])

  // Helper: set file in both state (for UI) and ref (for submission)
  const updateFile = useCallback((f, previewUrl) => {
    fileRef.current = f
    setFile(f)
    setPreview(previewUrl || (f ? URL.createObjectURL(f) : null))
    setResult(null)
    setError(null)
    setLastQueued(null)
    console.warn(`🖼️ I2I file set: ${f?.name || 'null'} (${f ? (f.size / 1024).toFixed(0) + 'KB' : '-'}, type=${f?.type || '-'})`)
  }, [])

  const handleApplyImport = async (selected) => {
    if (selected.image && importModal?.item) {
      const item = importModal.item

      // If item is a video, use the companion .png (first frame) instead
      let imageUrl, imageFilename
      if (item.type === 'video' && item.filename?.match(/\.(mp4|webm|mov)$/i)) {
        const pngFilename = item.filename.replace(/\.(mp4|webm|mov)$/i, '.png')
        const pngUrl = item.url?.replace(/\.(mp4|webm|mov)$/i, '.png')
        imageUrl = pngUrl
        imageFilename = pngFilename
        console.debug('🎬 I2I: video detected, using companion image:', pngFilename)
      } else {
        imageUrl = getMediaUrl(item.url, item.signed_url)
        imageFilename = item.filename || imageUrl.split('/').pop()
      }

      try {
        const response = await apiFetch(imageUrl)
        if (!response.ok) throw new Error(`Failed to fetch image: ${response.status}`)
        const blob = await response.blob()
        const fileObj = new File([blob], imageFilename, { type: blob.type || 'image/png' })
        updateFile(fileObj, imageUrl)
      } catch (e) {
        console.error('Failed to load image from import:', e)
        setError('⚠️ Failed to load image from import')
      }
    }
    if (selected.positive)  setPrompt(String(selected.positive))
    if (selected.negative)  setNegativePrompt(String(selected.negative))
    if (selected.steps)     setSteps(Number(selected.steps) || steps)
    if (selected.cfg)       setCfg(Number(selected.cfg) || cfg)
    if (selected.seed)      setSeed(Number(selected.seed) || seed)
    setImportModal(null)
  }

  const handleCreationsSelect = useCallback(async (item) => {
    try {
      let imageUrl, imageFilename
      if (item.type === 'video' && item.filename?.match(/\.(mp4|webm|mov)$/i)) {
        imageUrl = item.url?.replace(/\.(mp4|webm|mov)$/i, '.png')
        imageFilename = item.filename.replace(/\.(mp4|webm|mov)$/i, '.png')
      } else {
        imageUrl = getMediaUrl(item.url, item.signed_url)
        imageFilename = item.filename || imageUrl.split('/').pop()
      }
      const response = await apiFetch(imageUrl)
      if (!response.ok) throw new Error(`Failed to fetch: ${response.status}`)
      const blob = await response.blob()
      const fileObj = new File([blob], imageFilename, { type: blob.type || 'image/png' })
      updateFile(fileObj, imageUrl)
    } catch (e) {
      console.error('Failed to load from creations:', e)
      setError('⚠️ Failed to load image from My Creations')
    }
  }, [updateFile])

  const handleFileChange = useCallback((e) => {
    const f = e.target.files?.[0]
    if (f) {
      updateFile(f, URL.createObjectURL(f))
    }
    // Reset input value so re-selecting the same file triggers onChange
    e.target.value = ''
  }, [updateFile])

  const handleClearImage = useCallback((e) => {
    e?.stopPropagation()
    updateFile(null, null)
    fileInputKey.current += 1
  }, [updateFile])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0]
    if (f && f.type.startsWith('image/')) {
      updateFile(f, URL.createObjectURL(f))
    }
  }, [updateFile])

  const handleGenerate = async () => {
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
      // Use ref for the file — guaranteed fresh, not subject to React state batching
      const currentFile = fileRef.current
      if (!currentFile) {
        setError('No file selected')
        setSubmitting(false)
        return
      }

      // ALWAYS compute and log hash for debugging
      const buf = await currentFile.arrayBuffer()
      const hashBuf = await crypto.subtle.digest('SHA-256', buf)
      const hashHex = [...new Uint8Array(hashBuf)].map(b => b.toString(16).padStart(2, '0')).join('').substring(0, 16)
      console.warn(`🔐 I2I SENDING: hash=${hashHex} | name=${currentFile.name} | size=${currentFile.size} | type=${currentFile.type}`)

      // Re-create File from the arrayBuffer we just read (guarantees fresh copy)
      const freshBlob = new Blob([buf], { type: currentFile.type || 'image/png' })
      const freshFile = new File([freshBlob], currentFile.name, {
        type: currentFile.type || 'image/png',
        lastModified: currentFile.lastModified
      })

      const formData = new FormData()
      formData.append('file', freshFile)
      formData.append('prompt', prompt || 'high quality, detailed')
      formData.append('negative_prompt', negativePrompt)
      formData.append('denoise', String(denoise))
      formData.append('checkpoint', checkpoint)
      formData.append('steps', String(steps))
      formData.append('cfg', String(cfg))
      formData.append('seed', String(seed))
      formData.append('sampler_name', sampler)
      formData.append('scheduler', scheduler)
      formData.append('preset', preset)
      formData.append('face_id', String(faceId))
      formData.append('face_detailer', String(faceDetailer))
      formData.append('face_restore', String(faceRestore))
      formData.append('face_id_weight', String(faceIdWeight))

      if (DEBUG) console.debug('🖼️ I2I request:', {
        fileName: file?.name,
        fileSize: file?.size,
        fileType: file?.type,
        fileLastModified: file?.lastModified,
        denoise, checkpoint, steps,
        previewUrl: preview?.substring?.(0, 80)
      })

      const res = await postForm(`${BACKEND_BASE}/generate-i2i`, formData)

      if (!res.ok) {
        throw new Error(res.data?.detail || 'Generation failed')
      }

      const promptId = res.data?.prompt_id
      if (!promptId) {
        throw new Error('No prompt_id returned')
      }

      // Show queued confirmation
      setLastQueued({
        promptId,
        checkpoint: CHECKPOINTS.find(c => c.value === checkpoint)?.label || checkpoint
      })

      // Notify queue indicator
      if (onJobSubmitted) onJobSubmitted({ prompt_id: promptId })

      if (DEBUG) console.debug('📋 I2I queued:', promptId)

      // Don't wait for completion - job will appear in queue/history when done

    } catch (err) {
      console.error('I2I error:', err)
      setError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="tool-container">
      <div className="tool-section">
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <ImageIcon size={18} />
          Source Image
          <ResetDefaultsButton onReset={handleResetDefaults} />
        </h3>

        <div
          className={`upload-dropzone ${preview ? 'has-preview' : ''}`}
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => document.getElementById('i2i-file-input').click()}
        >
          {preview ? (
            <>
              <img src={preview} alt="Preview" className="upload-preview" />
              <button
                className="btn-clear-image"
                onClick={handleClearImage}
                title="Clear image"
              >
                <X size={16} />
              </button>
              {file && (
                <div className="upload-filename">
                  📎 {file.name} ({(file.size / 1024).toFixed(0)}KB)
                </div>
              )}
            </>
          ) : (
            <div className="upload-placeholder">
              <Upload size={32} />
              <p>Drop image here or click to upload</p>
            </div>
          )}
          <input
            key={fileInputKey.current}
            id="i2i-file-input"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
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
          title="Select Image from My Creations"
        />
      </div>

      <div className="tool-section">
        <h3>
          <Wand2 size={18} />
          Transformation
        </h3>

        <div className="form-group">
          <label>Prompt (describe desired changes)</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            placeholder="Describe what you want the image to become... (e.g., 'anime style illustration')"
          />
        </div>

        <div className="form-group">
          <label>
            <Sliders size={14} />
            Denoise Strength
            <span className="label-value">{denoise.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min="0.1"
            max="1.0"
            step="0.05"
            value={denoise}
            onChange={(e) => setDenoise(parseFloat(e.target.value))}
          />
          <div className="range-labels">
            <span>Subtle (0.1)</span>
            <span>Complete (1.0)</span>
          </div>
          <div className="denoise-hint">
            {denoise < 0.3 && '💡 Minor adjustments, preserves most of original'}
            {denoise >= 0.3 && denoise < 0.6 && '💡 Moderate changes, good balance'}
            {denoise >= 0.6 && denoise < 0.8 && '💡 Significant transformation'}
            {denoise >= 0.8 && '💡 Near-complete regeneration from prompt'}
          </div>
        </div>

        <div className="form-group">
          <label>Model</label>
          <select value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)}>
            {CHECKPOINTS.map((c) => (
              <option key={c.value} value={c.value}>{c.label}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Quality Preset */}
      <div className="tool-section">
        <h3>
          <Zap size={18} />
          Quality Preset
        </h3>
        <div className="preset-grid">
          {PRESETS.map((p) => (
            <button
              key={p.value}
              className={`preset-card ${preset === p.value ? 'active' : ''}`}
              onClick={() => setPreset(p.value)}
              style={preset === p.value ? { borderColor: p.color, boxShadow: `0 0 12px ${p.color}33` } : {}}
            >
              <span className="preset-icon">{p.icon}</span>
              <span className="preset-label">{p.label}</span>
              <span className="preset-desc">{p.desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Face Processing */}
      <div className="tool-section">
        <h3>
          <UserIcon size={18} />
          Face Processing
        </h3>

        <div className="face-toggles">
          <label className="toggle-row" title="IP-Adapter FaceID: Extracts face identity from source image and preserves it during generation">
            <div className="toggle-info">
              <span className="toggle-label">🛡️ Face Identity (IP-Adapter)</span>
              <span className="toggle-desc">Preserves face from source image</span>
            </div>
            <input
              type="checkbox"
              checked={faceId}
              onChange={(e) => { setFaceId(e.target.checked); if (preset !== 'custom') setPreset('custom') }}
            />
            <span className="toggle-slider" />
          </label>

          {faceId && (
            <div className="form-group face-weight-slider">
              <label>
                FaceID Strength
                <span className="label-value">{faceIdWeight.toFixed(2)}</span>
              </label>
              <input
                type="range"
                min="0.3"
                max="1.0"
                step="0.05"
                value={faceIdWeight}
                onChange={(e) => setFaceIdWeight(parseFloat(e.target.value))}
              />
              <div className="range-labels">
                <span>Subtle (0.3)</span>
                <span>Strong (1.0)</span>
              </div>
            </div>
          )}

          <label className="toggle-row" title="FaceDetailer: Auto-detects faces and refines them with a second pass using YOLO + SAM">
            <div className="toggle-info">
              <span className="toggle-label">✨ Face Detailer</span>
              <span className="toggle-desc">Auto-detect &amp; refine faces</span>
            </div>
            <input
              type="checkbox"
              checked={faceDetailer}
              onChange={(e) => { setFaceDetailer(e.target.checked); if (preset !== 'custom') setPreset('custom') }}
            />
            <span className="toggle-slider" />
          </label>

          <label className="toggle-row" title="GFPGAN v1.4: Final face quality enhancement pass for photorealistic faces">
            <div className="toggle-info">
              <span className="toggle-label">💎 Face Restore (GFPGAN)</span>
              <span className="toggle-desc">Final polish on face quality</span>
            </div>
            <input
              type="checkbox"
              checked={faceRestore}
              onChange={(e) => { setFaceRestore(e.target.checked); if (preset !== 'custom') setPreset('custom') }}
            />
            <span className="toggle-slider" />
          </label>
        </div>
      </div>

      {/* Advanced Settings */}
      <div className="tool-section collapsible">
        <button
          className="section-toggle"
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          <Settings size={16} />
          Advanced Settings
          <ChevronDown size={16} className={showAdvanced ? 'rotated' : ''} />
        </button>

        {showAdvanced && (
          <div className="advanced-content">
            <div className="form-group">
              <label>Negative Prompt</label>
              <textarea
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
                rows={2}
              />
            </div>

            <div className="form-row">
              <div className="form-group half">
                <label>Steps</label>
                <input
                  type="number"
                  value={steps}
                  onChange={(e) => setSteps(parseInt(e.target.value) || 25)}
                  min="1"
                  max="50"
                />
              </div>
              <div className="form-group half">
                <label>CFG Scale</label>
                <input
                  type="number"
                  value={cfg}
                  onChange={(e) => setCfg(parseFloat(e.target.value) || 7.0)}
                  min="1"
                  max="20"
                  step="0.5"
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group half">
                <label>Sampler</label>
                <select value={sampler} onChange={(e) => setSampler(e.target.value)}>
                  <option value="euler">Euler</option>
                  <option value="euler_ancestral">Euler Ancestral</option>
                  <option value="dpmpp_2m">DPM++ 2M</option>
                  <option value="dpmpp_2m_sde">DPM++ 2M SDE</option>
                  <option value="dpmpp_3m_sde">DPM++ 3M SDE</option>
                </select>
              </div>
              <div className="form-group half">
                <label>Scheduler</label>
                <select value={scheduler} onChange={(e) => setScheduler(e.target.value)}>
                  <option value="normal">Normal</option>
                  <option value="karras">Karras</option>
                  <option value="exponential">Exponential</option>
                  <option value="sgm_uniform">SGM Uniform</option>
                </select>
              </div>
            </div>

            <div className="form-group">
              <label>Seed (-1 = random)</label>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(parseInt(e.target.value) || -1)}
              />
            </div>
          </div>
        )}
      </div>

      {/* Queued notification */}
      {lastQueued && (
        <div className="queued-notice">
          ✅ Job queued! Check the Queue panel for progress.
          <span className="queued-mode">{lastQueued.checkpoint}</span>
        </div>
      )}

      {error && <div className="error-message">⚠️ {error}</div>}

      <button
        className="btn-primary btn-large"
        onClick={handleGenerate}
        disabled={!file || submitting}
      >
        {submitting ? (
          <>
            <Loader2 size={18} className="spin" />
            Queueing...
          </>
        ) : (
          <>
            <Wand2 size={18} />
            Transform Image
            {(faceId || faceDetailer || faceRestore) && (
              <span className="btn-badge">
                {[faceId && '🛡️', faceDetailer && '✨', faceRestore && '💎'].filter(Boolean).join('')}
              </span>
            )}
          </>
        )}
      </button>

      {/* Result */}
      {result && (
        <div className="result-section">
          <h3>Result</h3>
          <div className="comparison">
            <div className="comparison-item">
              <span className="comparison-label">Original</span>
              <img src={preview} alt="Original" />
            </div>
            <div className="comparison-item">
              <span className="comparison-label">Transformed</span>
              <img src={result} alt="Result" />
            </div>
          </div>
        </div>
      )}

      {importModal && (
        <MediaImportModal
          item={importModal.item}
          parsedData={importModal.workflow || {}}
          availableFields={['image', 'positive', 'negative', 'steps', 'cfg', 'seed']}
          onApply={handleApplyImport}
          onClose={() => setImportModal(null)}
        />
      )}

      <style>{`
        .upload-dropzone {
          border: 2px dashed var(--border-color, #444);
          border-radius: 12px;
          padding: 40px;
          text-align: center;
          cursor: pointer;
          transition: all 0.2s;
          min-height: 200px;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .upload-dropzone:hover {
          border-color: var(--accent-color, #7c3aed);
          background: rgba(124, 58, 237, 0.05);
        }
        .upload-dropzone.has-preview {
          padding: 8px;
          flex-direction: column;
          position: relative;
        }
        .upload-preview {
          max-width: 100%;
          max-height: 300px;
          border-radius: 8px;
          object-fit: contain;
        }
        .btn-clear-image {
          position: absolute;
          top: 12px;
          right: 12px;
          background: rgba(0, 0, 0, 0.6);
          color: white;
          border: none;
          border-radius: 50%;
          width: 28px;
          height: 28px;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          padding: 0;
          z-index: 2;
        }
        .btn-clear-image:hover {
          background: rgba(220, 50, 50, 0.8);
        }
        .upload-filename {
          margin-top: 6px;
          font-size: 11px;
          color: var(--text-muted, #888);
          text-align: center;
          word-break: break-all;
          max-width: 100%;
        }
        .upload-placeholder {
          color: var(--text-muted, #888);
        }
        .upload-placeholder p {
          margin-top: 12px;
        }

        /* Preset Grid */
        .preset-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 8px;
        }
        .preset-card {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 4px;
          padding: 12px 8px;
          border: 1px solid var(--border-color, #333);
          border-radius: 10px;
          background: var(--bg-secondary, #1a1a1a);
          cursor: pointer;
          transition: all 0.2s;
          text-align: center;
        }
        .preset-card:hover {
          border-color: var(--text-muted, #666);
          background: var(--bg-tertiary, #222);
        }
        .preset-card.active {
          background: var(--bg-tertiary, #252525);
        }
        .preset-icon {
          font-size: 20px;
        }
        .preset-label {
          font-size: 12px;
          font-weight: 600;
          color: var(--text-color, #fff);
        }
        .preset-desc {
          font-size: 10px;
          color: var(--text-muted, #888);
          line-height: 1.3;
        }

        /* Face Processing Toggles */
        .face-toggles {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .toggle-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 10px 12px;
          border: 1px solid var(--border-color, #333);
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.15s;
          background: var(--bg-secondary, #1a1a1a);
        }
        .toggle-row:hover {
          border-color: var(--text-muted, #555);
        }
        .toggle-info {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }
        .toggle-label {
          font-size: 13px;
          font-weight: 500;
          color: var(--text-color, #fff);
        }
        .toggle-desc {
          font-size: 11px;
          color: var(--text-muted, #888);
        }
        .toggle-row input[type="checkbox"] {
          appearance: none;
          -webkit-appearance: none;
          width: 40px;
          height: 22px;
          background: var(--border-color, #444);
          border-radius: 12px;
          position: relative;
          cursor: pointer;
          transition: background 0.2s;
          flex-shrink: 0;
        }
        .toggle-row input[type="checkbox"]:checked {
          background: var(--accent-color, #7c3aed);
        }
        .toggle-row input[type="checkbox"]::after {
          content: '';
          position: absolute;
          top: 3px;
          left: 3px;
          width: 16px;
          height: 16px;
          background: white;
          border-radius: 50%;
          transition: transform 0.2s;
        }
        .toggle-row input[type="checkbox"]:checked::after {
          transform: translateX(18px);
        }
        .toggle-row .toggle-slider {
          display: none;
        }
        .face-weight-slider {
          margin: 0 0 4px 0;
          padding: 8px 12px;
          background: var(--bg-tertiary, #1e1e1e);
          border-radius: 8px;
          border: 1px solid var(--border-color, #333);
        }
        .face-weight-slider label {
          margin-bottom: 6px !important;
        }

        /* Button Badge */
        .btn-badge {
          margin-left: 8px;
          font-size: 14px;
        }

        .form-group {
          margin-bottom: 16px;
        }
        .form-group label {
          display: flex;
          align-items: center;
          gap: 6px;
          margin-bottom: 8px;
          font-size: 13px;
          color: var(--text-secondary, #aaa);
        }
        .label-value {
          margin-left: auto;
          color: var(--accent-color, #7c3aed);
          font-weight: 500;
        }
        .form-group textarea,
        .form-group select,
        .form-group input[type="number"] {
          width: 100%;
          padding: 10px 12px;
          border-radius: 8px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          font-size: 13px;
        }
        .range-labels {
          display: flex;
          justify-content: space-between;
          font-size: 11px;
          color: var(--text-muted, #888);
          margin-top: 4px;
        }
        .denoise-hint {
          margin-top: 8px;
          font-size: 12px;
          color: var(--text-muted, #888);
        }
        .form-row {
          display: flex;
          gap: 16px;
        }
        .form-group.half {
          flex: 1;
        }
        .section-toggle {
          display: flex;
          align-items: center;
          gap: 8px;
          width: 100%;
          padding: 12px;
          background: transparent;
          border: 1px solid var(--border-color, #333);
          border-radius: 8px;
          color: var(--text-secondary, #aaa);
          cursor: pointer;
          font-size: 13px;
        }
        .section-toggle:hover {
          border-color: var(--border-color, #555);
        }
        .section-toggle .rotated {
          transform: rotate(180deg);
        }
        .section-toggle svg:last-child {
          margin-left: auto;
          transition: transform 0.2s;
        }
        .advanced-content {
          margin-top: 16px;
          padding-top: 16px;
          border-top: 1px solid var(--border-color, #333);
        }
        .progress-section {
          margin: 16px 0;
        }
        .progress-bar {
          height: 4px;
          background: var(--bg-secondary, #333);
          border-radius: 2px;
          overflow: hidden;
        }
        .progress-fill {
          height: 100%;
          background: var(--accent-color, #7c3aed);
          transition: width 0.3s;
        }
        .progress-status {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-top: 8px;
          font-size: 13px;
          color: var(--text-secondary, #aaa);
        }
        .error-message {
          padding: 12px;
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 8px;
          color: #ef4444;
          margin: 12px 0;
        }
        .result-section {
          margin-top: 24px;
          padding-top: 24px;
          border-top: 1px solid var(--border-color, #333);
        }
        .comparison {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 16px;
          margin-top: 16px;
        }
        .comparison-item {
          position: relative;
        }
        .comparison-label {
          position: absolute;
          top: 8px;
          left: 8px;
          background: rgba(0,0,0,0.7);
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 11px;
        }
        .comparison-item img {
          width: 100%;
          border-radius: 8px;
        }
        .queued-notice .queued-features {
          display: flex;
          gap: 6px;
          margin-top: 4px;
          font-size: 11px;
          color: var(--text-muted, #888);
        }
        .spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}
