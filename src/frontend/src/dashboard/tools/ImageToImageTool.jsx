import React, { useState, useCallback, useEffect } from 'react'
import { Upload, Wand2, Loader2, Image as ImageIcon, Settings, ChevronDown, Sliders } from 'lucide-react'
import { BACKEND_BASE, DEBUG, getMediaUrl } from '../../config'
import { postForm } from '../../api'
import { useAuth } from '../../contexts/AuthContext'
import MediaImportModal from '../../components/MediaImportModal'
import CreationsPickerModal from '../../components/CreationsPickerModal'

const ASPECT_RATIOS = ['1:1', '16:9', '9:16', '4:3', '3:4']

const CHECKPOINTS = [
  { value: 'CyberRealistic_Pony_v14.1_FP16.safetensors', label: 'CyberRealistic Pony' },
  { value: 'dreamshaperXL_lightningDPMSDE.safetensors', label: 'Dreamshaper Lightning' },
  { value: 'juggernautXL_ragnarok.safetensors', label: 'Juggernaut XL' },
  { value: 'waiIllustriousSDXL_v160.safetensors', label: 'Wai Illustrious (Anime)' },
]

export default function ImageToImageTool({ onOutput, onJobSubmitted, pendingImport, onImportConsumed }) {
  const { user, requestLogin } = useAuth()

  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [importModal, setImportModal] = useState(null)
  const [showCreationsPicker, setShowCreationsPicker] = useState(false)
  const [prompt, setPrompt] = useState('')
  const [negativePrompt, setNegativePrompt] = useState('ugly, deformed, blurry, low quality, bad anatomy, watermark')
  const [denoise, setDenoise] = useState(0.6)
  const [checkpoint, setCheckpoint] = useState('CyberRealistic_Pony_v14.1_FP16.safetensors')

  // Advanced
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [steps, setSteps] = useState(25)
  const [cfg, setCfg] = useState(7.0)
  const [seed, setSeed] = useState(-1)
  const [sampler, setSampler] = useState('dpmpp_2m')
  const [scheduler, setScheduler] = useState('karras')

  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)
  const [lastQueued, setLastQueued] = useState(null)
  const [result, setResult] = useState(null)

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
        const response = await fetch(imageUrl)
        if (!response.ok) throw new Error(`Failed to fetch image: ${response.status}`)
        const blob = await response.blob()
        const fileObj = new File([blob], imageFilename, { type: blob.type || 'image/png' })
        setFile(fileObj)
        setPreview(imageUrl)
        setResult(null)
        setError(null)
        setLastQueued(null)
        if (DEBUG) console.log('🖼️ I2I imported image:', imageFilename)
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
      const response = await fetch(imageUrl)
      if (!response.ok) throw new Error(`Failed to fetch: ${response.status}`)
      const blob = await response.blob()
      const fileObj = new File([blob], imageFilename, { type: blob.type || 'image/png' })
      setFile(fileObj)
      setPreview(imageUrl)
      setResult(null)
      setError(null)
      if (DEBUG) console.log('📁 I2I: loaded from creations:', imageFilename)
    } catch (e) {
      console.error('Failed to load from creations:', e)
      setError('⚠️ Failed to load image from My Creations')
    }
  }, [])

  const handleFileChange = useCallback((e) => {
    const f = e.target.files?.[0]
    if (f) {
      setFile(f)
      setPreview(URL.createObjectURL(f))
      setResult(null)
      setError(null)
      setLastQueued(null)
      if (DEBUG) console.log('🖼️ I2I file selected:', f.name, `(${(f.size / 1024).toFixed(0)}KB)`)
    }
    // Reset input value so re-selecting the same file triggers onChange
    e.target.value = ''
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0]
    if (f && f.type.startsWith('image/')) {
      setFile(f)
      setPreview(URL.createObjectURL(f))
      setResult(null)
      setError(null)
      setLastQueued(null)
      if (DEBUG) console.log('🖼️ I2I file dropped:', f.name, `(${(f.size / 1024).toFixed(0)}KB)`)
    }
  }, [])

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
      const formData = new FormData()
      formData.append('file', file)
      formData.append('prompt', prompt || 'high quality, detailed')
      formData.append('negative_prompt', negativePrompt)
      formData.append('denoise', String(denoise))
      formData.append('checkpoint', checkpoint)
      formData.append('steps', String(steps))
      formData.append('cfg', String(cfg))
      formData.append('seed', String(seed))
      formData.append('sampler_name', sampler)
      formData.append('scheduler', scheduler)

      if (DEBUG) console.debug('🖼️ I2I request:', { denoise, checkpoint, steps })

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
        <h3>
          <ImageIcon size={18} />
          Source Image
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

      <CreationsPickerModal
        show={showCreationsPicker}
        onClose={() => setShowCreationsPicker(false)}
        onSelect={handleCreationsSelect}
        filter="image"
        title="Select Image from My Creations"
      />

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
        }
        .upload-preview {
          max-width: 100%;
          max-height: 300px;
          border-radius: 8px;
          object-fit: contain;
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
