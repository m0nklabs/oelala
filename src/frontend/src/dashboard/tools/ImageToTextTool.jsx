import React, { useState, useCallback, useEffect, useMemo } from 'react'
import { Upload, Wand2, Copy, Send, Loader2, Image as ImageIcon, Pencil } from 'lucide-react'
import { BACKEND_BASE, DEBUG, getMediaUrl } from '../../config'
import { apiFetch } from '../../api'
import MediaImportModal from '../../components/MediaImportModal'
import CreationsPickerModal from '../../components/CreationsPickerModal'
import useLLMEnhance from '../../hooks/useLLMEnhance'
import LLMQueueIndicator from '../../components/LLMQueueIndicator'
import { VISION_MODELS, DEFAULT_VISION_MODEL } from '../../constants/llmModels'
import { useAuth } from '../../contexts/AuthContext'
import { useToolSettings } from '../../hooks/useToolSettings'
import ResetDefaultsButton from '../../components/ResetDefaultsButton'
import CameraMotionSelector, { getCameraMotionPrefix } from '../../components/CameraMotionSelector'

const CAPTION_MODES = [
  { id: 'brief', label: 'Brief', description: '1-line summary', group: 'caption' },
  { id: 'detailed', label: 'Detailed', description: 'Full paragraph', group: 'caption' },
  { id: 'tags', label: 'Tags', description: 'Comma-separated keywords', group: 'caption' },
  { id: 'structured', label: 'Structured', description: 'Subject, style, mood', group: 'caption' },
  { id: 'prompt_i2v', label: '🎬 I2V Prompt', description: 'Motion & activity for video gen', group: 'prompt' },
  { id: 'prompt_t2i', label: '🖼️ T2I Prompt', description: 'Tag-style for image gen', group: 'prompt' },
  { id: 'prompt_nsfw', label: '🔞 NSFW Prompt', description: 'Explicit & uncensored', group: 'prompt' },
]

const NSFW_LEVELS = [
  { value: 1, label: 'Sensual', emoji: '💋', description: 'Suggestive poses, lingerie, soft lighting' },
  { value: 2, label: 'Softcore', emoji: '🔥', description: 'Partial nudity, erotic posing, teasing' },
  { value: 3, label: 'Nude', emoji: '🍑', description: 'Full nudity, explicit body details' },
  { value: 4, label: 'Hardcore', emoji: '🍆', description: 'Explicit sexual acts, positions' },
  { value: 5, label: 'Extreme', emoji: '⛓️', description: 'Rough, aggressive, extreme acts' },
]

const isPromptMode = (m) => m.startsWith('prompt_')

// VISION_MODELS imported from shared constants/llmModels.js
const MODELS = VISION_MODELS

const I2T_DEFAULTS = {
  model: DEFAULT_VISION_MODEL, mode: 'detailed', nsfwIntensity: 3, includeMotion: false, cameraMotion: '',
}

export default function ImageToTextTool({ onSendToPrompt, pendingImport = null, onImportConsumed = null }) {
  const { user, requestLogin } = useAuth()
  const { initial, save: saveSettings, resetDefaults } = useToolSettings('image_to_text', I2T_DEFAULTS)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [model, setModel] = useState(initial.model)
  const [mode, setMode] = useState(initial.mode)
  const [caption, setCaption] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [isRefining, setIsRefining] = useState(false)
  const [showRefineInput, setShowRefineInput] = useState(false)
  const [refineInstruction, setRefineInstruction] = useState('')
  const [importModal, setImportModal] = useState(null)  // { item, workflow }
  const [showCreationsPicker, setShowCreationsPicker] = useState(false)
  const [nsfwIntensity, setNsfwIntensity] = useState(initial.nsfwIntensity)  // 1-5 NSFW intensity scale
  const [includeMotion, setIncludeMotion] = useState(initial.includeMotion || false)
  const [cameraMotion, setCameraMotion] = useState(initial.cameraMotion || '')

  // ── Auto-save settings ──────────────────────────────────────────
  const settingsSnapshot = useMemo(() => ({ model, mode, nsfwIntensity, includeMotion, cameraMotion }), [model, mode, nsfwIntensity, includeMotion, cameraMotion])
  useEffect(() => { saveSettings(settingsSnapshot) }, [settingsSnapshot, saveSettings])

  const handleResetDefaults = useCallback(() => {
    const d = resetDefaults()
    setModel(d.model); setMode(d.mode); setNsfwIntensity(d.nsfwIntensity)
    setIncludeMotion(d.includeMotion || false); setCameraMotion(d.cameraMotion || '')
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
      let imageUrl, fetchUrl, imageFilename
      if (item.type === 'video' && item.filename?.match(/\.(mp4|webm|mov)$/i)) {
        const pngFilename = item.filename.replace(/\.(mp4|webm|mov)$/i, '.png')
        const pngUrl = item.url?.replace(/\.(mp4|webm|mov)$/i, '.png')
        imageUrl = pngUrl
        fetchUrl = pngUrl
        imageFilename = pngFilename
        console.debug('🎬 I2T: video detected, using companion image:', pngFilename)
      } else {
        imageUrl = getMediaUrl(item.url, item.signed_url)
        fetchUrl = item.signed_url || (item.url?.startsWith('/') ? item.url : `/${item.url}`)
        imageFilename = item.filename || item.url?.split('/').pop() || 'image.png'
      }

      try {
        const response = await fetch(fetchUrl)
        const blob = await response.blob()
        const filename = imageFilename
        const fileObj = new File([blob], filename, { type: blob.type || 'image/png' })
        setFile(fileObj)
        setPreview(imageUrl)
        setCaption('')
        setError(null)
        if (DEBUG) console.log('🖼️ Imported image from creations:', filename)
      } catch (e) {
        console.error('Failed to load image from import:', e)
        setError('⚠️ Failed to load image from import')
      }
    }
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
      setFile(fileObj)
      setPreview(imageUrl)
      setCaption('')
      setError(null)
      if (DEBUG) console.log('\ud83d\udcc1 I2T: loaded from creations:', filename)
    } catch (e) {
      console.error('Failed to load from creations:', e)
      setError('\u26a0\ufe0f Failed to load image from My Creations')
    }
  }, [])

  const handleFileChange = useCallback((e) => {
    const f = e.target.files?.[0]
    if (f) {
      setFile(f)
      setPreview(URL.createObjectURL(f))
      setCaption('')
      setError(null)
    }
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0]
    if (f && f.type.startsWith('image/')) {
      setFile(f)
      setPreview(URL.createObjectURL(f))
      setCaption('')
      setError(null)
    }
  }, [])

  const handleCaption = async () => {
    if (!user) { requestLogin('Log in om afbeeldingen te analyseren'); return }
    if (!file) return

    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('model', model)
      formData.append('mode', mode)
      if (mode === 'prompt_nsfw') {
        formData.append('nsfw_intensity', nsfwIntensity.toString())
      }
      if (includeMotion) {
        formData.append('include_motion', 'true')
      }

      const res = await fetch(`${BACKEND_BASE}/caption-image`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Caption failed')
      }

      const data = await res.json()
      let captionText = data.caption || ''

      // Prepend camera motion prefix when in prompt mode and motion is selected
      if (isPromptMode(mode) && cameraMotion) {
        const motionPrefix = getCameraMotionPrefix(cameraMotion)
        if (motionPrefix) captionText = motionPrefix + captionText
      }
      setCaption(captionText)

      if (DEBUG) console.log('🖼️ Caption result:', data)
    } catch (err) {
      console.error('Caption error:', err)
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // LLM prompt enhancement queue
  const llm = useLLMEnhance()

  // Refine/improve caption with LLM — preserves original intent (via async queue)
  const handleRefineCaption = async () => {
    if (!caption.trim() || isRefining) return
    setIsRefining(true)
    setError(null)

    const result = await llm.enhance({
      input: caption.trim(),
      mode: 'refine',
      model: model,
      include_negative: false,
      include_motion: isPromptMode(mode),
      refine_instruction: refineInstruction.trim() || null,
    })

    if (result) {
      setCaption(result.prompt)
      setRefineInstruction('')
      setShowRefineInput(false)
    } else if (llm.error) {
      setError(`Refine failed: ${llm.error}`)
    }
    setIsRefining(false)
  }

  const handleCopy = () => {
    if (caption) {
      navigator.clipboard.writeText(caption)
    }
  }

  const handleSendToPrompt = () => {
    if (caption && onSendToPrompt) {
      onSendToPrompt(caption)
    }
  }

  return (
    <div className="tool-container">

      {/* Import from previous generation modal */}
      {importModal && (
        <MediaImportModal
          item={importModal.item}
          parsedData={importModal.workflow}
          availableFields={['image']}
          onApply={handleApplyImport}
          onClose={() => setImportModal(null)}
        />
      )}

      <div className="tool-section">
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <ImageIcon size={18} />
          Upload Image
          <ResetDefaultsButton onReset={handleResetDefaults} />
        </h3>

        <div
          className={`upload-dropzone ${preview ? 'has-preview' : ''}`}
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => document.getElementById('i2t-file-input').click()}
        >
          {preview ? (
            <img src={preview} alt="Preview" className="upload-preview" />
          ) : (
            <div className="upload-placeholder">
              <Upload size={32} />
              <p>Drop image here or click to upload</p>
            </div>
          )}
          <input
            id="i2t-file-input"
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
          {'📁'} From My Creations
        </button>

        <CreationsPickerModal
          show={showCreationsPicker}
          onClose={() => setShowCreationsPicker(false)}
          onSelect={handleCreationsSelect}
          filter="image"
          title="Select Image for Captioning"
        />
      </div>

      <div className="tool-section">
        <h3>
          <Wand2 size={18} />
          Caption Settings
        </h3>

        <div className="form-group">
          <label>Model</label>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {MODELS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label} - {m.description}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label>Caption Mode</label>
          <div className="button-group">
            {CAPTION_MODES.filter(m => m.group === 'caption').map((m) => (
              <button
                key={m.id}
                className={`btn-option ${mode === m.id ? 'active' : ''}`}
                onClick={() => setMode(m.id)}
                title={m.description}
              >
                {m.label}
              </button>
            ))}
          </div>
          <label style={{ marginTop: '12px' }}>Prompt Generator</label>
          <div className="button-group">
            {CAPTION_MODES.filter(m => m.group === 'prompt').map((m) => (
              <button
                key={m.id}
                className={`btn-option ${mode === m.id ? 'active' : ''} ${m.id === 'prompt_nsfw' ? 'btn-option--nsfw' : ''}`}
                onClick={() => setMode(m.id)}
                title={m.description}
              >
                {m.label}
              </button>
            ))}
          </div>

          {/* Include Motion checkbox — shown for all prompt modes */}
          {isPromptMode(mode) && (
            <div style={{ marginTop: '12px' }}>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={includeMotion}
                  onChange={(e) => setIncludeMotion(e.target.checked)}
                />
                Include motion prompts (for video)
              </label>
            </div>
          )}

          {/* Camera Motion Selector — shown for all prompt modes (I2V, T2I, NSFW) */}
          {isPromptMode(mode) && (
            <div style={{ marginTop: '12px' }}>
              <CameraMotionSelector
                value={cameraMotion}
                onChange={setCameraMotion}
              />
              <p style={{ margin: '6px 0 0', fontSize: '11px', color: 'var(--text-muted, #666)' }}>
                Selected motion is prepended to the generated prompt — ready for T2V / I2V.
              </p>
            </div>
          )}

          {/* NSFW Intensity Slider */}
          {mode === 'prompt_nsfw' && (
            <div className="nsfw-intensity-section">
              <div className="nsfw-intensity-header">
                <label>Intensity</label>
                <span className="nsfw-intensity-badge">
                  {NSFW_LEVELS.find(l => l.value === nsfwIntensity)?.emoji}{' '}
                  {NSFW_LEVELS.find(l => l.value === nsfwIntensity)?.label}
                </span>
              </div>
              <input
                type="range"
                min={1}
                max={5}
                step={1}
                value={nsfwIntensity}
                onChange={(e) => setNsfwIntensity(Number(e.target.value))}
                className="nsfw-slider"
              />
              <div className="nsfw-intensity-labels">
                {NSFW_LEVELS.map((level) => (
                  <span
                    key={level.value}
                    className={`nsfw-label ${nsfwIntensity === level.value ? 'active' : ''}`}
                    onClick={() => setNsfwIntensity(level.value)}
                  >
                    {level.emoji}
                  </span>
                ))}
              </div>
              <p className="nsfw-intensity-desc">
                {NSFW_LEVELS.find(l => l.value === nsfwIntensity)?.description}
              </p>
            </div>
          )}
        </div>
      </div>

      <button
        className="btn-primary btn-large"
        onClick={handleCaption}
        disabled={!file || loading}
      >
        {loading ? (
          <>
            <Loader2 size={18} className="spin" />
            {isPromptMode(mode) ? 'Generating prompt...' : 'Generating caption...'}
          </>
        ) : (
          <>
            <Wand2 size={18} />
            {isPromptMode(mode) ? 'Generate Prompt' : 'Generate Caption'}
          </>
        )}
      </button>

      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}

      {caption && (
        <div className="tool-section result-section">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <h3>{isPromptMode(mode) ? '🎯 Generated Prompt' : 'Generated Caption'}</h3>
            <button
              className="icon-btn"
              style={{
                width: '28px', height: '28px', padding: '6px',
                background: showRefineInput ? 'var(--accent-color, #8b5cf6)' : undefined,
                color: showRefineInput ? 'white' : undefined,
                borderRadius: '6px',
                border: '1px solid var(--border-color, #444)',
                cursor: 'pointer',
              }}
              onClick={() => setShowRefineInput(!showRefineInput)}
              disabled={!caption.trim()}
              title="Refine/improve with AI (keeps original intent)"
            >
              <Pencil size={14} />
            </button>
          </div>

          {/* Refine instruction input */}
          {showRefineInput && (
            <div style={{
              marginTop: '8px',
              padding: '8px 12px',
              backgroundColor: 'rgba(139, 92, 246, 0.08)',
              border: '1px solid rgba(139, 92, 246, 0.25)',
              borderRadius: '8px',
              display: 'flex',
              gap: '8px',
              alignItems: 'center',
            }}>
              <Pencil size={14} style={{ color: '#a78bfa', flexShrink: 0 }} />
              <input
                type="text"
                value={refineInstruction}
                onChange={(e) => setRefineInstruction(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter' && caption.trim()) handleRefineCaption() }}
                placeholder="What to improve? (e.g., more detail, different style...) — leave empty for general polish"
                style={{
                  flex: 1,
                  background: 'var(--bg-input, #1a1a1a)',
                  border: '1px solid var(--border-color, #444)',
                  borderRadius: '6px',
                  padding: '6px 10px',
                  fontSize: '0.8rem',
                  color: 'var(--text-primary, #eee)',
                  outline: 'none',
                }}
              />
              <button
                className="icon-btn"
                style={{
                  height: '28px',
                  padding: '4px 12px',
                  fontSize: '0.75rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  background: isRefining ? 'var(--bg-input)' : 'linear-gradient(135deg, #8b5cf6, #6d28d9)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  whiteSpace: 'nowrap',
                  cursor: 'pointer',
                }}
                onClick={handleRefineCaption}
                disabled={isRefining || !caption.trim()}
                title="Refine with AI"
              >
                {isRefining ? <Loader2 size={12} className="spin" /> : <Pencil size={12} />}
                <span>{isRefining ? 'Refining...' : 'Refine'}</span>
                <LLMQueueIndicator queuePosition={llm.queuePosition} isLoading={llm.isLoading} />
              </button>
            </div>
          )}

          <div className="caption-result">
            <textarea
              value={caption}
              onChange={(e) => setCaption(e.target.value)}
              rows={isPromptMode(mode) ? 6 : 4}
            />
            <div className="caption-actions">
              <button className="btn-secondary" onClick={handleCopy}>
                <Copy size={16} />
                Copy
              </button>
              {onSendToPrompt && (
                <button className={isPromptMode(mode) ? 'btn-primary btn-glow' : 'btn-primary'} onClick={handleSendToPrompt}>
                  <Send size={16} />
                  Use as Prompt
                </button>
              )}
            </div>
            {isPromptMode(mode) && (
              <p className="prompt-hint">
                💡 Edit the prompt above, then send it directly to Image-to-Video or Text-to-Image
              </p>
            )}
          </div>
        </div>
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
        }
        .upload-preview {
          max-width: 100%;
          max-height: 300px;
          border-radius: 8px;
          object-fit: contain;
        }
        .upload-placeholder {
          color: var(--text-muted, #888);
        }
        .upload-placeholder p {
          margin-top: 12px;
        }
        .button-group {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
        }
        .btn-option {
          padding: 8px 16px;
          border: 1px solid var(--border-color, #444);
          border-radius: 8px;
          background: transparent;
          color: var(--text-color, #fff);
          cursor: pointer;
          transition: all 0.2s;
        }
        .btn-option:hover {
          border-color: var(--accent-color, #7c3aed);
        }
        .btn-option.active {
          background: var(--accent-color, #7c3aed);
          border-color: var(--accent-color, #7c3aed);
        }
        .caption-result textarea {
          width: 100%;
          padding: 12px;
          border-radius: 8px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          font-family: inherit;
          resize: vertical;
        }
        .caption-actions {
          display: flex;
          gap: 8px;
          margin-top: 12px;
        }
        .error-message {
          padding: 12px;
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 8px;
          color: #ef4444;
          margin-top: 12px;
        }
        .btn-option--nsfw.active {
          background: #dc2626;
          border-color: #dc2626;
        }
        .nsfw-intensity-section {
          margin-top: 12px;
          padding: 12px 16px;
          background: rgba(220, 38, 38, 0.06);
          border: 1px solid rgba(220, 38, 38, 0.2);
          border-radius: 10px;
        }
        .nsfw-intensity-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 8px;
        }
        .nsfw-intensity-header label {
          font-size: 0.85rem;
          font-weight: 600;
          color: var(--text-primary, #eee);
        }
        .nsfw-intensity-badge {
          font-size: 0.8rem;
          padding: 2px 10px;
          border-radius: 12px;
          background: rgba(220, 38, 38, 0.15);
          color: #f87171;
          font-weight: 600;
        }
        .nsfw-slider {
          width: 100%;
          height: 6px;
          -webkit-appearance: none;
          appearance: none;
          background: linear-gradient(90deg, #f97316 0%, #dc2626 50%, #7f1d1d 100%);
          border-radius: 3px;
          outline: none;
          cursor: pointer;
        }
        .nsfw-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: #ef4444;
          border: 2px solid #fff;
          box-shadow: 0 0 8px rgba(239, 68, 68, 0.5);
          cursor: pointer;
        }
        .nsfw-slider::-moz-range-thumb {
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: #ef4444;
          border: 2px solid #fff;
          box-shadow: 0 0 8px rgba(239, 68, 68, 0.5);
          cursor: pointer;
        }
        .nsfw-intensity-labels {
          display: flex;
          justify-content: space-between;
          margin-top: 6px;
          padding: 0 2px;
        }
        .nsfw-label {
          cursor: pointer;
          font-size: 1rem;
          opacity: 0.4;
          transition: all 0.2s;
          user-select: none;
        }
        .nsfw-label:hover {
          opacity: 0.8;
          transform: scale(1.2);
        }
        .nsfw-label.active {
          opacity: 1;
          transform: scale(1.3);
        }
        .nsfw-intensity-desc {
          margin-top: 6px;
          font-size: 0.78rem;
          color: var(--text-muted, #888);
          text-align: center;
          font-style: italic;
        }
        .btn-glow {
          box-shadow: 0 0 12px rgba(124, 58, 237, 0.4);
        }
        .prompt-hint {
          margin-top: 8px;
          font-size: 0.85em;
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
