import React, { useState, useCallback, useEffect } from 'react'
import { Upload, Wand2, Copy, Send, Loader2, Image as ImageIcon } from 'lucide-react'
import { BACKEND_BASE, DEBUG, getMediaUrl } from '../../config'
import MediaImportModal from '../../components/MediaImportModal'

const CAPTION_MODES = [
  { id: 'brief', label: 'Brief', description: '1-line summary' },
  { id: 'detailed', label: 'Detailed', description: 'Full paragraph' },
  { id: 'tags', label: 'Tags', description: 'Comma-separated keywords' },
  { id: 'structured', label: 'Structured', description: 'Subject, style, mood' },
]

const MODELS = [
  { id: 'Qwen3-VL-32B-Gemini-Heretic-Uncensored-Thinking', label: 'Qwen3-VL 32B Heretic', description: 'Best quality · uncensored · slow' },
  { id: 'Gemma3-27B-it-vl-GLM-4.7-Uncensored-Heretic', label: 'Gemma3 27B VL Heretic', description: 'Vision + reasoning · uncensored' },
  { id: 'Qwen3-VL-30B-A3B-Thinking', label: 'Qwen3-VL 30B MoE', description: 'MoE · thinking mode · fast' },
  { id: 'Step3-VL-10B', label: 'Step3-VL 10B', description: 'Fast · good quality' },
  { id: 'moondream', label: 'Moondream', description: 'Ultra-light · fastest' },
]

export default function ImageToTextTool({ onSendToPrompt, pendingImport = null, onImportConsumed = null }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [model, setModel] = useState('Qwen3-VL-32B-Gemini-Heretic-Uncensored-Thinking')
  const [mode, setMode] = useState('detailed')
  const [caption, setCaption] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [importModal, setImportModal] = useState(null)  // { item, workflow }

  // Auto-open import modal when Dashboard sends a pendingImport
  useEffect(() => {
    if (!pendingImport) return
    setImportModal(pendingImport)
    if (onImportConsumed) onImportConsumed()
  }, [pendingImport])

  const handleApplyImport = async (selected) => {
    if (selected.image && importModal?.item) {
      const item = importModal.item
      const imageUrl = getMediaUrl(item.url, item.signed_url)
      // Use relative URL for fetch to go through Vite proxy (avoids CORS).
      // Signed URLs (http://...) are used as-is; relative paths (/comfyui/...) go through proxy.
      const fetchUrl = item.signed_url || (item.url?.startsWith('/') ? item.url : `/${item.url}`)
      try {
        const response = await fetch(fetchUrl)
        const blob = await response.blob()
        const filename = item.filename || item.url?.split('/').pop() || 'image.png'
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
    if (!file) return

    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('model', model)
      formData.append('mode', mode)

      const res = await fetch(`${BACKEND_BASE}/caption-image`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Caption failed')
      }

      const data = await res.json()
      setCaption(data.caption || '')

      if (DEBUG) console.log('🖼️ Caption result:', data)
    } catch (err) {
      console.error('Caption error:', err)
      setError(err.message)
    } finally {
      setLoading(false)
    }
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
        <h3>
          <ImageIcon size={18} />
          Upload Image
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
            {CAPTION_MODES.map((m) => (
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
            Generating caption...
          </>
        ) : (
          <>
            <Wand2 size={18} />
            Generate Caption
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
          <h3>Generated Caption</h3>
          <div className="caption-result">
            <textarea
              value={caption}
              onChange={(e) => setCaption(e.target.value)}
              rows={4}
            />
            <div className="caption-actions">
              <button className="btn-secondary" onClick={handleCopy}>
                <Copy size={16} />
                Copy
              </button>
              {onSendToPrompt && (
                <button className="btn-primary" onClick={handleSendToPrompt}>
                  <Send size={16} />
                  Use as Prompt
                </button>
              )}
            </div>
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
