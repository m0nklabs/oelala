import React, { useState, useCallback, useMemo, useEffect } from 'react'
import { Upload, ZoomIn, Loader2, Image as ImageIcon, Settings, ChevronDown } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm } from '../../api'
import { useAuth } from '../../contexts/AuthContext'
import { useToolSettings } from '../../hooks/useToolSettings'
import ResetDefaultsButton from '../../components/ResetDefaultsButton'

const UPSCALE_MODELS = [
  { value: 'RealESRGAN_x4plus.pth', label: 'RealESRGAN 4x (General)', scale: 4 },
  { value: 'RealESRGAN_x4plus_anime_6B.pth', label: 'RealESRGAN 4x (Anime)', scale: 4 },
  { value: 'RealESRGAN_x2plus.pth', label: 'RealESRGAN 2x', scale: 2 },
  { value: '4x-UltraSharp.pth', label: '4x UltraSharp', scale: 4 },
  { value: '4x_NMKD-Siax_200k.pth', label: '4x NMKD-Siax', scale: 4 },
]

const SCALE_OPTIONS = [2, 4]

const UPSCALER_DEFAULTS = { model: 'RealESRGAN_x4plus.pth', scale: 4, faceEnhance: false }

export default function UpscalerTool({ onOutput, onJobSubmitted }) {
  const { user, requestLogin } = useAuth()
  const { initial, save: saveSettings, resetDefaults } = useToolSettings('upscaler', UPSCALER_DEFAULTS)

  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [imageInfo, setImageInfo] = useState(null)
  const [model, setModel] = useState(initial.model)
  const [scale, setScale] = useState(initial.scale)
  const [faceEnhance, setFaceEnhance] = useState(initial.faceEnhance)

  // Auto-save settings
  const settingsSnapshot = useMemo(() => ({ model, scale, faceEnhance }), [model, scale, faceEnhance])
  useEffect(() => { saveSettings(settingsSnapshot) }, [settingsSnapshot, saveSettings])

  const handleResetDefaults = useCallback(() => {
    const d = resetDefaults()
    setModel(d.model); setScale(d.scale); setFaceEnhance(d.faceEnhance)
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

      // Get image dimensions
      const img = new Image()
      img.onload = () => {
        setImageInfo({ width: img.width, height: img.height })
      }
      img.src = url
    }
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0]
    if (f && f.type.startsWith('image/')) {
      setFile(f)
      const url = URL.createObjectURL(f)
      setPreview(url)
      setResult(null)
      setError(null)
      setLastQueued(null)

      const img = new Image()
      img.onload = () => {
        setImageInfo({ width: img.width, height: img.height })
      }
      img.src = url
    }
  }, [])

  const handleUpscale = async () => {
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
      formData.append('scale', String(scale))
      formData.append('face_enhance', String(faceEnhance))

      if (DEBUG) console.debug('🔍 Upscale request:', { model, scale, faceEnhance })

      const res = await postForm(`${BACKEND_BASE}/upscale`, formData)

      if (!res.ok) {
        throw new Error(res.data?.detail || 'Upscaling failed')
      }

      const promptId = res.data?.prompt_id
      if (!promptId) {
        throw new Error('No prompt_id returned')
      }

      // Show queued confirmation
      setLastQueued({
        promptId,
        model: UPSCALE_MODELS.find(m => m.value === model)?.label || model,
        scale
      })

      // Notify queue indicator
      if (onJobSubmitted) onJobSubmitted({ prompt_id: promptId })

      if (DEBUG) console.debug('📋 Upscale queued:', promptId)

      // Don't wait for completion - job will appear in queue/history when done

    } catch (err) {
      console.error('Upscale error:', err)
      setError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  const selectedModel = UPSCALE_MODELS.find(m => m.value === model)
  const outputWidth = imageInfo ? imageInfo.width * scale : 0
  const outputHeight = imageInfo ? imageInfo.height * scale : 0

  return (
    <div className="tool-container">
      <div className="tool-section">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h3 style={{ margin: 0 }}>
            <ImageIcon size={18} />
            Source Image
          </h3>
          <ResetDefaultsButton onReset={handleResetDefaults} />
        </div>

        <div
          className={`upload-dropzone ${preview ? 'has-preview' : ''}`}
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => document.getElementById('upscale-file-input').click()}
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
            id="upscale-file-input"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
        </div>

        {imageInfo && (
          <div className="image-info">
            <span>📐 {imageInfo.width} × {imageInfo.height}px</span>
            <span>→</span>
            <span className="output-size">{outputWidth} × {outputHeight}px</span>
          </div>
        )}
      </div>

      <div className="tool-section">
        <h3>
          <ZoomIn size={18} />
          Upscale Settings
        </h3>

        <div className="form-group">
          <label>Scale Factor</label>
          <div className="button-group">
            {SCALE_OPTIONS.map((s) => (
              <button
                key={s}
                className={`btn-option ${scale === s ? 'active' : ''}`}
                onClick={() => setScale(s)}
                type="button"
              >
                {s}x
              </button>
            ))}
          </div>
        </div>

        <div className="form-group">
          <label>Upscale Model</label>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {UPSCALE_MODELS.map((m) => (
              <option key={m.value} value={m.value}>{m.label}</option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={faceEnhance}
              onChange={(e) => setFaceEnhance(e.target.checked)}
            />
            Face Enhancement (GFPGAN)
            <span className="hint">Improves face details</span>
          </label>
        </div>
      </div>

      {/* Queued notification */}
      {lastQueued && (
        <div className="queued-notice">
          ✅ Job queued! Check the Queue panel for progress.
          <span className="queued-mode">{lastQueued.scale}x {lastQueued.model}</span>
        </div>
      )}

      {error && <div className="error-message">⚠️ {error}</div>}

      <button
        className="btn-primary btn-large"
        onClick={handleUpscale}
        disabled={!file || submitting}
      >
        {submitting ? (
          <>
            <Loader2 size={18} className="spin" />
            Queueing...
          </>
        ) : (
          <>
            <ZoomIn size={18} />
            Upscale Image
          </>
        )}
      </button>

      {/* Result */}
      {result && (
        <div className="result-section">
          <h3>Result ({scale}x Upscaled)</h3>
          <div className="result-image">
            <img src={result} alt="Upscaled" />
          </div>
          <a
            href={result}
            download
            className="btn-secondary"
            style={{ marginTop: 12, display: 'inline-flex', alignItems: 'center', gap: 8 }}
          >
            Download Full Resolution
          </a>
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
        .image-info {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 12px;
          margin-top: 12px;
          font-size: 13px;
          color: var(--text-muted, #888);
        }
        .output-size {
          color: var(--accent-color, #7c3aed);
          font-weight: 500;
        }
        .form-group {
          margin-bottom: 16px;
        }
        .form-group label {
          display: block;
          margin-bottom: 8px;
          font-size: 13px;
          color: var(--text-secondary, #aaa);
        }
        .form-group select {
          width: 100%;
          padding: 10px 12px;
          border-radius: 8px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          font-size: 13px;
        }
        .button-group {
          display: flex;
          gap: 8px;
        }
        .btn-option {
          padding: 12px 24px;
          border: 1px solid var(--border-color, #444);
          border-radius: 8px;
          background: transparent;
          color: var(--text-color, #fff);
          cursor: pointer;
          transition: all 0.2s;
          font-size: 15px;
          font-weight: 500;
        }
        .btn-option:hover {
          border-color: var(--accent-color, #7c3aed);
        }
        .btn-option.active {
          background: var(--accent-color, #7c3aed);
          border-color: var(--accent-color, #7c3aed);
        }
        .checkbox-label {
          display: flex !important;
          align-items: center;
          gap: 8px;
          cursor: pointer;
        }
        .checkbox-label input {
          width: 16px;
          height: 16px;
        }
        .checkbox-label .hint {
          margin-left: auto;
          font-size: 12px;
          color: var(--text-muted, #666);
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
        .result-image img {
          width: 100%;
          max-height: 400px;
          object-fit: contain;
          border-radius: 8px;
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
