import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Check,
  CheckCheck,
  Copy,
  Cpu,
  Layers,
  Loader2,
  Plus,
  RefreshCw,
  Sparkles,
  Trash2,
  Zap,
} from 'lucide-react'
import InfoTooltip from '../../components/InfoTooltip'
import { apiFetch } from '../../api'
import { useAuth } from '../../contexts/AuthContext'

const MIN_IMAGES = 5
const SAFE_PREVIEW_IMAGE_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp', 'image/gif'])

function isSafePreviewImage(file) {
  return file && SAFE_PREVIEW_IMAGE_TYPES.has(file.type)
}

async function createRasterPreview(file) {
  if (!isSafePreviewImage(file)) {
    throw new Error('Unsupported image preview type')
  }

  const bitmap = await createImageBitmap(file)
  const canvas = document.createElement('canvas')
  canvas.width = bitmap.width
  canvas.height = bitmap.height
  const context = canvas.getContext('2d')
  context.drawImage(bitmap, 0, 0)
  bitmap.close?.()
  return canvas.toDataURL('image/png')
}

const STATUS_COLORS = {
  pending: '#f59e0b',
  running: '#60a5fa',
  done: '#22c55e',
  failed: '#f87171',
  cancelled: '#94a3b8',
}

function formatTimestamp(timestamp) {
  if (!timestamp) return 'Just now'
  try {
    return new Date(timestamp * 1000).toLocaleString()
  } catch {
    return 'Unknown'
  }
}

export default function LoRATrainingTool({ onOutput }) {
  const { user, requestLogin } = useAuth()
  const fileInputRef = useRef(null)

  const [images, setImages] = useState([])
  const [previews, setPreviews] = useState([])
  const [name, setName] = useState('')
  const [steps, setSteps] = useState(1000)
  const [jobs, setJobs] = useState([])
  const [loras, setLoras] = useState([])
  const [submitting, setSubmitting] = useState(false)
  const [refreshingBrowser, setRefreshingBrowser] = useState(false)
  const [error, setError] = useState(null)
  const [copiedTrigger, setCopiedTrigger] = useState(null)
  const [statusMessage, setStatusMessage] = useState('')

  const triggerPreview = useMemo(() => {
    if (!name.trim()) return null
    return `ohwx_${name.trim().toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '')}`
  }, [name])

  const remainingImages = Math.max(0, MIN_IMAGES - images.length)
  const canSubmit = name.trim() && images.length >= MIN_IMAGES && !submitting

  const loadStatus = useCallback(async () => {
    try {
      const [jobsRes, lorasRes] = await Promise.all([
        apiFetch('/api/face-train'),
        apiFetch('/api/face-train/loras'),
      ])
      const [jobsData, lorasData] = await Promise.all([jobsRes.json(), lorasRes.json()])
      setJobs((jobsData.jobs || []).slice().reverse())
      setLoras((lorasData.loras || []).slice().sort((a, b) => (b.modified || 0) - (a.modified || 0)))
    } catch (err) {
      setError(err?.message || 'Failed to load LoRA training status')
    }
  }, [])

  useEffect(() => {
    loadStatus()
    const interval = setInterval(loadStatus, 5000)
    return () => clearInterval(interval)
  }, [loadStatus])

  const handleImagePick = async (pickedFiles) => {
    const files = Array.from(pickedFiles || []).filter(isSafePreviewImage)
    if (files.length === 0) {
      setError('Upload PNG, JPG, WebP, or GIF images only')
      return
    }

    try {
      const safePreviews = await Promise.all(files.map(createRasterPreview))
      setImages((prev) => [...prev, ...files])
      setPreviews((prev) => [...prev, ...safePreviews])
      setError(null)
      setStatusMessage('')
    } catch (err) {
      setError(err?.message || 'Failed to prepare image previews')
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const removeImage = (index) => {
    setImages((prev) => prev.filter((_, idx) => idx !== index))
    setPreviews((prev) => prev.filter((_, idx) => idx !== index))
  }

  const clearImages = () => {
    setImages([])
    setPreviews([])
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const handleSubmit = async () => {
    if (!user) {
      requestLogin('Log in om een person LoRA te trainen')
      return
    }

    if (!name.trim()) {
      setError('Person name is required')
      return
    }

    if (images.length < MIN_IMAGES) {
      setError(`Upload at least ${MIN_IMAGES} reference photos`)
      return
    }

    setSubmitting(true)
    setError(null)
    setStatusMessage('')

    try {
      const formData = new FormData()
      formData.append('name', name.trim())
      formData.append('steps', String(steps))
      images.forEach((image) => formData.append('images', image))

      const response = await apiFetch('/api/face-train', { method: 'POST', body: formData })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to start training')
      }

      setStatusMessage(`Training started for ${data.trigger} using official SDXL base`)
      setName('')
      clearImages()
      await loadStatus()
      onOutput?.({ kind: 'face-lora-training', ...data })
    } catch (err) {
      setError(err?.message || 'Failed to start training')
    } finally {
      setSubmitting(false)
    }
  }

  const handleCancel = async (jobId) => {
    try {
      await apiFetch(`/api/face-train/${jobId}`, { method: 'DELETE' })
      await loadStatus()
    } catch (err) {
      setError(err?.message || 'Failed to cancel training job')
    }
  }

  const handleRefreshBrowserIndex = async () => {
    setRefreshingBrowser(true)
    setStatusMessage('')
    try {
      const response = await apiFetch('/api/loras/refresh', { method: 'POST' })
      const data = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to refresh LoRA browser index')
      }
      setStatusMessage('LoRA Browser index refreshed. New face LoRAs should now appear under the face_loras category.')
    } catch (err) {
      setError(err?.message || 'Failed to refresh LoRA browser index')
    } finally {
      setRefreshingBrowser(false)
    }
  }

  const copyTrigger = async (trigger) => {
    try {
      await navigator.clipboard.writeText(trigger)
      setCopiedTrigger(trigger)
      setTimeout(() => setCopiedTrigger(null), 2000)
    } catch {
      setError('Failed to copy trigger word')
    }
  }

  return (
    <div className="tool-container" style={{ gap: '12px' }}>
      <div className="grok-card" style={{ borderColor: 'rgba(168, 85, 247, 0.3)' }}>
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Cpu size={16} style={{ color: '#c084fc' }} />
            Person LoRA Studio
          </div>
          <span className="nav-badge">SDXL Base</span>
        </div>

        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '12px' }}>
          This is the dedicated home for person LoRA training in the UI. It trains against the official SDXL base model,
          stores finished LoRAs in <code>face_loras</code>, and keeps jobs plus ready-to-use triggers in one place.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '8px', marginBottom: '12px' }}>
          <div className="grok-card" style={{ padding: '10px 12px', background: 'rgba(15, 23, 42, 0.45)' }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Base model</div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-primary)', fontWeight: 600 }}>stabilityai/stable-diffusion-xl-base-1.0</div>
          </div>
          <div className="grok-card" style={{ padding: '10px 12px', background: 'rgba(15, 23, 42, 0.45)' }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Storage</div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-primary)', fontWeight: 600 }}>ComfyUI/models/loras/face_loras</div>
          </div>
          <div className="grok-card" style={{ padding: '10px 12px', background: 'rgba(15, 23, 42, 0.45)' }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Minimum dataset</div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-primary)', fontWeight: 600 }}>{MIN_IMAGES}+ images</div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '12px' }}>
          <div className="form-group">
            <label className="grok-section-label">
              Person Name * <InfoTooltip text="Used to generate the trigger word for the LoRA. This is what you will put into SDXL prompts later." />
            </label>
            <input
              className="form-input"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="e.g. Jane Doe"
            />
            {triggerPreview && (
              <p style={{ fontSize: '0.72rem', color: '#c084fc', marginTop: '6px' }}>
                Trigger preview: <code>{triggerPreview}</code>
              </p>
            )}
          </div>

          <div className="form-group">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <label className="grok-section-label">
                Training Steps <InfoTooltip text="800-1200 is a good starting range for a person LoRA. Higher steps can overfit when the dataset is small." />
              </label>
              <span className="nav-badge">{steps}</span>
            </div>
            <input
              type="range"
              className="form-range"
              min={600}
              max={2000}
              step={100}
              value={steps}
              onChange={(event) => setSteps(Number(event.target.value))}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
              <span>600</span>
              <span>2000</span>
            </div>
          </div>
        </div>

        <div className="form-group">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <label className="grok-section-label">
              Reference Photos * <InfoTooltip text="Use 5-20 clean photos with varied angle, lighting, and expression. Avoid duplicates, group photos, or heavily filtered selfies." />
            </label>
            <span className="nav-badge">{images.length} selected</span>
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '8px' }}>
            {previews.map((preview, index) => (
              <div key={`${preview}-${index}`} style={{ position: 'relative' }}>
                <img
                  src={preview}
                  alt="Training preview"
                  style={{ width: '64px', height: '64px', objectFit: 'cover', borderRadius: '8px', border: '1px solid var(--border-color)' }}
                />
                <button
                  onClick={() => removeImage(index)}
                  style={{
                    position: 'absolute',
                    top: '-4px',
                    right: '-4px',
                    width: '18px',
                    height: '18px',
                    borderRadius: '50%',
                    border: 'none',
                    background: '#dc2626',
                    color: '#fff',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    cursor: 'pointer',
                  }}
                >
                  <Trash2 size={10} />
                </button>
              </div>
            ))}

            <button
              onClick={() => fileInputRef.current?.click()}
              style={{
                width: '64px',
                height: '64px',
                border: '2px dashed var(--border-color)',
                borderRadius: '8px',
                background: 'transparent',
                color: 'var(--text-muted)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
              }}
              title="Add photos"
            >
              <Plus size={18} />
            </button>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp,image/gif"
            multiple
            onChange={(event) => handleImagePick(event.target.files)}
            style={{ display: 'none' }}
          />

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <p style={{ fontSize: '0.72rem', color: remainingImages > 0 ? '#f59e0b' : '#86efac' }}>
              {remainingImages > 0
                ? `${remainingImages} more image${remainingImages === 1 ? '' : 's'} needed before training can start`
                : 'Dataset minimum reached. More variation still helps.'}
            </p>
            {images.length > 0 && (
              <button className="upload-btn secondary" onClick={clearImages} style={{ padding: '6px 10px', fontSize: '0.78rem' }}>
                Clear photos
              </button>
            )}
          </div>
        </div>

        {error && <div className="status-banner error">{error}</div>}
        {statusMessage && <div className="status-banner success">{statusMessage}</div>}

        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <button
            className="primary-btn"
            onClick={handleSubmit}
            disabled={!canSubmit}
            style={{ height: '44px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', flex: 1, minWidth: '220px' }}
          >
            {submitting
              ? <><Loader2 size={16} className="animate-spin" /> Starting training...</>
              : <><Zap size={16} /> Train Person LoRA</>}
          </button>

          <button
            className="upload-btn secondary"
            onClick={handleRefreshBrowserIndex}
            disabled={refreshingBrowser}
            style={{ height: '44px', minWidth: '220px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}
          >
            {refreshingBrowser
              ? <><Loader2 size={16} className="animate-spin" /> Refreshing index...</>
              : <><RefreshCw size={16} /> Refresh LoRA Browser Index</>}
          </button>
        </div>
      </div>

      {jobs.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <label className="grok-section-label">
            Training Jobs <InfoTooltip text="Background training jobs for person LoRAs. Running jobs auto-refresh every 5 seconds." />
          </label>
          {jobs.map((job) => {
            const progress = job.steps_total > 0 ? Math.round((job.steps_done / job.steps_total) * 100) : 0
            return (
              <div key={job.id} className="grok-card" style={{ padding: '12px 16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', alignItems: 'flex-start' }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                      <span style={{ width: '10px', height: '10px', borderRadius: '999px', background: STATUS_COLORS[job.status] || '#94a3b8' }} />
                      <span style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-primary)' }}>{job.name}</span>
                      <span className="nav-badge">{job.status}</span>
                    </div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                      Trigger: <code>{job.trigger}</code> · {job.images_count} photos · {job.steps_total} steps · {job.base_model || 'SDXL base'}
                    </div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                      Created: {formatTimestamp(job.created_at)}
                    </div>
                    {job.status === 'running' && job.steps_done === 0 && (
                      <div style={{ fontSize: '0.72rem', color: '#c4b5fd', marginTop: '6px' }}>
                        Preparing training pipeline. First run can stay at 0% while the SDXL base model downloads and loads into cache.
                      </div>
                    )}
                    {job.status === 'running' && (
                      <div style={{ marginTop: '8px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
                          <span>Step {job.steps_done} / {job.steps_total}</span>
                          <span>{progress}%</span>
                        </div>
                        <div style={{ width: '100%', height: '6px', background: 'var(--bg-secondary)', borderRadius: '999px' }}>
                          <div style={{ width: `${progress}%`, height: '100%', background: '#a855f7', borderRadius: '999px', transition: 'width 0.4s ease' }} />
                        </div>
                      </div>
                    )}
                    {job.error && (
                      <div style={{ fontSize: '0.72rem', color: '#fca5a5', marginTop: '6px' }}>{job.error}</div>
                    )}
                  </div>

                  {(job.status === 'pending' || job.status === 'running') && (
                    <button
                      onClick={() => handleCancel(job.id)}
                      style={{ padding: '6px', border: 'none', background: 'transparent', color: 'var(--text-muted)', cursor: 'pointer' }}
                      title="Cancel training job"
                    >
                      <Trash2 size={14} />
                    </button>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <label className="grok-section-label">
            Saved Face LoRAs <InfoTooltip text="These LoRAs are already stored on disk and can be reused in SDXL text-to-image flows. They also show up in the general LoRA Browser after an index refresh." />
          </label>
          <span className="nav-badge">{loras.length} saved</span>
        </div>

        {loras.length === 0 ? (
          <div className="grok-card" style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>
            <Layers size={28} style={{ opacity: 0.35, marginBottom: '8px' }} />
            <div style={{ fontSize: '0.85rem', fontWeight: 600 }}>No saved face LoRAs yet</div>
            <div style={{ fontSize: '0.75rem', marginTop: '4px' }}>Train one above and it will be stored under the face_loras library.</div>
          </div>
        ) : (
          loras.map((lora) => (
            <div key={lora.filename} className="grok-card" style={{ padding: '12px 16px', borderColor: 'rgba(34, 197, 94, 0.25)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', minWidth: 0, flex: 1 }}>
                  <Check size={16} style={{ color: '#22c55e', flexShrink: 0 }} />
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {lora.filename}
                    </div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                      {lora.size_mb} MB · Saved {formatTimestamp(lora.modified)}
                    </div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{lora.path}</div>
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                  <span className="nav-badge" style={{ fontFamily: 'monospace' }}>{lora.trigger}</span>
                  <button
                    onClick={() => copyTrigger(lora.trigger)}
                    className="upload-btn secondary"
                    style={{ padding: '6px 10px', fontSize: '0.78rem', display: 'flex', alignItems: 'center', gap: '6px' }}
                  >
                    {copiedTrigger === lora.trigger ? <CheckCheck size={14} /> : <Copy size={14} />}
                    {copiedTrigger === lora.trigger ? 'Copied' : 'Copy trigger'}
                  </button>
                  <button
                    onClick={() => onOutput?.({ kind: 'face-lora', ...lora })}
                    className="upload-btn secondary"
                    style={{ padding: '6px 10px', fontSize: '0.78rem', display: 'flex', alignItems: 'center', gap: '6px' }}
                  >
                    <Sparkles size={14} />
                    Send to output panel
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
