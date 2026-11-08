import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Upload, X, Video as VideoIcon, Image as ImageIcon,
  Loader2, Trash2, Check, Film, Wand2,
} from 'lucide-react'
import InfoTooltip from '../../components/InfoTooltip'
import { DEBUG } from '../../config'
import { apiFetch } from '../../api'
import useGeneration from '../../hooks/useGeneration'
import { sendClientLog } from '../../logging'
import { useAuth } from '../../contexts/AuthContext'
import { useNSFW } from '../../contexts/NSFWContext'
import MyMediaTool from './MyMediaTool'

// ── MiniMax-H3 R2V (Reference-to-Video / video2video character swap) ─────────
//
// The user supplies a source video (the choreography/camera that is preserved
// frame-for-frame) plus up to 4 reference images (the identities to swap in)
// and a prompt describing the swap in H3 style (<Subject>/<Picture>/<Video>
// references). Both adapters (local Windows-PC ComfyUI + RunPod cloud) use
// operation="generate" with the video as raw base64 data URL — no upload path.

const SUPPORTED_VIDEO_FORMATS = ['video/mp4', 'video/webm', 'video/quicktime']
const MAX_REFERENCE_IMAGES = 4

// R2V constraints (17k+5 frame grid, fixed 24 fps — same H3 contract as I2V)
const MIN_FRAMES = 5
const MAX_FRAMES = 362
const FRAME_STEP = 17
const DEFAULT_FRAMES = 124
const FPS = 24

const MODEL_MODES = [
  { value: 'local', label: '🪟 Windows PC', desc: 'R2V op je eigen Windows-PC ComfyUI (lokaal, geen credits)' },
  { value: 'cloud', label: '☁️ RunPod', desc: 'R2V via de cloud worker (80GB+ GPU, per gen credits)' },
]

const ADAPTER_BY_MODE = {
  local: 'minimax-h3-local-r2v',
  cloud: 'minimax-h3-cloud-r2v',
}

const COMPUTE_TARGET_BY_MODE = {
  local: 'local',
  cloud: 'cloud',
}

// Sampling: draft = 4-step turbo, full = 20 steps
const QUALITY_MODES = [
  { value: 'draft', label: 'Draft', tag: '4 stappen', note: 'Snelst en goedkoopst — turbo distill' },
  { value: 'full', label: 'Full', tag: '20 stappen', note: 'Maximale kwaliteit — duurst' }
]

// Megapixels presets within the R2V 0.2–2.0 range. Low MP + upscale via the
// latent upscaler flow is the cheap-iteration pattern for swaps.
const MP_PRESETS = [
  { value: '0.2', label: '0.2 MP', tag: 'Snel preview', note: 'Goedkoop itereren; combinabel met latent upscaling' },
  { value: '0.4', label: '0.4 MP', tag: 'Snel', note: 'Template default' },
  { value: '0.98', label: '0.98 MP', tag: 'Native 768p', note: 'Aanbevolen — full quality' },
  { value: '2.0', label: '2.0 MP', tag: '2K', note: 'Maximaal, langzaamst' },
]

const ASPECT_RATIOS = ['16:9', '9:16', '1:1', '4:3', '3:4']

// Same math as ComfyUI's ResolutionSelector node (multiple = 32)
const h3CanvasDims = (mp, aspect) => {
  const [a, b] = aspect.split(':').map(Number)
  const scale = Math.sqrt((parseFloat(mp) * 1024 * 1024) / (a * b))
  const w = Math.round((a * scale) / 32) * 32
  const h = Math.round((b * scale) / 32) * 32
  return `${w}×${h}`
}

const readFileAsDataUrl = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = (e) => resolve(e.target.result)
    reader.onerror = reject
    reader.readAsDataURL(file)
  })

export default function VideoToVideoTool({
  onOutput: _onOutput,
  onJobSubmitted,
  pendingImport = null,
  onImportConsumed = null,
}) {
  const { user, requestLogin } = useAuth()
  const { nsfwEnabled } = useNSFW()

  // Source video (the choreography/camera structure to preserve)
  const [videoFile, setVideoFile] = useState(null)
  const [videoPreviewUrl, setVideoPreviewUrl] = useState('')

  // Reference images (identities to swap in) — up to 4
  const [refImages, setRefImages] = useState([]) // [{ file, previewUrl }]
  const refInputRef = useRef(null)

  // Generation settings
  const [prompt, setPrompt] = useState('')
  const [modelMode, setModelMode] = useState('cloud')
  const [frames, setFrames] = useState(DEFAULT_FRAMES)
  const [qualityMode, setQualityMode] = useState('draft')
  const [megapixels, setMegapixels] = useState('0.4')
  const [aspectRatio, setAspectRatio] = useState('16:9')
  const [seed, setSeed] = useState(-1)

  // LoRA stack — MiniMax-H3 category only (single-stage LoRAs)
  const [availableLoras, setAvailableLoras] = useState({})
  const [loraConfigs, setLoraConfigs] = useState([]) // [{ name, strength }]
  const [showLoraPanel, setShowLoraPanel] = useState(false)

  // Media picker (My Creations / gallery)
  const [showMediaPicker, setShowMediaPicker] = useState(false)

  // Submit state
  const [error, setError] = useState('')
  const [queued, setQueued] = useState(null)
  const { generate, loading } = useGeneration()

  // Reading a source video to a data URL is asynchronous and can take a while
  // for a large file. While that runs `loading` is still false, so the button
  // stayed live and a double click submitted the same input twice — the second
  // request could even start after the first one finished. The lock covers the
  // whole submit, not just the network call.
  const [submitting, setSubmitting] = useState(false)
  const submittingRef = useRef(false)
  const busy = loading || submitting
  const canSubmit = !!videoFile && !busy

  // Fetch available LoRAs on mount
  useEffect(() => {
    const fetchLoras = async () => {
      try {
        const res = await apiFetch('/loras')
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

  // R2V only sees MiniMax-H3 LoRAs (single-stage, same category rule as the
  // isH3 branch in ImageToVideoTool's filteredLoras)
  const filteredLoras = useMemo(() => {
    const filterList = (list) => {
      let items = list || []
      if (!nsfwEnabled) items = items.filter((l) => !l.nsfw)
      return items
    }
    const byCategory = {}
    if (availableLoras.by_category) {
      Object.keys(availableLoras.by_category).forEach((cat) => {
        if (cat !== 'minimax-h3') return
        const filtered = filterList(availableLoras.by_category[cat])
        if (filtered.length > 0) {
          byCategory[cat] = filtered
        }
      })
    }
    return { by_category: byCategory }
  }, [availableLoras, nsfwEnabled])

  // ── Video input ────────────────────────────────────────────────────────
  const handleVideoSelect = useCallback((file) => {
    if (file && SUPPORTED_VIDEO_FORMATS.some((fmt) => file.type.includes(fmt.split('/')[1]))) {
      setVideoFile(file)
      setVideoPreviewUrl(URL.createObjectURL(file))
      setQueued(null)
      setError('')
    } else if (file) {
      setError('Please upload a valid video file (MP4, WebM)')
    }
  }, [])

  const clearVideo = () => {
    setVideoFile(null)
    setVideoPreviewUrl('')
  }

  // Object URLs hold the whole file in memory until they are revoked. A source
  // video is large, so revoke the previous preview whenever it is replaced and
  // on unmount — swapping tools otherwise keeps every dropped file alive.
  useEffect(() => {
    if (!videoPreviewUrl) return undefined
    return () => URL.revokeObjectURL(videoPreviewUrl)
  }, [videoPreviewUrl])

  const refPreviewUrls = useRef(new Set())
  useEffect(() => {
    const active = new Set(refImages.map((ref) => ref.previewUrl))
    for (const url of refPreviewUrls.current) {
      if (!active.has(url)) URL.revokeObjectURL(url)
    }
    refPreviewUrls.current = active
  }, [refImages])

  useEffect(() => () => {
    // Read .current at cleanup time: the effect above replaces the Set on every
    // change, so a captured reference would still point at the initial empty one
    // and revoke nothing.
    for (const url of refPreviewUrls.current) URL.revokeObjectURL(url)
  }, [])

  // ── Pick a source video from My Creations / the gallery ────────────────
  //
  // The R2V adapters take the video as a raw base64 data URL, so a picked
  // item has to become the same File the upload path produces: fetch it once
  // and hand it to handleVideoSelect. Everything downstream stays unchanged.
  // Fetching a creation takes time, so picking A and then B can finish in the
  // wrong order. Every pick claims a ticket; only the newest one may write to
  // state, and nothing writes after the tool unmounts.
  const pickTicketRef = useRef(0)
  const mountedRef = useRef(true)
  useEffect(() => () => {
    mountedRef.current = false
  }, [])

  const handleMediaSelect = useCallback(async (item) => {
    if (!item) return
    setShowMediaPicker(false)
    const ticket = pickTicketRef.current + 1
    pickTicketRef.current = ticket
    const rawUrl = item.signed_url || item.url || ''
    if (!rawUrl) {
      setError('That creation has no usable file URL')
      return
    }
    const fetchUrl = /^https?:/i.test(rawUrl) ? rawUrl : `/${rawUrl.replace(/^\//, '')}`
    try {
      const res = await apiFetch(fetchUrl)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const blob = await res.blob()
      if (ticket !== pickTicketRef.current || !mountedRef.current) return
      const filename = item.filename || rawUrl.split('/').pop()?.split('?')[0] || 'source.mp4'
      // The stored object may come back as application/octet-stream; derive the
      // type from the extension so the format check in handleVideoSelect passes.
      const type = blob.type?.startsWith('video/')
        ? blob.type
        : /\.webm$/i.test(filename)
          ? 'video/webm'
          : /\.mov$/i.test(filename)
            ? 'video/quicktime'
            : 'video/mp4'
      handleVideoSelect(new File([blob], filename, { type }))
    } catch (err) {
      if (ticket !== pickTicketRef.current || !mountedRef.current) return
      console.warn('⚠️ R2V: could not load the picked creation', err)
      setError(`Could not load that video from your creations: ${err.message}`)
    }
  }, [handleVideoSelect])

  // Dashboard hand-off: MyMedia / Gallery "Use in tool" sends { item, workflow }.
  useEffect(() => {
    if (!pendingImport) return
    const item = pendingImport.item || pendingImport
    if (item) handleMediaSelect(item)
    if (onImportConsumed) onImportConsumed()
  }, [pendingImport, handleMediaSelect, onImportConsumed])

  // ── Reference images ───────────────────────────────────────────────────
  const handleRefImages = useCallback((fileList) => {
    const files = Array.from(fileList || [])
    if (!files.length) return
    setRefImages((prev) => {
      const room = MAX_REFERENCE_IMAGES - prev.length
      if (room <= 0) {
        setError(`Maximum ${MAX_REFERENCE_IMAGES} reference images`)
        return prev
      }
      const accepted = files.slice(0, room).map((file) => ({
        file,
        previewUrl: URL.createObjectURL(file),
      }))
      if (files.length > room) {
        setError(`Maximum ${MAX_REFERENCE_IMAGES} reference images — extra files ignored`)
      } else {
        setError('')
      }
      return [...prev, ...accepted]
    })
  }, [])

  const removeRefImage = (idx) => {
    setRefImages((prev) => prev.filter((_, i) => i !== idx))
  }

  // ── Submit ─────────────────────────────────────────────────────────────
  const handleSubmit = async () => {
    if (busy || submittingRef.current) return
    if (!user) {
      requestLogin("Log in om video's te genereren")
      return
    }

    if (!videoFile) {
      setError('Source video is required')
      return
    }

    // Claim the lock only after the validations: an early return above must not
    // leave it set, or the button stays dead for the rest of the session.
    submittingRef.current = true
    setError('')
    setQueued(null)
    setSubmitting(true)

    try {
      const inputVideo = await readFileAsDataUrl(videoFile)
      const inputImages = []
      for (const ref of refImages) {
        inputImages.push(await readFileAsDataUrl(ref.file))
      }

      const reqPayload = {
        operation: 'generate',
        target_type: 'video',
        adapter_hint: ADAPTER_BY_MODE[modelMode] || ADAPTER_BY_MODE.cloud,
        input_video: inputVideo,
        input_images: inputImages,
        prompt:
          prompt.trim() ||
          'Subject 1 from the video is replaced by the person in Picture 1; camera, framing and timing unchanged',
        frames: Number(frames) || DEFAULT_FRAMES,
        fps: FPS,
        aspect_ratio: aspectRatio,
        megapixels: parseFloat(megapixels),
        seed: seed === -1 ? Math.floor(Math.random() * 1000000) : seed,
        steps: qualityMode === 'full' ? 20 : 4,
        quality_mode: qualityMode,
        compute_target: COMPUTE_TARGET_BY_MODE[modelMode] || 'cloud',
      }

      if (loraConfigs.length > 0) {
        reqPayload.loras = loraConfigs
      }

      if (DEBUG) console.debug('🐛 submit video-to-video (R2V):', {
        ...reqPayload,
        input_video: `${String(reqPayload.input_video).slice(0, 64)}...`,
        input_images: reqPayload.input_images.map((i) => `${String(i).slice(0, 48)}...`),
      })

      const result = await generate(reqPayload)

      if (result) {
        if (DEBUG) {
          const isCloud = result.compute_target === 'cloud'
          console.debug(`🐛 Job queued (${isCloud ? 'cloud' : 'local'}):`, result.prompt_id || result.runpod_job_id)
        }
        setQueued({ jobId: result.prompt_id || result.runpod_job_id, compute: result.compute_target })
        if (onJobSubmitted) {
          onJobSubmitted(result)
        }
      }
    } catch (e) {
      const message = e?.message || 'Failed to generate video'
      setError(message)
      await sendClientLog({
        level: 'error',
        message: 'Video-to-video (R2V) failed',
        timestamp: new Date().toISOString(),
        meta: { message, modelMode },
      })
    } finally {
      submittingRef.current = false
      setSubmitting(false)
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

      {/* Header hint */}
      <div className="grok-card" style={{ padding: '12px 16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
          <Film size={16} />
          <span style={{ fontWeight: 600, fontSize: '0.95rem' }}>MiniMax-H3 R2V — character swap / reference video</span>
        </div>
        <p style={{ margin: 0, fontSize: '0.8rem', color: 'var(--text-muted, #999)' }}>
          Give a source video (camera, framing and choreography are preserved
          frame-for-frame) plus reference images of the person(s) to swap in.
          Describe the swap in H3 prompt style using <code>&lt;Video 1&gt;</code>,{' '}
          <code>&lt;Picture 1&gt;</code> and <code>&lt;Subject 1&gt;</code> references.
        </p>
      </div>

      {/* Source Video */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <VideoIcon size={16} />
            Source Video (choreography to preserve)
          </div>
        </div>

        {!videoFile ? (
          <div
            className="drop-zone"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault()
              handleVideoSelect(e.dataTransfer?.files?.[0])
            }}
            onClick={() => document.getElementById('r2v-video-input')?.click()}
            style={{ textAlign: 'center', padding: '36px 16px', cursor: 'pointer' }}
          >
            <Upload size={28} style={{ margin: '0 auto 8px', opacity: 0.6 }} />
            <p style={{ margin: 0, fontSize: '0.9rem' }}>
              Drag &amp; drop a video here, or click to browse
            </p>
            <p style={{ margin: '4px 0 0', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              MP4 / WebM — this video provides camera, framing and timing
            </p>
            <input
              id="r2v-video-input"
              type="file"
              accept="video/mp4,video/webm,video/quicktime"
              style={{ display: 'none' }}
              onChange={(e) => handleVideoSelect(e.target.files?.[0])}
            />
          </div>
        ) : (
          <div style={{ position: 'relative' }}>
            <video
              src={videoPreviewUrl}
              controls
              style={{
                width: '100%',
                maxHeight: '400px',
                borderRadius: '8px',
                border: '1px solid var(--border-color)',
                background: '#000',
              }}
            />
            <button
              onClick={clearVideo}
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
                backdropFilter: 'blur(4px)',
              }}
            >
              <X size={18} />
            </button>
            <p style={{ margin: '6px 0 0', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              {videoFile.name}
            </p>
          </div>
        )}

        <button
          type="button"
          onClick={() => setShowMediaPicker(true)}
          className="btn-secondary"
          style={{ width: '100%', marginTop: '12px' }}
        >
          📁 Select from My Creations
        </button>
      </div>

      {/* Creations / gallery picker */}
      {showMediaPicker && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0,0,0,0.8)',
            zIndex: 1000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '20px',
          }}
        >
          <div
            style={{
              backgroundColor: 'var(--bg-primary)',
              borderRadius: '12px',
              width: '100%',
              maxWidth: '1200px',
              maxHeight: '80vh',
              overflow: 'auto',
              position: 'relative',
            }}
          >
            <div
              style={{
                position: 'sticky',
                top: 0,
                padding: '16px',
                backgroundColor: 'var(--bg-primary)',
                borderBottom: '1px solid var(--border-color)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <h3 style={{ margin: 0 }}>Select a source video</h3>
              <button
                type="button"
                onClick={() => setShowMediaPicker(false)}
                aria-label="Close"
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                  fontSize: '1.5rem',
                }}
              >
                ×
              </button>
            </div>
            <div style={{ padding: '16px' }}>
              <MyMediaTool filter="video" selectionMode onSelectItem={handleMediaSelect} />
            </div>
          </div>
        </div>
      )}

      {/* Reference Images */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <ImageIcon size={16} />
            Reference Images (identities to swap in)
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {refImages.length}/{MAX_REFERENCE_IMAGES}
          </span>
        </div>

        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {refImages.map((ref, idx) => (
            <div
              key={idx}
              style={{
                position: 'relative',
                width: '96px',
                height: '96px',
                borderRadius: '6px',
                overflow: 'hidden',
                border: '1px solid var(--border-color)',
              }}
            >
              <img
                src={ref.previewUrl}
                alt={`Reference ${idx + 1}`}
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              />
              <button
                onClick={() => removeRefImage(idx)}
                title="Remove reference image"
                style={{
                  position: 'absolute',
                  top: '4px',
                  right: '4px',
                  background: 'rgba(0,0,0,0.7)',
                  border: 'none',
                  color: '#ff6b6b',
                  borderRadius: '50%',
                  width: '22px',
                  height: '22px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                }}
              >
                <Trash2 size={12} />
              </button>
              <div
                style={{
                  position: 'absolute',
                  bottom: 0,
                  left: 0,
                  right: 0,
                  background: 'linear-gradient(transparent, rgba(0,0,0,0.7))',
                  padding: '10px 4px 2px',
                  fontSize: '0.65rem',
                  color: '#fff',
                  textAlign: 'center',
                }}
              >
                Picture {idx + 1}
              </div>
            </div>
          ))}

          {refImages.length < MAX_REFERENCE_IMAGES && (
            <button
              onClick={() => refInputRef.current?.click()}
              style={{
                width: '96px',
                height: '96px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '4px',
                background: 'transparent',
                border: '1px dashed var(--border-color)',
                borderRadius: '6px',
                color: 'var(--text-muted)',
                cursor: 'pointer',
                fontSize: '0.7rem',
              }}
            >
              <Upload size={18} />
              Add
            </button>
          )}
        </div>

        <input
          ref={refInputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          multiple
          style={{ display: 'none' }}
          onChange={(e) => {
            handleRefImages(e.target.files)
            e.target.value = ''
          }}
        />

        <p style={{ margin: '8px 0 0', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Up to {MAX_REFERENCE_IMAGES} images — Picture 1, Picture 2, ... in the prompt.
        </p>
      </div>

      {/* Prompt */}
      <div className="grok-card">
        <div className="form-group">
          <label className="grok-section-label">
            Swap Prompt
            <InfoTooltip text="Describe the swap in H3 prompt style: reference <Video 1> (the source video), <Picture N> (your reference images) and <Subject N>. State exactly what changes (identity, clothing, environment) and what stays fixed (camera, framing, timing, choreography)." />
          </label>
          <textarea
            className="form-textarea"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={6}
            placeholder={'Subject 1 from the video is replaced by the person in Picture 1; camera, framing and timing unchanged.\n\nExample: <Subject 1> is the original performer in <Video 1>. <Subject 2> is the adult woman shown in <Picture 1>. The target video is an edited version of <Video 1> in which only the visual identity of <Subject 1> is replaced by <Subject 2>; the camera, framing, environment and timing remain unchanged frame for frame.'}
            style={{
              width: '100%',
              boxSizing: 'border-box',
              padding: '10px',
              backgroundColor: '#0f0f0f',
              border: '1px solid var(--border-color)',
              borderRadius: '8px',
              resize: 'vertical',
              fontSize: '0.85rem',
            }}
          />
          <p style={{ margin: '6px 0 0', fontSize: '0.72rem', color: 'var(--text-muted, #666)' }}>
            💡 H3 obeys structured prompts: describe per moment what the target subject does and keep the retention rules (camera/timing) explicit.
          </p>
        </div>
      </div>

      {/* Settings */}
      <div className="grok-card">
        <div className="form-group">
          <label className="grok-section-label">Model</label>
          <div className="grok-toggle-group vertical">
            {MODEL_MODES.map((m) => (
              <button
                key={m.value}
                type="button"
                className={`grok-toggle-btn ${modelMode === m.value ? 'active' : ''}`}
                onClick={() => setModelMode(m.value)}
                title={m.desc}
              >
                {m.label}
                <span style={{ fontSize: '0.7rem', opacity: 0.7, display: 'block' }}>
                  {ADAPTER_BY_MODE[m.value]}
                </span>
              </button>
            ))}
          </div>
        </div>

        <div className="form-group">
          <label className="grok-section-label">Quality</label>
          <div className="grok-toggle-group">
            {QUALITY_MODES.map((q) => (
              <button
                key={q.value}
                className={`grok-toggle-btn ${qualityMode === q.value ? 'active' : ''}`}
                onClick={() => setQualityMode(q.value)}
                type="button"
                title={q.note}
              >
                {q.label}
                <span style={{ fontSize: '0.7rem', opacity: 0.7, display: 'block' }}>{q.tag}</span>
              </button>
            ))}
          </div>
        </div>

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
          <label className="grok-section-label">
            Kwaliteit (megapixels)
            <InfoTooltip text="Doel-canvas via de officiële H3 ResolutionSelector-formule. Lage MP (0.2–0.4) is de goedkope iteratielus voor swaps; scherpere resultaten via latent upscaling in een tweede pass." />
          </label>
          <div className="grok-toggle-group">
            {MP_PRESETS.map((q) => (
              <button
                key={q.value}
                className={`grok-toggle-btn ${megapixels === q.value ? 'active' : ''}`}
                onClick={() => setMegapixels(q.value)}
                title={q.note}
              >
                {q.label}
                <span style={{ fontSize: '0.7rem', opacity: 0.7, display: 'block' }}>
                  {h3CanvasDims(q.value, aspectRatio)}
                </span>
              </button>
            ))}
          </div>
        </div>

        <div className="form-group">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <label className="grok-section-label">
              Frames
              <InfoTooltip text="MiniMax-H3 snapped het frame-aantal naar het 17k+5 grid bij vaste 24 fps (5, 22, 39, ..., 362). 124 frames ≈ 5,2 seconden; het getrainde bereik is ~124–362." />
            </label>
            <span className="nav-badge" style={{ fontSize: '0.8rem' }}>
              {frames}f · ~{(frames / FPS).toFixed(1)}s · 24 fps vast
            </span>
          </div>
          <div style={{ position: 'relative', height: '24px', marginBottom: '8px' }}>
            <input
              type="range"
              min={MIN_FRAMES}
              max={MAX_FRAMES}
              step={FRAME_STEP}
              value={frames}
              onChange={(e) => setFrames(parseInt(e.target.value, 10))}
              style={{ width: '100%', opacity: 0, position: 'absolute', zIndex: 2, cursor: 'pointer' }}
            />
            <div
              style={{
                position: 'absolute',
                top: '10px',
                left: 0,
                right: 0,
                height: '4px',
                backgroundColor: '#333',
                borderRadius: '2px',
              }}
            >
              <div
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  height: '100%',
                  width: `${((frames - MIN_FRAMES) / (MAX_FRAMES - MIN_FRAMES)) * 100}%`,
                  backgroundColor: 'var(--accent-color, #a78bfa)',
                  borderRadius: '2px',
                }}
              />
            </div>
          </div>
          <input
            className="form-input"
            type="number"
            min={MIN_FRAMES}
            max={MAX_FRAMES}
            step={FRAME_STEP}
            value={frames}
            onChange={(e) => {
              const v = parseInt(e.target.value, 10)
              if (!Number.isNaN(v)) {
                // Snap to the 17k+5 grid and clamp to the valid range
                let f = Math.min(MAX_FRAMES, Math.max(MIN_FRAMES, v))
                while (f % FRAME_STEP !== MIN_FRAMES % FRAME_STEP) f -= 1
                setFrames(f)
              }
            }}
            style={{ width: '120px' }}
          />
        </div>

        <div className="form-group">
          <label className="grok-section-label">Seed</label>
          <div style={{ display: 'flex', gap: '4px' }}>
            <input
              className="form-input"
              type="number"
              value={seed}
              onChange={(e) => {
                const value = parseInt(e.target.value, 10)
                setSeed(Number.isNaN(value) ? -1 : value)
              }}
              style={{ flex: 1 }}
            />
            <button className="btn ghost sm" onClick={() => setSeed(-1)} style={{ whiteSpace: 'nowrap' }}>
              Random
            </button>
          </div>
        </div>

        {/* LoRA Settings — MiniMax-H3 single-stage LoRAs only */}
        <div style={{ marginTop: '4px', paddingTop: '12px', borderTop: '1px solid var(--border-color)' }}>
          <div
            onClick={() => setShowLoraPanel(!showLoraPanel)}
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              cursor: 'pointer',
              marginBottom: showLoraPanel ? '12px' : 0,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '0.9rem', fontWeight: 500 }}>🎨 LoRA Stack</span>
              {loraConfigs.length > 0 && (
                <span
                  style={{
                    fontSize: '0.7rem',
                    padding: '2px 6px',
                    backgroundColor: 'rgba(var(--accent-rgb), 0.2)',
                    borderRadius: '10px',
                    color: 'var(--accent-color)',
                  }}
                >
                  {loraConfigs.length} active
                </span>
              )}
            </div>
            <span style={{ opacity: 0.5, fontSize: '0.8rem' }}>{showLoraPanel ? '▼' : '▶'}</span>
          </div>

          {showLoraPanel && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {loraConfigs.map((config, idx) => (
                <div
                  key={idx}
                  style={{
                    padding: '10px',
                    backgroundColor: 'var(--bg-secondary)',
                    borderRadius: '6px',
                    border: '1px solid var(--border-color)',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                    <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>LoRA #{idx + 1}</span>
                    <button
                      onClick={() => setLoraConfigs(loraConfigs.filter((_, i) => i !== idx))}
                      style={{ background: 'transparent', border: 'none', color: '#ef4444', cursor: 'pointer', fontSize: '0.75rem' }}
                    >
                      ✕ Remove
                    </button>
                  </div>

                  <div style={{ marginBottom: '8px' }}>
                    <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '4px' }}>LoRA</label>
                    <select
                      value={config.name || ''}
                      onChange={(e) => {
                        const nc = [...loraConfigs]
                        nc[idx] = { ...config, name: e.target.value }
                        setLoraConfigs(nc)
                      }}
                      style={{ width: '100%', padding: '6px 10px', backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: '4px', color: 'var(--text-primary)', fontSize: '0.8rem' }}
                    >
                      <option value="">None</option>
                      {filteredLoras.by_category &&
                        Object.keys(filteredLoras.by_category).sort().map((category) => (
                          <optgroup key={category} label={category === 'root' ? 'Root' : category}>
                            {filteredLoras.by_category[category].map((lora) => (
                              <option key={lora.path} value={lora.path}>
                                {lora.name} ({lora.size_mb}MB)
                              </option>
                            ))}
                          </optgroup>
                        ))}
                    </select>
                  </div>

                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Strength</label>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{(config.strength ?? 1.0).toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="2"
                      step="0.05"
                      value={config.strength ?? 1.0}
                      onChange={(e) => {
                        const nc = [...loraConfigs]
                        nc[idx] = { ...config, strength: parseFloat(e.target.value) }
                        setLoraConfigs(nc)
                      }}
                      style={{ width: '100%' }}
                    />
                  </div>
                </div>
              ))}

              <button
                onClick={() => setLoraConfigs([...loraConfigs, { name: '', strength: 1.0 }])}
                style={{
                  padding: '8px',
                  backgroundColor: 'transparent',
                  border: '1px dashed var(--border-color)',
                  borderRadius: '6px',
                  color: 'var(--text-secondary)',
                  cursor: 'pointer',
                  fontSize: '0.85rem',
                }}
              >
                + Add LoRA
              </button>

              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                💡 MiniMax-H3 LoRAs only (minimax-h3 category) — single-stage, one selector per slot.
              </div>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div
          style={{
            padding: '12px',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.2)',
            borderRadius: '8px',
            color: '#ef4444',
            marginBottom: '16px',
            fontSize: '0.9rem',
          }}
        >
          {error}
        </div>
      )}

      {queued && (
        <div
          style={{
            padding: '12px',
            backgroundColor: 'rgba(34, 197, 94, 0.1)',
            border: '1px solid rgba(34, 197, 94, 0.2)',
            borderRadius: '8px',
            color: '#22c55e',
            marginBottom: '16px',
            fontSize: '0.85rem',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <Check size={16} />
          <span>
            Job queued ({queued.compute === 'cloud' ? '☁️ cloud' : '🪟 local'}) —{' '}
            {queued.jobId ? <code style={{ fontSize: '0.75rem' }}>{queued.jobId}</code> : 'watch the queue panel for progress'}
          </span>
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
          color: 'black',
        }}
      >
        {busy ? (
          <>
            <Loader2 size={18} className="animate-spin" />
            Queuing...
          </>
        ) : (
          <>
            <Wand2 size={18} />
            Generate Video Swap
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
