import React, { useState, useCallback, useRef } from 'react'
import { Upload, Video, Loader2, Settings, ChevronDown, Wand2 } from 'lucide-react'
import { BACKEND_BASE, DEBUG, getMediaUrl } from '../../config'
import { postForm } from '../../api'
import { useAuth } from '../../contexts/AuthContext'
import CreationsPickerModal from '../../components/CreationsPickerModal'

// Style presets for V2V
const STYLE_PRESETS = [
  { value: 'none', label: 'Custom', desc: 'Use your own prompt' },
  { value: 'anime', label: 'Anime', desc: 'Japanese animation style' },
  { value: 'cartoon', label: 'Cartoon', desc: 'Cartoon/comic style' },
  { value: 'sketch', label: 'Sketch', desc: 'Pencil sketch effect' },
  { value: 'oil-painting', label: 'Oil Painting', desc: 'Classic oil painting style' },
  { value: 'watercolor', label: 'Watercolor', desc: 'Watercolor painting effect' },
  { value: 'pixel-art', label: 'Pixel Art', desc: 'Retro pixel art style' },
  { value: 'cyberpunk', label: 'Cyberpunk', desc: 'Neon futuristic style' },
  { value: '3d-render', label: '3D Render', desc: 'Modern 3D rendered look' },
]

const STYLE_PROMPTS = {
  'anime': 'anime style, japanese animation, cel shading, vibrant colors, detailed linework',
  'cartoon': 'cartoon style, comic art, bold outlines, bright colors, disney style',
  'sketch': 'pencil sketch, hand-drawn, graphite, detailed linework, black and white',
  'oil-painting': 'oil painting style, classical art, brush strokes, rich colors, masterpiece',
  'watercolor': 'watercolor painting, soft edges, translucent colors, artistic, flowing',
  'pixel-art': 'pixel art style, 8-bit, retro gaming, blocky, nostalgic',
  'cyberpunk': 'cyberpunk style, neon lights, futuristic, rain, dark atmosphere, high tech',
  '3d-render': '3d render, modern cgi, photorealistic, octane render, unreal engine',
}

export default function VideoToVideoTool({ onOutput, onJobSubmitted }) {
  const { user, requestLogin } = useAuth()

  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [videoInfo, setVideoInfo] = useState(null)
  const [showCreationsPicker, setShowCreationsPicker] = useState(false)

  const [style, setStyle] = useState('none')
  const [prompt, setPrompt] = useState('')
  const [negativePrompt, setNegativePrompt] = useState('blurry, low quality, distorted, watermark')
  const [denoise, setDenoise] = useState(0.5)
  const [fps, setFps] = useState(8)
  const [maxFrames, setMaxFrames] = useState(32)

  const [showAdvanced, setShowAdvanced] = useState(false)
  const [steps, setSteps] = useState(20)
  const [cfg, setCfg] = useState(7.5)
  const [seed, setSeed] = useState(-1)

  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)
  const [lastQueued, setLastQueued] = useState(null)
  const [result, setResult] = useState(null)

  const videoRef = useRef(null)

  const handleFileChange = useCallback((e) => {
    const f = e.target.files?.[0]
    if (f) {
      setFile(f)
      const url = URL.createObjectURL(f)
      setPreview(url)
      setResult(null)
      setError(null)
      setLastQueued(null)

      // Get video info
      const video = document.createElement('video')
      video.onloadedmetadata = () => {
        setVideoInfo({
          duration: video.duration.toFixed(1),
          width: video.videoWidth,
          height: video.videoHeight,
        })
      }
      video.src = url
    }
  }, [])

  const handleCreationsSelect = useCallback(async (item) => {
    try {
      const mediaUrl = getMediaUrl(item.url, item.signed_url)
      const response = await fetch(mediaUrl)
      if (!response.ok) throw new Error(`Failed to fetch: ${response.status}`)
      const blob = await response.blob()
      const filename = item.filename || mediaUrl.split('/').pop()
      const fileObj = new File([blob], filename, { type: blob.type || 'video/mp4' })
      const url = URL.createObjectURL(fileObj)
      setFile(fileObj)
      setPreview(url)
      setResult(null)
      setError(null)
      setLastQueued(null)
      // Get video info
      const video = document.createElement('video')
      video.onloadedmetadata = () => {
        setVideoInfo({
          duration: video.duration.toFixed(1),
          width: video.videoWidth,
          height: video.videoHeight,
        })
      }
      video.src = url
      if (DEBUG) console.log('\ud83d\udcc1 V2V: loaded from creations:', filename)
    } catch (e) {
      console.error('Failed to load from creations:', e)
      setError('\u26a0\ufe0f Failed to load video from My Creations')
    }
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0]
    if (f && f.type.startsWith('video/')) {
      setFile(f)
      const url = URL.createObjectURL(f)
      setPreview(url)
      setResult(null)
      setError(null)
      setLastQueued(null)

      const video = document.createElement('video')
      video.onloadedmetadata = () => {
        setVideoInfo({
          duration: video.duration.toFixed(1),
          width: video.videoWidth,
          height: video.videoHeight,
        })
      }
      video.src = url
    }
  }, [])

  const handleTransform = async () => {
    // Check if user is logged in
    if (!user) {
      requestLogin('Log in om te genereren')
      return
    }

    if (!file) return

    // Determine final prompt
    const finalPrompt = style !== 'none'
      ? STYLE_PROMPTS[style] + (prompt ? ', ' + prompt : '')
      : prompt

    if (!finalPrompt.trim()) {
      setError('Please select a style or enter a prompt')
      return
    }

    setSubmitting(true)
    setError(null)
    setLastQueued(null)

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('prompt', finalPrompt)
      formData.append('negative_prompt', negativePrompt)
      formData.append('denoise', String(denoise))
      formData.append('fps', String(fps))
      formData.append('max_frames', String(maxFrames))
      formData.append('steps', String(steps))
      formData.append('cfg', String(cfg))
      formData.append('seed', String(seed))

      if (DEBUG) console.debug('🎬 V2V request:', { style, denoise, fps, maxFrames })

      const res = await postForm(`${BACKEND_BASE}/generate-v2v`, formData)

      if (!res.ok) {
        throw new Error(res.data?.detail || 'V2V transform failed')
      }

      const promptId = res.data?.prompt_id
      if (!promptId) {
        throw new Error('No prompt_id returned')
      }

      // Show queued confirmation
      setLastQueued({
        promptId,
        style: style !== 'none' ? style : 'custom'
      })

      // Notify queue indicator
      if (onJobSubmitted) onJobSubmitted({ prompt_id: promptId })

      if (DEBUG) console.debug('📋 V2V queued:', promptId)

      // Don't wait for completion - job will appear in queue/history when done

    } catch (err) {
      console.error('V2V error:', err)
      setError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="tool-container">
      <CreationsPickerModal
        show={showCreationsPicker}
        onClose={() => setShowCreationsPicker(false)}
        onSelect={handleCreationsSelect}
        filter="video"
        title="Select Video from My Creations"
      />

      {/* Source Video */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Video size={16} />
            Source Video
          </div>
        </div>

        <div
          className="upload-box"
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => document.getElementById('v2v-file-input').click()}
          style={{ cursor: 'pointer' }}
        >
          {preview ? (
            <video
              ref={videoRef}
              src={preview}
              controls
              muted
              loop
              style={{ maxHeight: '250px', maxWidth: '100%', borderRadius: '8px' }}
            />
          ) : (
            <>
              <Upload size={32} className="text-muted" />
              <div className="text-muted">Drop video here or click to upload</div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>MP4, WebM, MOV</span>
            </>
          )}
          <input
            id="v2v-file-input"
            type="file"
            accept="video/*"
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

        {videoInfo && (
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            gap: '16px', marginTop: '12px', fontSize: '0.8rem', color: 'var(--text-muted)',
          }}>
            <span>📐 {videoInfo.width} × {videoInfo.height}px</span>
            <span>⏱️ {videoInfo.duration}s</span>
          </div>
        )}
      </div>

      {/* Style Selection */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Wand2 size={16} />
            Style Transform
          </div>
        </div>

        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px',
        }}>
          {STYLE_PRESETS.map((preset) => (
            <button
              key={preset.value}
              onClick={() => setStyle(preset.value)}
              style={{
                padding: '12px', textAlign: 'left',
                border: `1px solid ${style === preset.value ? 'var(--text-primary)' : 'var(--border-color)'}`,
                borderRadius: '8px', cursor: 'pointer',
                backgroundColor: style === preset.value ? '#262626' : 'var(--bg-input)',
                transition: 'all 0.15s',
              }}
            >
              <span style={{ display: 'block', fontSize: '0.85rem', fontWeight: 500, color: 'var(--text-primary)' }}>{preset.label}</span>
              <span style={{ display: 'block', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>{preset.desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Prompt */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Prompt {style !== 'none' && <span style={{ fontWeight: 400, fontSize: '0.8rem', color: 'var(--text-muted)' }}>(optional - adds to style)</span>}</div>
        </div>
        <textarea
          className="form-textarea"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder={style !== 'none'
            ? 'Add extra details to the style...'
            : 'Describe the desired look...'}
          rows={3}
          style={{ minHeight: '70px' }}
        />
      </div>

      {/* Transform Strength */}
      <div className="grok-card">
        <div className="form-group" style={{ marginBottom: 0 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <label className="grok-section-label" style={{ marginBottom: 0 }}>Transform Strength</label>
            <span className="nav-badge" style={{ fontSize: '0.8rem' }}>{(denoise * 100).toFixed(0)}%</span>
          </div>
          <div style={{ position: 'relative', height: '24px', marginBottom: '8px' }}>
            <input
              type="range"
              min="0.1"
              max="1"
              step="0.05"
              value={denoise}
              onChange={(e) => setDenoise(parseFloat(e.target.value))}
              style={{ width: '100%', opacity: 0, position: 'absolute', zIndex: 2, cursor: 'pointer' }}
            />
            <div style={{
              position: 'absolute', top: '10px', left: 0, right: 0,
              height: '4px', backgroundColor: '#333', borderRadius: '2px'
            }}>
              <div style={{
                width: `${((denoise - 0.1) / 0.9) * 100}%`,
                height: '100%', backgroundColor: 'var(--accent-color, #a855f7)', borderRadius: '2px'
              }} />
            </div>
            <div style={{
              position: 'absolute', top: '2px',
              left: `calc(${((denoise - 0.1) / 0.9) * 100}% - 10px)`,
              width: '20px', height: '20px', backgroundColor: 'white',
              borderRadius: '50%', boxShadow: '0 2px 4px rgba(0,0,0,0.3)'
            }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            <span>Subtle</span>
            <span>Complete</span>
          </div>
        </div>
      </div>

      {/* Advanced Settings */}
      <div className="grok-card" style={{ padding: 0 }}>
        <div
          onClick={() => setShowAdvanced(!showAdvanced)}
          style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            padding: '16px 20px', cursor: 'pointer',
            color: 'var(--text-secondary)',
          }}
        >
          <Settings size={16} />
          <span style={{ fontSize: '0.85rem', fontWeight: 500 }}>Advanced Settings</span>
          <ChevronDown size={16} style={{
            marginLeft: 'auto',
            transform: showAdvanced ? 'rotate(180deg)' : 'rotate(0deg)',
            transition: 'transform 0.2s'
          }} />
        </div>

        {showAdvanced && (
          <div style={{ padding: '0 20px 20px', borderTop: '1px solid var(--border-color)' }}>
            <div className="form-group" style={{ marginTop: '16px' }}>
              <label className="grok-section-label">Output FPS</label>
              <div className="grok-toggle-group">
                {[8, 12, 16, 24].map((f) => (
                  <button key={f} className={`grok-toggle-btn ${fps === f ? 'active' : ''}`} onClick={() => setFps(f)}>{f}</button>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label className="grok-section-label">Max Frames</label>
              <select
                className="form-input"
                value={maxFrames}
                onChange={(e) => setMaxFrames(parseInt(e.target.value))}
                style={{ cursor: 'pointer' }}
              >
                <option value={16}>16 frames (~2s @8fps)</option>
                <option value={32}>32 frames (~4s @8fps)</option>
                <option value={48}>48 frames (~6s @8fps)</option>
                <option value={64}>64 frames (~8s @8fps)</option>
                <option value={96}>96 frames (~12s @8fps)</option>
                <option value={128}>128 frames (~16s @8fps)</option>
                <option value={160}>160 frames (~20s @8fps)</option>
                <option value={192}>192 frames (~24s @8fps)</option>
                <option value={240}>240 frames (~30s @8fps)</option>
              </select>
            </div>

            <div style={{ display: 'flex', gap: '16px' }}>
              <div className="form-group" style={{ flex: 1 }}>
                <label className="grok-section-label">Steps</label>
                <input
                  className="form-input"
                  type="number"
                  min={10}
                  max={50}
                  value={steps}
                  onChange={(e) => setSteps(parseInt(e.target.value))}
                />
              </div>
              <div className="form-group" style={{ flex: 1 }}>
                <label className="grok-section-label">CFG Scale</label>
                <input
                  className="form-input"
                  type="number"
                  min={1}
                  max={15}
                  step={0.5}
                  value={cfg}
                  onChange={(e) => setCfg(parseFloat(e.target.value))}
                />
              </div>
            </div>

            <div className="form-group">
              <label className="grok-section-label">Seed</label>
              <div style={{ display: 'flex', gap: '8px' }}>
                <input
                  className="form-input"
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(parseInt(e.target.value) || -1)}
                  placeholder="-1 for random"
                  style={{ flex: 1 }}
                />
                <button className="icon-btn" onClick={() => setSeed(-1)} style={{ whiteSpace: 'nowrap', width: 'auto', padding: '0 12px', fontSize: '0.8rem' }}>Random</button>
              </div>
            </div>

            <div className="form-group">
              <label className="grok-section-label">Negative Prompt</label>
              <textarea
                className="form-textarea"
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
                rows={2}
                style={{ minHeight: '50px' }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Queued notification */}
      {lastQueued && (
        <div style={{
          padding: '12px', backgroundColor: 'rgba(34, 197, 94, 0.1)',
          border: '1px solid rgba(34, 197, 94, 0.3)', borderRadius: '8px',
          color: '#22c55e', marginBottom: '12px', fontSize: '0.85rem',
        }}>
          ✅ Job queued! Check the Queue panel for progress.
          <span style={{ marginLeft: '8px', fontWeight: 600, textTransform: 'uppercase', fontSize: '0.75rem' }}>{lastQueued.style}</span>
        </div>
      )}

      {error && (
        <div style={{
          padding: '12px', backgroundColor: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '8px',
          color: '#ef4444', marginBottom: '12px', fontSize: '0.85rem',
        }}>
          ⚠️ {error}
        </div>
      )}

      <button
        className="primary-btn"
        onClick={handleTransform}
        disabled={!file || submitting}
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
      >
        {submitting ? (
          <>
            <Loader2 size={18} className="animate-spin" />
            Queueing...
          </>
        ) : (
          <>
            <Wand2 size={18} />
            Transform Video
          </>
        )}
      </button>

      {/* Result */}
      {result && (
        <div style={{ marginTop: '24px', paddingTop: '24px', borderTop: '1px solid var(--border-color)' }}>
          <div className="grok-section-label">Result</div>
          <video src={result} controls style={{ width: '100%', borderRadius: '8px', marginTop: '12px' }} />
          <a
            href={result}
            download
            className="primary-btn"
            style={{ marginTop: 12, textAlign: 'center', textDecoration: 'none', display: 'block' }}
          >
            Download Video
          </a>
        </div>
      )}
    </div>
  )
}
