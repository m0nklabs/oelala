import React, { useMemo, useState, useRef } from 'react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm } from '../../api'
import { sendClientLog } from '../../logging'
import { Settings, Wand2, Loader2, Video, ChevronDown } from 'lucide-react'

// Resolution presets
const RESOLUTION_PRESETS = [
  { value: '480p', label: '480p', desc: 'Fast' },
  { value: '720p', label: '720p', desc: 'Balanced' },
]

const FPS_OPTIONS = [8, 12, 16, 24]
const ASPECT_RATIOS = ['16:9', '9:16', '1:1', '4:3', '3:4']

// Camera motion presets for Wan2.2
const CAMERA_MOTIONS = [
  { value: '', label: 'None', desc: 'No camera motion', prefix: '' },
  { value: 'static', label: '📷 Static', desc: 'Camera stays still', prefix: 'static camera shot, ' },
  { value: 'pan_left', label: '⬅️ Pan Left', desc: 'Camera pans left', prefix: 'camera slowly panning left, ' },
  { value: 'pan_right', label: '➡️ Pan Right', desc: 'Camera pans right', prefix: 'camera slowly panning right, ' },
  { value: 'tilt_up', label: '⬆️ Tilt Up', desc: 'Camera tilts up', prefix: 'camera slowly tilting up, ' },
  { value: 'tilt_down', label: '⬇️ Tilt Down', desc: 'Camera tilts down', prefix: 'camera slowly tilting down, ' },
  { value: 'zoom_in', label: '🔍 Zoom In', desc: 'Camera zooms in', prefix: 'camera slowly zooming in, ' },
  { value: 'zoom_out', label: '🔭 Zoom Out', desc: 'Camera zooms out', prefix: 'camera slowly zooming out, ' },
  { value: 'dolly_in', label: '🎬 Dolly In', desc: 'Camera moves forward', prefix: 'camera dollying forward, ' },
  { value: 'dolly_out', label: '🎬 Dolly Out', desc: 'Camera moves back', prefix: 'camera dollying backward, ' },
  { value: 'orbit_left', label: '🔄 Orbit Left', desc: 'Camera orbits left', prefix: 'camera orbiting left around subject, ' },
  { value: 'orbit_right', label: '🔄 Orbit Right', desc: 'Camera orbits right', prefix: 'camera orbiting right around subject, ' },
  { value: 'handheld', label: '📹 Handheld', desc: 'Slight shake', prefix: 'shaky handheld camera, ' },
  { value: 'tracking', label: '🏃 Tracking', desc: 'Follows subject', prefix: 'camera tracking shot following subject, ' },
  { value: 'crane_up', label: '🏗️ Crane Up', desc: 'Camera rises up', prefix: 'crane shot rising up, ' },
  { value: 'crane_down', label: '🏗️ Crane Down', desc: 'Camera lowers', prefix: 'crane shot lowering down, ' },
]

export default function TextToVideoTool({ onOutput, onRefreshHistory, onJobSubmitted }) {
  const [prompt, setPrompt] = useState(() => localStorage.getItem('t2v_prompt') || '')
  const [negativePrompt, setNegativePrompt] = useState('blurry, low quality, distorted, ugly')
  const [numFrames, setNumFrames] = useState(41)
  const [aspectRatio, setAspectRatio] = useState('1:1')
  const [resolution, setResolution] = useState('480p')
  const [fps, setFps] = useState(16)
  const [cameraMotion, setCameraMotion] = useState('')
  
  // Advanced settings
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [steps, setSteps] = useState(6)
  const [cfg, setCfg] = useState(1.0)
  const [seed, setSeed] = useState(-1)
  const [t2iSteps, setT2iSteps] = useState(20)
  const [t2iCfg, setT2iCfg] = useState(6.0)

  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [status, setStatus] = useState('')
  const [progress, setProgress] = useState(0)
  
  const pollerRef = useRef(null)

  // Save prompt to localStorage
  const handlePromptChange = (value) => {
    setPrompt(value)
    localStorage.setItem('t2v_prompt', value)
  }

  const canSubmit = useMemo(() => prompt.trim().length > 0 && !busy, [prompt, busy])

  // Poll for job completion
  const pollForCompletion = async (promptId, maxAttempts = 180) => {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      await new Promise(resolve => setTimeout(resolve, 1000))
      try {
        const res = await fetch(`${BACKEND_BASE}/comfyui/job/${promptId}`)
        if (!res.ok) continue
        const data = await res.json()
        
        if (data.status === 'pending') {
          setStatus('Queued...')
          setProgress(Math.min(10, attempt))
        } else if (data.status === 'running') {
          setStatus('Generating...')
          setProgress(Math.min(90, 10 + attempt))
        } else if (data.status === 'completed') {
          setProgress(100)
          setStatus('Done!')
          return data
        } else if (data.status === 'failed') {
          throw new Error(data.error || 'Generation failed')
        }
      } catch (e) {
        if (e.message.includes('failed')) throw e
      }
    }
    throw new Error('Generation timed out')
  }

  const handleSubmit = async () => {
    if (!prompt.trim()) {
      setError('Prompt is required')
      return
    }

    setBusy(true)
    setError('')
    setStatus('Submitting...')
    setProgress(0)

    // Build prompt with camera motion prefix
    const motionPreset = CAMERA_MOTIONS.find(m => m.value === cameraMotion)
    const motionPrefix = motionPreset?.prefix || ''
    const finalPrompt = motionPrefix + prompt

    const formData = new FormData()
    formData.append('prompt', finalPrompt)
    formData.append('num_frames', String(numFrames))
    formData.append('aspect_ratio', aspectRatio)
    formData.append('resolution', resolution)
    formData.append('fps', String(fps))

    try {
      if (DEBUG) console.debug('🎬 T2V request:', { prompt, numFrames, resolution, fps })
      
      const result = await postForm(`${BACKEND_BASE}/generate-text`, formData)
      
      if (!result.ok) {
        throw new Error(result.data?.detail || `Generation failed (status ${result.status})`)
      }

      const promptId = result.data?.prompt_id
      if (!promptId) {
        throw new Error('No prompt_id returned')
      }

      if (DEBUG) console.debug('📋 T2V queued:', promptId)
      setStatus('Queued...')
      
      // Notify queue indicator
      if (onJobSubmitted) onJobSubmitted()

      // Poll for completion
      const completed = await pollForCompletion(promptId)
      
      if (completed.output_video || completed.url) {
        const videoUrl = completed.output_video || completed.url
        const fullUrl = videoUrl.startsWith('http') ? videoUrl : `${BACKEND_BASE}${videoUrl}`
        
        onOutput({
          kind: 'video',
          url: fullUrl,
          backendUrl: fullUrl,
          filename: videoUrl.split('/').pop(),
          meta: { ...result.data?.meta, prompt_id: promptId },
        })
        
        if (onRefreshHistory) onRefreshHistory()
      }
      
    } catch (e) {
      const message = e?.message || 'Failed to generate video'
      setError(message)
      await sendClientLog({
        level: 'error',
        message: 'Text-to-video failed',
        timestamp: new Date().toISOString(),
        meta: { message },
      })
    } finally {
      setBusy(false)
      setStatus('')
      setProgress(0)
    }
  }

  return (
    <div className="tool-container">
      {/* Prompt Card */}
      <div className="tool-section">
        <h3>
          <Video size={18} />
          Video Prompt
        </h3>
        <textarea
          className="prompt-textarea"
          value={prompt}
          onChange={(e) => handlePromptChange(e.target.value)}
          rows={4}
          placeholder="Describe the video you want to generate... (e.g., 'a cat walking through a field of flowers, cinematic')"
        />
        <div className="char-count">{prompt.length} characters</div>
        
        {/* Camera Motion Selector */}
        <div style={{ marginTop: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Camera Motion</span>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
              {cameraMotion ? CAMERA_MOTIONS.find(m => m.value === cameraMotion)?.desc : 'Optional'}
            </span>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {CAMERA_MOTIONS.map(motion => (
              <button
                key={motion.value}
                onClick={() => setCameraMotion(motion.value === cameraMotion ? '' : motion.value)}
                type="button"
                style={{
                  padding: '6px 10px',
                  borderRadius: '6px',
                  border: cameraMotion === motion.value ? '1px solid var(--accent-color)' : '1px solid var(--border-color)',
                  background: cameraMotion === motion.value ? 'rgba(59, 130, 246, 0.2)' : 'rgba(255,255,255,0.05)',
                  color: cameraMotion === motion.value ? 'var(--accent-color)' : 'var(--text-secondary)',
                  fontSize: '0.8rem',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
                title={motion.desc}
              >
                {motion.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Quick Settings */}
      <div className="tool-section">
        <h3>Settings</h3>
        
        {/* Resolution */}
        <div className="form-group">
          <label>Resolution</label>
          <div className="button-group">
            {RESOLUTION_PRESETS.map((preset) => (
              <button
                key={preset.value}
                className={`btn-option ${resolution === preset.value ? 'active' : ''}`}
                onClick={() => setResolution(preset.value)}
                type="button"
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>

        {/* Aspect Ratio */}
        <div className="form-group">
          <label>Aspect Ratio</label>
          <div className="button-group">
            {ASPECT_RATIOS.map((ratio) => (
              <button
                key={ratio}
                className={`btn-option ${aspectRatio === ratio ? 'active' : ''}`}
                onClick={() => setAspectRatio(ratio)}
                type="button"
              >
                {ratio}
              </button>
            ))}
          </div>
        </div>

        {/* FPS */}
        <div className="form-group">
          <label>Frame Rate</label>
          <div className="button-group">
            {FPS_OPTIONS.map((f) => (
              <button
                key={f}
                className={`btn-option ${fps === f ? 'active' : ''}`}
                onClick={() => setFps(f)}
                type="button"
              >
                {f} fps
              </button>
            ))}
          </div>
        </div>

        {/* Duration */}
        <div className="form-group">
          <label>
            Duration 
            <span className="label-value">{(numFrames / fps).toFixed(1)}s ({numFrames} frames)</span>
          </label>
          <input
            type="range"
            min="17"
            max="81"
            step="4"
            value={numFrames}
            onChange={(e) => setNumFrames(parseInt(e.target.value, 10))}
          />
          <div className="range-labels">
            <span>{(17 / fps).toFixed(1)}s</span>
            <span>{(81 / fps).toFixed(1)}s</span>
          </div>
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
            <div className="form-row">
              <div className="form-group half">
                <label>Video Steps</label>
                <input
                  type="number"
                  value={steps}
                  onChange={(e) => setSteps(parseInt(e.target.value) || 6)}
                  min="1"
                  max="30"
                />
              </div>
              <div className="form-group half">
                <label>Video CFG</label>
                <input
                  type="number"
                  value={cfg}
                  onChange={(e) => setCfg(parseFloat(e.target.value) || 1.0)}
                  min="0.1"
                  max="10"
                  step="0.1"
                />
              </div>
            </div>
            
            <div className="form-row">
              <div className="form-group half">
                <label>T2I Steps</label>
                <input
                  type="number"
                  value={t2iSteps}
                  onChange={(e) => setT2iSteps(parseInt(e.target.value) || 20)}
                  min="1"
                  max="50"
                />
              </div>
              <div className="form-group half">
                <label>T2I CFG</label>
                <input
                  type="number"
                  value={t2iCfg}
                  onChange={(e) => setT2iCfg(parseFloat(e.target.value) || 6.0)}
                  min="1"
                  max="20"
                  step="0.5"
                />
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

            <div className="form-group">
              <label>Negative Prompt</label>
              <textarea
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
                rows={2}
                placeholder="Things to avoid..."
              />
            </div>
          </div>
        )}
      </div>

      {/* Progress */}
      {busy && (
        <div className="progress-section">
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
          <div className="progress-status">
            <Loader2 size={16} className="spin" />
            {status}
          </div>
        </div>
      )}

      {error && <div className="error-message">⚠️ {error}</div>}

      <button 
        className="btn-primary btn-large" 
        type="button" 
        disabled={!canSubmit} 
        onClick={handleSubmit}
      >
        {busy ? (
          <>
            <Loader2 size={18} className="spin" />
            Generating...
          </>
        ) : (
          <>
            <Video size={18} />
            Generate Video
          </>
        )}
      </button>

      <div className="tool-info">
        💡 Text-to-Video first generates an image from your prompt, then animates it using Wan2.2
      </div>

      <style>{`
        .prompt-textarea {
          width: 100%;
          padding: 12px;
          border-radius: 8px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          font-family: inherit;
          font-size: 14px;
          resize: vertical;
        }
        .char-count {
          text-align: right;
          font-size: 12px;
          color: var(--text-muted, #888);
          margin-top: 4px;
        }
        .form-group {
          margin-bottom: 16px;
        }
        .form-group label {
          display: flex;
          justify-content: space-between;
          margin-bottom: 8px;
          font-size: 13px;
          color: var(--text-secondary, #aaa);
        }
        .label-value {
          color: var(--accent-color, #7c3aed);
          font-weight: 500;
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
          font-size: 13px;
        }
        .btn-option:hover {
          border-color: var(--accent-color, #7c3aed);
        }
        .btn-option.active {
          background: var(--accent-color, #7c3aed);
          border-color: var(--accent-color, #7c3aed);
        }
        .range-labels {
          display: flex;
          justify-content: space-between;
          font-size: 11px;
          color: var(--text-muted, #888);
          margin-top: 4px;
        }
        .form-row {
          display: flex;
          gap: 16px;
        }
        .form-group.half {
          flex: 1;
        }
        .form-group input[type="number"],
        .form-group textarea {
          width: 100%;
          padding: 8px 12px;
          border-radius: 6px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          font-size: 13px;
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
        .tool-info {
          margin-top: 16px;
          padding: 12px;
          background: rgba(124, 58, 237, 0.1);
          border-radius: 8px;
          font-size: 13px;
          color: var(--text-secondary, #aaa);
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
