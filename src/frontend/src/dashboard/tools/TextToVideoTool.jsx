import React, { useMemo, useState } from 'react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm } from '../../api'
import { useAuth } from '../../contexts/AuthContext'
import { sendClientLog } from '../../logging'
import { Settings, Wand2, Loader2, Video, ChevronDown, Sparkles, Clock } from 'lucide-react'
import CameraMotionSelector, { getCameraMotionPrefix } from '../../components/CameraMotionSelector'
import { getDefaultPrompt, getRandomPrompt } from '../../data/defaultPrompts'
import { estimateT2VTime } from '../../utils/timeEstimates'

// Resolution presets
const RESOLUTION_PRESETS = [
  { value: '480p', label: '480p', desc: 'Fast' },
  { value: '720p', label: '720p', desc: 'Balanced' },
]

const FPS_OPTIONS = [8, 12, 16, 24]
const ASPECT_RATIOS = ['16:9', '9:16', '1:1', '4:3', '3:4']

export default function TextToVideoTool({ onOutput, onRefreshHistory, onJobSubmitted }) {
  const { user, requestLogin } = useAuth()

  const [prompt, setPrompt] = useState(() => {
    const saved = localStorage.getItem('t2v_prompt')
    return saved && saved.trim() ? saved : getDefaultPrompt(false)
  })
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

  const [submitting, setSubmitting] = useState(false)  // Brief state while submitting
  const [error, setError] = useState('')
  const [lastQueued, setLastQueued] = useState(null)   // Track last queued job

  // Save prompt to localStorage
  const handlePromptChange = (value) => {
    setPrompt(value)
    localStorage.setItem('t2v_prompt', value)
  }

  const canSubmit = useMemo(() => prompt.trim().length > 0 && !submitting, [prompt, submitting])

  // Calculate estimated generation time
  const timeEstimate = useMemo(() => {
    return estimateT2VTime({ resolution, numFrames, steps, t2iSteps })
  }, [resolution, numFrames, steps, t2iSteps])

  const handleSubmit = async () => {
    // Check if user is logged in
    if (!user) {
      requestLogin('Log in om te genereren')
      return
    }

    if (!prompt.trim()) {
      setError('Prompt is required')
      return
    }

    setSubmitting(true)
    setError('')
    setLastQueued(null)

    // Build prompt with camera motion prefix
    const motionPrefix = getCameraMotionPrefix(cameraMotion)
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

      // Show queued confirmation
      setLastQueued({
        promptId,
        prompt: prompt.substring(0, 40) + (prompt.length > 40 ? '...' : '')
      })

      // Notify queue indicator - job will be tracked in queue panel
      if (onJobSubmitted) onJobSubmitted({ prompt_id: promptId })

      // Don't wait for completion - job will appear in queue/history when done

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
      setSubmitting(false)
    }
  }

  return (
    <div className="tool-container">
      {/* Prompt Card */}
      <div className="tool-section">
        <h3 style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Video size={18} />
            Video Prompt
          </span>
          <button
            className="icon-btn"
            style={{ width: '28px', height: '28px', fontSize: '16px' }}
            onClick={() => handlePromptChange(getRandomPrompt(false))}
            title="Generate random creative prompt"
          >
            ✨
          </button>
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
        <CameraMotionSelector value={cameraMotion} onChange={setCameraMotion} style={{ marginTop: '12px' }} />
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

      {/* Queued notification */}
      {lastQueued && (
        <div className="queued-notice">
          ✅ Job queued! Check the Queue panel for progress.
        </div>
      )}

      {error && <div className="error-message">⚠️ {error}</div>}

      {/* Time estimate indicator */}
      {!submitting && canSubmit && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '6px',
          marginBottom: '8px',
          fontSize: '0.85rem',
          color: 'var(--text-muted)',
        }}>
          <Clock size={14} />
          <span>Estimated time: ~{timeEstimate.formatted}</span>
        </div>
      )}

      <button
        className="btn-primary btn-large"
        type="button"
        disabled={!canSubmit}
        onClick={handleSubmit}
      >
        {submitting ? (
          <>
            <Loader2 size={18} className="spin" />
            Queueing...
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
