import React, { useMemo, useState, useEffect } from 'react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm } from '../../api'
import { useAuth } from '../../contexts/AuthContext'
import { sendClientLog } from '../../logging'
import { Settings, Wand2, Loader2, Video, ChevronDown, Sparkles, Clock, Cpu, Zap, Pencil } from 'lucide-react'
import CameraMotionSelector, { getCameraMotionPrefix } from '../../components/CameraMotionSelector'
import { getDefaultPrompt, getRandomPrompt } from '../../data/defaultPrompts'
import { estimateT2VTime } from '../../utils/timeEstimates'
import MediaImportModal from '../../components/MediaImportModal'

// Resolution presets
const RESOLUTION_PRESETS = [
  { value: '480p', label: '480p', desc: 'Fast' },
  { value: '720p', label: '720p', desc: 'Balanced' },
]

const FPS_OPTIONS = [8, 12, 16, 24]
const ASPECT_RATIOS = ['16:9', '9:16', '1:1', '4:3', '3:4']

// T2V Models
const T2V_MODELS = {
  wan22: {
    name: 'Wan2.2 14B',
    description: 'High quality T2V with T2I pipeline',
    maxFrames: 81,
    defaultFrames: 41,
    frameStep: 4,
  },
  ltx2: {
    name: 'LTX-2 19B',
    description: 'Fast direct text-to-video',
    maxFrames: 97,
    defaultFrames: 25,
    frameStep: 1,
  },
}

export default function TextToVideoTool({ onOutput, onRefreshHistory, onJobSubmitted, pendingImport = null, onImportConsumed = null }) {
  const { user, requestLogin } = useAuth()

  const [prompt, setPrompt] = useState(() => {
    const saved = localStorage.getItem('t2v_prompt')
    return saved && saved.trim() ? saved : getDefaultPrompt(false)
  })
  const [negativePrompt, setNegativePrompt] = useState('blurry, low quality, distorted, ugly')
  const [modelType, setModelType] = useState('wan22')
  const [numFrames, setNumFrames] = useState(41)
  const [aspectRatio, setAspectRatio] = useState('1:1')
  const [resolution, setResolution] = useState('480p')
  const [fps, setFps] = useState(16)
  const [cameraMotion, setCameraMotion] = useState('')
  const [isEnhancing, setIsEnhancing] = useState(false)
  const [isRefining, setIsRefining] = useState(false)
  const [showRefineInput, setShowRefineInput] = useState(false)
  const [refineInstruction, setRefineInstruction] = useState('')
  const [enhanceModel, setEnhanceModel] = useState('GLM-4.7-Flash-Claude-Opus-Reasoning')

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
  const [availableModels, setAvailableModels] = useState(T2V_MODELS)

  // Pending import modal state
  const [importModal, setImportModal] = useState(null)  // { item, workflow }

  // When Dashboard sends a new pendingImport, show the modal
  useEffect(() => {
    if (!pendingImport) return
    setImportModal(pendingImport)
    if (onImportConsumed) onImportConsumed()
  }, [pendingImport])

  const handleApplyImport = (selected) => {
    if (selected.positive) setPrompt(selected.positive)
    if (selected.steps)    setSteps(selected.steps)
    if (selected.cfg)      setCfg(selected.cfg)
    if (selected.seed)     setSeed(selected.seed)
    setImportModal(null)
  }

  // Post-processing options (chained jobs after generation)
  const [showPostProcessing, setShowPostProcessing] = useState(false)
  const [postUpscale, setPostUpscale] = useState(false)
  const [postUpscaleScale, setPostUpscaleScale] = useState(2)
  const [postInterpolate, setPostInterpolate] = useState(false)
  const [postInterpolateFps, setPostInterpolateFps] = useState(60)

  // Fetch available T2V modes from backend
  useEffect(() => {
    const fetchT2VModes = async () => {
      try {
        const res = await fetch(`${BACKEND_BASE}/api/t2v-modes`)
        if (res.ok) {
          const data = await res.json()
          if (data.modes) {
            // Merge backend config with frontend display info
            const merged = { ...T2V_MODELS }
            Object.entries(data.modes).forEach(([key, config]) => {
              if (merged[key]) {
                merged[key] = { ...merged[key], ...config }
              } else {
                merged[key] = config
              }
            })
            setAvailableModels(merged)
          }
        }
      } catch (e) {
        if (DEBUG) console.warn('Failed to fetch T2V modes:', e)
      }
    }
    fetchT2VModes()
  }, [])

  // Update frames when model changes
  useEffect(() => {
    const modelConfig = availableModels[modelType]
    if (modelConfig) {
      // Clamp numFrames to model's max
      if (numFrames > modelConfig.maxFrames) {
        setNumFrames(modelConfig.defaultFrames)
      }
    }
  }, [modelType, availableModels])

  // Save prompt to localStorage
  const handlePromptChange = (value) => {
    setPrompt(value)
    localStorage.setItem('t2v_prompt', value)
  }

  // Enhance prompt with LLM
  const handleEnhancePrompt = async () => {
    if (!prompt.trim() || isEnhancing) return
    setIsEnhancing(true)
    setError('')

    try {
      const res = await fetch(`${BACKEND_BASE}/generate-prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: prompt.trim(),
          style: null,
          mode: 'expand',
          include_negative: true,
          include_motion: true,
          use_llm: true,
          model: enhanceModel,
        }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Enhancement failed')
      }

      const data = await res.json()
      if (DEBUG) console.log('✨ Enhanced prompt:', data)

      // Update prompts
      handlePromptChange(data.prompt)
      if (data.negative_prompt) {
        setNegativePrompt(data.negative_prompt)
      }
    } catch (err) {
      console.error('Enhance error:', err)
      setError(`Enhance failed: ${err.message}`)
    } finally {
      setIsEnhancing(false)
    }
  }

  // Refine/improve prompt with LLM — preserves original intent
  const handleRefinePrompt = async () => {
    if (!prompt.trim() || isRefining) return
    setIsRefining(true)
    setError('')

    try {
      const res = await fetch(`${BACKEND_BASE}/generate-prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: prompt.trim(),
          style: null,
          mode: 'refine',
          include_negative: true,
          include_motion: true,
          use_llm: true,
          model: enhanceModel,
          refine_instruction: refineInstruction.trim() || null,
        }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Refine failed')
      }

      const data = await res.json()
      if (DEBUG) console.log('✏️ Refined prompt:', data)

      handlePromptChange(data.prompt)
      if (data.negative_prompt) {
        setNegativePrompt(data.negative_prompt)
      }
      setRefineInstruction('')
      setShowRefineInput(false)
    } catch (err) {
      console.error('Refine error:', err)
      setError(`Refine failed: ${err.message}`)
    } finally {
      setIsRefining(false)
    }
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
    formData.append('model_type', modelType)
    formData.append('aspect_ratio', aspectRatio)
    formData.append('resolution', resolution)
    formData.append('fps', String(fps))

    // Add post-processing chain if any options selected
    const postProcessingSteps = []
    if (postUpscale) {
      postProcessingSteps.push({ type: 'upscale', scale: postUpscaleScale, model: 'realesrgan-x4plus' })
    }
    if (postInterpolate) {
      postProcessingSteps.push({ type: 'interpolate', target_fps: postInterpolateFps })
    }
    if (postProcessingSteps.length > 0) {
      formData.append('post_processing', JSON.stringify(postProcessingSteps))
    }

    try {
      if (DEBUG) console.debug('🎬 T2V request:', { prompt, modelType, numFrames, resolution, fps })

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
      {/* Import from previous generation modal */}
      {importModal && (
        <MediaImportModal
          item={importModal.item}
          parsedData={importModal.workflow}
          availableFields={['positive', 'steps', 'cfg', 'seed']}
          onApply={handleApplyImport}
          onClose={() => setImportModal(null)}
        />
      )}

      {/* Prompt Card */}
      <div className="tool-section">
        <h3 style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Video size={18} />
            Video Prompt
          </span>
          <div style={{ display: 'flex', gap: '4px' }}>
            <select
              value={enhanceModel}
              onChange={(e) => setEnhanceModel(e.target.value)}
              style={{ fontSize: '10px', height: '24px', padding: '0 4px',
                background: 'var(--bg-secondary, #1a1a1a)',
                border: '1px solid var(--border-color, #444)',
                borderRadius: '4px', color: 'var(--text-muted, #aaa)',
                cursor: 'pointer', maxWidth: '120px' }}
              title="LLM model for prompt enhancement"
            >
              <option value="GLM-4.7-Flash-Claude-Opus-Reasoning">GLM+Claude ✨</option>
              <option value="GLM-4.7-Flash">GLM Flash</option>
              <option value="GLM-4.7-Flash-Uncensored-Balanced">GLM Uncensored</option>
              <option value="Qwen3-30B-A3B-Thinking-2507">Qwen3 30B</option>
              <option value="gemma-3-27b-it">Gemma 27B</option>
            </select>
            <button
              className="icon-btn"
              style={{ width: '24px', height: '24px', padding: '4px' }}
              onClick={handleEnhancePrompt}
              disabled={isEnhancing || !prompt.trim()}
              title="Enhance prompt with AI (reimagines creatively)"
            >
              {isEnhancing ? <Loader2 size={12} className="spin" /> : <Wand2 size={12} />}
            </button>
            <button
              className="icon-btn"
              style={{
                width: '24px', height: '24px', padding: '4px',
                background: showRefineInput ? 'var(--accent-color, #8b5cf6)' : undefined,
                color: showRefineInput ? 'white' : undefined,
              }}
              onClick={() => setShowRefineInput(!showRefineInput)}
              disabled={!prompt.trim()}
              title="Refine/improve prompt (keeps original intent)"
            >
              <Pencil size={12} />
            </button>
            <button
              className="icon-btn"
              style={{ width: '24px', height: '24px', fontSize: '14px' }}
              onClick={() => handlePromptChange(getRandomPrompt(false))}
              title="Generate random creative prompt"
            >
              ✨
            </button>
          </div>
        </h3>

        {/* Refine Prompt - inline instruction input */}
        {showRefineInput && (
          <div style={{
            marginBottom: '8px',
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
              onKeyDown={(e) => { if (e.key === 'Enter' && prompt.trim()) handleRefinePrompt() }}
              placeholder="What to improve? (e.g., add more motion, better lighting...) — leave empty for general polish"
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
              }}
              onClick={handleRefinePrompt}
              disabled={isRefining || !prompt.trim()}
              title="Refine prompt with AI"
            >
              {isRefining ? <Loader2 size={12} className="spin" /> : <Pencil size={12} />}
              <span>{isRefining ? 'Refining...' : 'Refine'}</span>
            </button>
          </div>
        )}

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

        {/* Model Selection */}
        <div className="form-group">
          <label style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Cpu size={14} />
            Model
          </label>
          <div className="button-group">
            {Object.entries(availableModels).map(([key, config]) => (
              <button
                key={key}
                className={`btn-option ${modelType === key ? 'active' : ''}`}
                onClick={() => setModelType(key)}
                type="button"
                title={config.description}
              >
                {config.name}
              </button>
            ))}
          </div>
          <div className="model-hint">
            {availableModels[modelType]?.description}
          </div>
        </div>

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
            min={modelType === 'ltx2' ? 9 : 17}
            max={availableModels[modelType]?.maxFrames || 81}
            step={availableModels[modelType]?.frameStep || 4}
            value={numFrames}
            onChange={(e) => setNumFrames(parseInt(e.target.value, 10))}
          />
          <div className="range-labels">
            <span>{((modelType === 'ltx2' ? 9 : 17) / fps).toFixed(1)}s</span>
            <span>{((availableModels[modelType]?.maxFrames || 81) / fps).toFixed(1)}s</span>
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

      {/* Post-Processing Options */}
      <div className="tool-section collapsible">
        <button
          className="section-toggle"
          onClick={() => setShowPostProcessing(!showPostProcessing)}
        >
          <Zap size={16} />
          Post-Processing
          <ChevronDown size={16} className={showPostProcessing ? 'rotated' : ''} />
        </button>

        {showPostProcessing && (
          <div className="advanced-content">
            <p className="help-text" style={{ marginBottom: '12px', fontSize: '0.85rem' }}>
              These will run automatically after generation completes.
            </p>

            {/* Upscale option */}
            <div className="form-group" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={postUpscale}
                  onChange={(e) => setPostUpscale(e.target.checked)}
                />
                Upscale video (Real-ESRGAN)
              </label>
              {postUpscale && (
                <select
                  value={postUpscaleScale}
                  onChange={(e) => setPostUpscaleScale(parseInt(e.target.value))}
                  style={{ width: 'auto', padding: '4px 8px' }}
                >
                  <option value={2}>2x</option>
                  <option value={4}>4x</option>
                </select>
              )}
            </div>

            {/* Frame interpolation option */}
            <div className="form-group" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={postInterpolate}
                  onChange={(e) => setPostInterpolate(e.target.checked)}
                />
                Frame interpolation (RIFE)
              </label>
              {postInterpolate && (
                <select
                  value={postInterpolateFps}
                  onChange={(e) => setPostInterpolateFps(parseInt(e.target.value))}
                  style={{ width: 'auto', padding: '4px 8px' }}
                >
                  <option value={30}>30 fps</option>
                  <option value={48}>48 fps</option>
                  <option value={60}>60 fps</option>
                </select>
              )}
            </div>

            {(postUpscale || postInterpolate) && (
              <p className="help-text" style={{ marginTop: '8px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                ℹ️ Post-processing adds extra credits: {postUpscale ? '+5 upscale' : ''}{postUpscale && postInterpolate ? ', ' : ''}{postInterpolate ? '+3 interpolation' : ''}
              </p>
            )}
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
        💡 {modelType === 'ltx2'
          ? 'LTX-2 generates video directly from text (faster)'
          : 'Wan2.2 first generates an image, then animates it (higher quality)'}
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
        .model-hint {
          font-size: 11px;
          color: var(--text-muted, #888);
          margin-top: 6px;
          font-style: italic;
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
