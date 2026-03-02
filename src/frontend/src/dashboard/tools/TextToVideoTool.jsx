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
import useLLMEnhance from '../../hooks/useLLMEnhance'
import LLMQueueIndicator from '../../components/LLMQueueIndicator'

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
    maxFrames: 481,
    defaultFrames: 41,
    frameStep: 4,
  },
  ltx2: {
    name: 'LTX-2 19B',
    description: 'Fast direct text-to-video',
    maxFrames: 481,
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

  // LLM prompt enhancement queue
  const llm = useLLMEnhance()

  // Enhance prompt with LLM (via async queue)
  const handleEnhancePrompt = async () => {
    if (!prompt.trim() || isEnhancing) return
    setIsEnhancing(true)
    setError('')

    const result = await llm.enhance({
      input: prompt.trim(),
      mode: 'expand',
      include_negative: true,
      include_motion: true,
      model: enhanceModel,
    })

    if (result) {
      handlePromptChange(result.prompt)
      if (result.negative_prompt) setNegativePrompt(result.negative_prompt)
    } else if (llm.error) {
      setError(`Enhance failed: ${llm.error}`)
    }
    setIsEnhancing(false)
  }

  // Refine/improve prompt with LLM — preserves original intent (via async queue)
  const handleRefinePrompt = async () => {
    if (!prompt.trim() || isRefining) return
    setIsRefining(true)
    setError('')

    const result = await llm.enhance({
      input: prompt.trim(),
      mode: 'refine',
      include_negative: true,
      include_motion: true,
      model: enhanceModel,
      refine_instruction: refineInstruction.trim() || null,
    })

    if (result) {
      handlePromptChange(result.prompt)
      if (result.negative_prompt) setNegativePrompt(result.negative_prompt)
      setRefineInstruction('')
      setShowRefineInput(false)
    } else if (llm.error) {
      setError(`Refine failed: ${llm.error}`)
    }
    setIsRefining(false)
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
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Video size={16} />
            Video Prompt
          </div>
          <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
            <select
              value={enhanceModel}
              onChange={(e) => setEnhanceModel(e.target.value)}
              style={{ fontSize: '10px', height: '24px', padding: '0 4px',
                background: 'var(--bg-input)', border: '1px solid var(--border-color)',
                borderRadius: '4px', color: 'var(--text-muted)',
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
            <LLMQueueIndicator queuePosition={llm.queuePosition} isLoading={llm.isLoading} />
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
        </div>

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
              className="form-input"
              style={{ flex: 1, padding: '6px 10px', fontSize: '0.8rem' }}
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
          className="form-textarea"
          value={prompt}
          onChange={(e) => handlePromptChange(e.target.value)}
          rows={4}
          placeholder="Describe the video you want to generate... (e.g., 'a cat walking through a field of flowers, cinematic')"
          style={{ minHeight: '80px' }}
        />
        <div style={{ textAlign: 'right', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
          {prompt.length} characters
        </div>

        {/* Camera Motion Selector */}
        <CameraMotionSelector value={cameraMotion} onChange={setCameraMotion} style={{ marginTop: '12px' }} />
      </div>

      {/* Settings */}
      <div className="grok-card">
        {/* Model Selection */}
        <div className="form-group">
          <label className="grok-section-label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Cpu size={14} />
            Model
          </label>
          <div className="grok-toggle-group">
            {Object.entries(availableModels).map(([key, config]) => (
              <button
                key={key}
                className={`grok-toggle-btn ${modelType === key ? 'active' : ''}`}
                onClick={() => setModelType(key)}
                type="button"
                title={config.description}
              >
                {config.name}
              </button>
            ))}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '6px', fontStyle: 'italic' }}>
            {availableModels[modelType]?.description}
          </div>
        </div>

        {/* Resolution */}
        <div className="form-group">
          <label className="grok-section-label">Resolution</label>
          <div className="grok-toggle-group">
            {RESOLUTION_PRESETS.map((preset) => (
              <button
                key={preset.value}
                className={`grok-toggle-btn ${resolution === preset.value ? 'active' : ''}`}
                onClick={() => setResolution(preset.value)}
                type="button"
              >
                {preset.label}
                <span style={{ fontSize: '0.7rem', opacity: 0.7, display: 'block' }}>{preset.desc}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Aspect Ratio */}
        <div className="form-group">
          <label className="grok-section-label">Aspect Ratio</label>
          <div className="grok-toggle-group">
            {ASPECT_RATIOS.map((ratio) => (
              <button
                key={ratio}
                className={`grok-toggle-btn ${aspectRatio === ratio ? 'active' : ''}`}
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
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <label className="grok-section-label" style={{ marginBottom: 0 }}>Frame Rate</label>
            <span className="nav-badge" style={{ fontSize: '0.8rem' }}>{fps} fps</span>
          </div>
          <div className="grok-toggle-group">
            {FPS_OPTIONS.map((f) => (
              <button
                key={f}
                className={`grok-toggle-btn ${fps === f ? 'active' : ''}`}
                onClick={() => setFps(f)}
                type="button"
              >
                {f}
              </button>
            ))}
          </div>
        </div>

        {/* Duration */}
        <div className="form-group">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <label className="grok-section-label" style={{ marginBottom: 0 }}>Duration</label>
            <span className="nav-badge" style={{ fontSize: '0.8rem' }}>{(numFrames / fps).toFixed(1)}s ({numFrames}f)</span>
          </div>
          <div style={{ position: 'relative', height: '24px', marginBottom: '8px' }}>
            <input
              type="range"
              min={modelType === 'ltx2' ? 9 : 17}
              max={availableModels[modelType]?.maxFrames || 81}
              step={availableModels[modelType]?.frameStep || 4}
              value={numFrames}
              onChange={(e) => setNumFrames(parseInt(e.target.value, 10))}
              style={{ width: '100%', opacity: 0, position: 'absolute', zIndex: 2, cursor: 'pointer' }}
            />
            <div style={{
              position: 'absolute', top: '10px', left: 0, right: 0,
              height: '4px', backgroundColor: '#333', borderRadius: '2px'
            }}>
              <div style={{
                width: `${((numFrames - (modelType === 'ltx2' ? 9 : 17)) / ((availableModels[modelType]?.maxFrames || 81) - (modelType === 'ltx2' ? 9 : 17))) * 100}%`,
                height: '100%', backgroundColor: 'var(--accent-color, #a855f7)', borderRadius: '2px'
              }} />
            </div>
            <div style={{
              position: 'absolute', top: '2px',
              left: `calc(${((numFrames - (modelType === 'ltx2' ? 9 : 17)) / ((availableModels[modelType]?.maxFrames || 81) - (modelType === 'ltx2' ? 9 : 17))) * 100}% - 10px)`,
              width: '20px', height: '20px', backgroundColor: 'white',
              borderRadius: '50%', boxShadow: '0 2px 4px rgba(0,0,0,0.3)'
            }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            <span>{((modelType === 'ltx2' ? 9 : 17) / fps).toFixed(1)}s</span>
            <span>{((availableModels[modelType]?.maxFrames || 81) / fps).toFixed(1)}s</span>
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
            <div style={{ display: 'flex', gap: '16px', marginTop: '16px' }}>
              <div className="form-group" style={{ flex: 1 }}>
                <label className="grok-section-label">Video Steps</label>
                <input
                  className="form-input"
                  type="number"
                  value={steps}
                  onChange={(e) => setSteps(parseInt(e.target.value) || 6)}
                  min="1"
                  max="30"
                />
              </div>
              <div className="form-group" style={{ flex: 1 }}>
                <label className="grok-section-label">Video CFG</label>
                <input
                  className="form-input"
                  type="number"
                  value={cfg}
                  onChange={(e) => setCfg(parseFloat(e.target.value) || 1.0)}
                  min="0.1"
                  max="10"
                  step="0.1"
                />
              </div>
            </div>

            <div style={{ display: 'flex', gap: '16px' }}>
              <div className="form-group" style={{ flex: 1 }}>
                <label className="grok-section-label">T2I Steps</label>
                <input
                  className="form-input"
                  type="number"
                  value={t2iSteps}
                  onChange={(e) => setT2iSteps(parseInt(e.target.value) || 20)}
                  min="1"
                  max="50"
                />
              </div>
              <div className="form-group" style={{ flex: 1 }}>
                <label className="grok-section-label">T2I CFG</label>
                <input
                  className="form-input"
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
                placeholder="Things to avoid..."
                style={{ minHeight: '50px' }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Post-Processing Options */}
      <div className="grok-card" style={{ padding: 0 }}>
        <div
          onClick={() => setShowPostProcessing(!showPostProcessing)}
          style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            padding: '16px 20px', cursor: 'pointer',
            color: 'var(--text-secondary)',
          }}
        >
          <Zap size={16} />
          <span style={{ fontSize: '0.85rem', fontWeight: 500 }}>Post-Processing</span>
          <ChevronDown size={16} style={{
            marginLeft: 'auto',
            transform: showPostProcessing ? 'rotate(180deg)' : 'rotate(0deg)',
            transition: 'transform 0.2s'
          }} />
        </div>

        {showPostProcessing && (
          <div style={{ padding: '0 20px 20px', borderTop: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '12px', marginBottom: '16px' }}>
              These will run automatically after generation completes.
            </div>

            {/* Upscale option */}
            <div className="form-group" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <label className="grok-switch">
                  <input type="checkbox" checked={postUpscale} onChange={(e) => setPostUpscale(e.target.checked)} />
                  <span className="grok-slider"></span>
                </label>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-primary)' }}>Upscale (Real-ESRGAN)</span>
              </div>
              {postUpscale && (
                <div className="grok-toggle-group" style={{ width: 'auto' }}>
                  <button className={`grok-toggle-btn ${postUpscaleScale === 2 ? 'active' : ''}`} onClick={() => setPostUpscaleScale(2)} style={{ padding: '4px 12px', fontSize: '0.8rem' }}>2x</button>
                  <button className={`grok-toggle-btn ${postUpscaleScale === 4 ? 'active' : ''}`} onClick={() => setPostUpscaleScale(4)} style={{ padding: '4px 12px', fontSize: '0.8rem' }}>4x</button>
                </div>
              )}
            </div>

            {/* Frame interpolation option */}
            <div className="form-group" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <label className="grok-switch">
                  <input type="checkbox" checked={postInterpolate} onChange={(e) => setPostInterpolate(e.target.checked)} />
                  <span className="grok-slider"></span>
                </label>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-primary)' }}>Interpolation (RIFE)</span>
              </div>
              {postInterpolate && (
                <div className="grok-toggle-group" style={{ width: 'auto' }}>
                  {[30, 48, 60].map((targetFps) => (
                    <button key={targetFps} className={`grok-toggle-btn ${postInterpolateFps === targetFps ? 'active' : ''}`} onClick={() => setPostInterpolateFps(targetFps)} style={{ padding: '4px 12px', fontSize: '0.8rem' }}>{targetFps}</button>
                  ))}
                </div>
              )}
            </div>

            {(postUpscale || postInterpolate) && (
              <div className="info-badge" style={{ marginTop: '8px' }}>
                ℹ️ Post-processing adds extra credits: {postUpscale ? '+5 upscale' : ''}{postUpscale && postInterpolate ? ', ' : ''}{postInterpolate ? '+3 interpolation' : ''}
              </div>
            )}
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

      {/* Time estimate indicator */}
      {!submitting && canSubmit && (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          gap: '6px', marginBottom: '8px', fontSize: '0.85rem', color: 'var(--text-muted)',
        }}>
          <Clock size={14} />
          <span>Estimated time: ~{timeEstimate.formatted}</span>
        </div>
      )}

      <button
        className="primary-btn"
        type="button"
        disabled={!canSubmit}
        onClick={handleSubmit}
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
      >
        {submitting ? (
          <>
            <Loader2 size={18} className="animate-spin" />
            Queueing...
          </>
        ) : (
          <>
            <Video size={18} />
            Generate Video
          </>
        )}
      </button>

      <div className="info-badge" style={{ marginTop: '12px', textAlign: 'center' }}>
        💡 {modelType === 'ltx2'
          ? 'LTX-2 generates video directly from text (faster)'
          : 'Wan2.2 first generates an image, then animates it (higher quality)'}
      </div>
    </div>
  )
}
