import React, { useMemo, useState, useEffect, useCallback } from 'react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm } from '../../api'
import { useAuth } from '../../contexts/AuthContext'
import { useNSFW } from '../../contexts/NSFWContext'
import { sendClientLog } from '../../logging'
import { Settings, Wand2, Loader2, Video, ChevronDown, Sparkles, Clock, Cpu, Zap, Pencil, Settings2, Save, Check, X, Layers, HelpCircle, Sliders } from 'lucide-react'
import CameraMotionSelector, { getCameraMotionPrefix } from '../../components/CameraMotionSelector'
import { getDefaultPrompt, getRandomPrompt } from '../../data/defaultPrompts'
import { estimateT2VTime } from '../../utils/timeEstimates'
import MediaImportModal from '../../components/MediaImportModal'
import useLLMEnhance from '../../hooks/useLLMEnhance'
import LLMQueueIndicator from '../../components/LLMQueueIndicator'
import { PROMPT_LLM_MODELS, DEFAULT_PROMPT_LLM } from '../../constants/llmModels'
import { useToolProfile } from '../../hooks/useToolProfile'

// Resolution presets with pixel dimensions per aspect ratio (aligned with I2V)
const RESOLUTION_PRESETS = {
  '480p': {
    label: '480p',
    dimensions: {
      '16:9': '848×480', '9:16': '480×848', '1:1': '480×480', '4:3': '640×480', '3:4': '480×640'
    },
    max_duration: 30,
  },
  '576p': {
    label: '576p',
    dimensions: {
      '16:9': '1024×576', '9:16': '576×1024', '1:1': '576×576', '4:3': '768×576', '3:4': '576×768'
    },
    max_duration: 30,
  },
  '720p': {
    label: '720p',
    dimensions: {
      '16:9': '1280×720', '9:16': '720×1280', '1:1': '720×720', '4:3': '960×720', '3:4': '720×960'
    },
    max_duration: 30,
  },
}

const FPS_OPTIONS = [8, 12, 16, 24]
const ASPECT_RATIOS = ['16:9', '9:16', '1:1', '4:3', '3:4']

// T2V Model modes (aligned with I2V MODEL_MODES format)
const MODEL_MODES = [
  { value: 'cloud_max', label: '☁️ Cloud Max — bf16 Full Precision', desc: 'Cloud GPU • bf16 unquantized • 25 steps • Maximum quality' },
  { value: 'wan22', label: '🎬 Wan2.2 14B Q6 DisTorch2', desc: 'High quality dual-pass T2V via ComfyUI' },
  { value: 'ltx2', label: '⚡ LTX-2 19B Distilled', desc: 'Fast direct text-to-video' },
]

const T2V_DEFAULTS = {
  prompt: '',
  negativePrompt: 'blurry, low quality, distorted, ugly, artifacts, overexposed, underexposed, flickering, jitter',
  modelType: 'wan22',
  duration: 5,
  aspectRatio: '9:16',
  resolution: '480p',
  fps: 16,
  cameraMotion: '',
  enhanceModel: DEFAULT_PROMPT_LLM,
  steps: 6,
  cfg: 1.0,
  seed: -1,
  t2iSteps: 20,
  t2iCfg: 6.0,
  computeTarget: 'local',
  postUpscale: false,
  postUpscaleScale: 2,
  postInterpolate: false,
  postInterpolateFps: 60,
  loraConfigs: [],
  unetHighNoise: 'wan2.2_t2v_14B_Q6_K.gguf',
  unetLowNoise: '',
  extendMode: false,
  clipCount: 1,
}

export default function TextToVideoTool({ onOutput, onRefreshHistory, onJobSubmitted, pendingImport = null, onImportConsumed = null }) {
  const { user, requestLogin } = useAuth()
  const { nsfwEnabled } = useNSFW()

  // ── Profile persistence (auto-save on every change) ──────────────────
  const applyProfileSettings = useCallback((s) => {
    if (s.prompt) setPrompt(s.prompt)
    if (s.negativePrompt !== undefined) setNegativePrompt(s.negativePrompt)
    if (s.duration !== undefined) setDuration(s.duration)
    if (s.resolution) setResolution(s.resolution)
    if (s.modelType) setModelType(s.modelType)
    if (s.aspectRatio) setAspectRatio(s.aspectRatio)
    if (s.fps !== undefined) setFps(s.fps)
    if (s.steps !== undefined) setSteps(s.steps)
    if (s.cfg !== undefined) setCfg(s.cfg)
    if (s.seed !== undefined) setSeed(s.seed)
    if (s.cameraMotion !== undefined) setCameraMotion(s.cameraMotion)
    if (s.t2iSteps !== undefined) setT2iSteps(s.t2iSteps)
    if (s.t2iCfg !== undefined) setT2iCfg(s.t2iCfg)
    if (s.enhanceModel) setEnhanceModel(s.enhanceModel)
    if (s.computeTarget) setComputeTarget(s.computeTarget)
    if (s.loraConfigs !== undefined) setLoraConfigs(s.loraConfigs)
    if (s.unetHighNoise) setUnetHighNoise(s.unetHighNoise)
    if (s.unetLowNoise !== undefined) setUnetLowNoise(s.unetLowNoise)
    if (s.extendMode !== undefined) setExtendMode(s.extendMode)
    if (s.clipCount !== undefined) setClipCount(s.clipCount)
    if (s.postUpscale !== undefined) setPostUpscale(s.postUpscale)
    if (s.postUpscaleScale !== undefined) setPostUpscaleScale(s.postUpscaleScale)
    if (s.postInterpolate !== undefined) setPostInterpolate(s.postInterpolate)
    if (s.postInterpolateFps !== undefined) setPostInterpolateFps(s.postInterpolateFps)
  }, [])

  const {
    settings: profileSettings,
    updateSettings: updateProfile,
    saveAs: saveProfileAs,
    profiles: profileList,
    presets: factoryPresets,
    loadProfiles,
    switchProfile,
    deleteProfile,
    applyPreset,
    loaded: profileLoaded,
    saving: profileSaving,
    activeProfile: activeProfileName,
  } = useToolProfile('text_to_video', T2V_DEFAULTS, {
    onLoad: applyProfileSettings,
  })
  const [showProfileMenu, setShowProfileMenu] = useState(false)
  const [profileSaveInput, setProfileSaveInput] = useState('')

  const [prompt, setPrompt] = useState(() => getDefaultPrompt(false))
  const [negativePrompt, setNegativePrompt] = useState(T2V_DEFAULTS.negativePrompt)
  const [showNegativePrompt, setShowNegativePrompt] = useState(false)
  const [showPromptTips, setShowPromptTips] = useState(false)
  const [modelType, setModelType] = useState('wan22')
  const [duration, setDuration] = useState(5) // seconds, 3-30 range
  const [aspectRatio, setAspectRatio] = useState('9:16')
  const [resolution, setResolution] = useState('480p')
  const [fps, setFps] = useState(16)
  const [cameraMotion, setCameraMotion] = useState('')
  const [isEnhancing, setIsEnhancing] = useState(false)
  const [isRefining, setIsRefining] = useState(false)
  const [showRefineInput, setShowRefineInput] = useState(false)
  const [refineInstruction, setRefineInstruction] = useState('')
  const [enhanceModel, setEnhanceModel] = useState(DEFAULT_PROMPT_LLM)

  // Advanced settings
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [steps, setSteps] = useState(6)
  const [cfg, setCfg] = useState(1.0)
  const [seed, setSeed] = useState(-1)
  const [t2iSteps, setT2iSteps] = useState(20)
  const [t2iCfg, setT2iCfg] = useState(6.0)

  const [computeTarget, setComputeTarget] = useState('local')

  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [lastQueued, setLastQueued] = useState(null)
  const [availableModels, setAvailableModels] = useState({})

  // LoRA state - multi-LoRA with individual strengths
  const [availableLoras, setAvailableLoras] = useState({ high_noise: [], low_noise: [], general: [] })
  const [loraConfigs, setLoraConfigs] = useState([])
  const [showLoraPanel, setShowLoraPanel] = useState(false)

  // Unet model state
  const [availableUnets, setAvailableUnets] = useState({ high_noise: [], low_noise: [], pairs: [] })
  const [unetHighNoise, setUnetHighNoise] = useState('wan2.2_t2v_14B_Q6_K.gguf')
  const [unetLowNoise, setUnetLowNoise] = useState('')
  const [showUnetPanel, setShowUnetPanel] = useState(false)

  // Extend Duration - Sequential clip generation
  const [extendMode, setExtendMode] = useState(false)
  const [clipCount, setClipCount] = useState(1)

  // Post-processing options (chained jobs after generation)
  const [showPostProcessing, setShowPostProcessing] = useState(false)
  const [postUpscale, setPostUpscale] = useState(false)
  const [postUpscaleScale, setPostUpscaleScale] = useState(2)
  const [postInterpolate, setPostInterpolate] = useState(false)
  const [postInterpolateFps, setPostInterpolateFps] = useState(60)
  const [postAudio, setPostAudio] = useState(false)
  const [postAudioFile, setPostAudioFile] = useState(null)

  // Pending import modal state
  const [importModal, setImportModal] = useState(null)

  // When Dashboard sends a new pendingImport, show the modal
  useEffect(() => {
    if (!pendingImport) return
    setImportModal(pendingImport)
    if (onImportConsumed) onImportConsumed()
  }, [pendingImport])

  const handleApplyImport = (selected) => {
    if (selected.positive) setPrompt(selected.positive)
    if (selected.negative) setNegativePrompt(selected.negative)
    if (selected.steps)    setSteps(Number(selected.steps) || selected.steps)
    if (selected.cfg)      setCfg(Number(selected.cfg) || selected.cfg)
    if (selected.seed)     setSeed(String(selected.seed))
    if (selected.loras && Array.isArray(selected.loras)) {
      setLoraConfigs(selected.loras.map(l => ({
        high: l.high || '',
        low: l.low || '',
        strength: l.strength ?? 1.0,
      })))
    }
    setImportModal(null)
  }

  // Calculate max duration based on resolution
  const maxDuration = useMemo(() => {
    const preset = RESOLUTION_PRESETS[resolution]
    return preset?.max_duration || 30
  }, [resolution])

  // Clamp duration when max changes
  useEffect(() => {
    if (duration > maxDuration) setDuration(maxDuration)
  }, [maxDuration, duration])

  // Fetch available T2V modes from backend
  useEffect(() => {
    const fetchT2VModes = async () => {
      try {
        const res = await fetch(`${BACKEND_BASE}/api/t2v-modes`)
        if (res.ok) {
          const data = await res.json()
          if (data.modes) setAvailableModels(data.modes)
        }
      } catch (e) {
        if (DEBUG) console.warn('Failed to fetch T2V modes:', e)
      }
    }
    fetchT2VModes()
  }, [])

  // Fetch available LoRAs on mount
  useEffect(() => {
    const fetchLoras = async () => {
      try {
        const res = await fetch(`${BACKEND_BASE}/loras`)
        if (res.ok) {
          const data = await res.json()
          setAvailableLoras(data)
          if (DEBUG) console.debug('🐛 T2V loaded LoRAs:', data.count)
        }
      } catch (e) {
        console.error('Failed to fetch LoRAs:', e)
      }
    }
    fetchLoras()
  }, [])

  // Filter LoRAs based on NSFW setting
  const filteredLoras = useMemo(() => {
    if (nsfwEnabled) return availableLoras
    const filterList = (list) => (list || []).filter(l => !l.nsfw)
    const filteredByCategory = {}
    if (availableLoras.by_category) {
      Object.keys(availableLoras.by_category).forEach(cat => {
        const filtered = filterList(availableLoras.by_category[cat])
        if (filtered.length > 0) filteredByCategory[cat] = filtered
      })
    }
    return {
      high_noise: filterList(availableLoras.high_noise),
      low_noise: filterList(availableLoras.low_noise),
      general: filterList(availableLoras.general),
      loras: filterList(availableLoras.loras),
      by_category: filteredByCategory,
    }
  }, [availableLoras, nsfwEnabled])

  // Fetch available unet models on mount
  useEffect(() => {
    const fetchUnets = async () => {
      try {
        const res = await fetch(`${BACKEND_BASE}/unet-models`)
        if (res.ok) {
          const data = await res.json()
          setAvailableUnets(data)
          if (DEBUG) console.debug('🐛 T2V loaded Unet models:', data.count)
        }
      } catch (e) {
        console.error('Failed to fetch Unet models:', e)
      }
    }
    fetchUnets()
  }, [])

  const handlePromptChange = (value) => {
    setPrompt(value)
  }

  // ── Auto-save settings to profile on every change ─────────────────────
  const settingsSnapshot = useMemo(() => ({
    prompt, negativePrompt, modelType, duration, aspectRatio, resolution,
    fps, cameraMotion, enhanceModel, steps, cfg, seed, t2iSteps, t2iCfg,
    computeTarget, loraConfigs, unetHighNoise, unetLowNoise,
    extendMode, clipCount,
    postUpscale, postUpscaleScale, postInterpolate, postInterpolateFps,
  }), [
    prompt, negativePrompt, modelType, duration, aspectRatio, resolution,
    fps, cameraMotion, enhanceModel, steps, cfg, seed, t2iSteps, t2iCfg,
    computeTarget, loraConfigs, unetHighNoise, unetLowNoise,
    extendMode, clipCount,
    postUpscale, postUpscaleScale, postInterpolate, postInterpolateFps,
  ])

  useEffect(() => {
    if (!profileLoaded || !user) return
    updateProfile(settingsSnapshot)
  }, [settingsSnapshot, profileLoaded, user, updateProfile])

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
  const numFrames = duration * fps
  const timeEstimate = useMemo(() => {
    return estimateT2VTime({ resolution, numFrames: duration * fps, steps, t2iSteps })
  }, [resolution, duration, fps, steps, t2iSteps])

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

    const calcNumFrames = duration * fps
    const formData = new FormData()
    formData.append('prompt', finalPrompt)
    formData.append('num_frames', String(calcNumFrames))
    formData.append('model_type', modelType)
    formData.append('aspect_ratio', aspectRatio)
    formData.append('resolution', resolution)
    formData.append('fps', String(fps))
    formData.append('compute_target', computeTarget)

    // Add negative prompt
    formData.append('negative_prompt', negativePrompt)

    // Add post-processing chain if any options selected
    const postProcessingSteps = []
    if (postUpscale) {
      postProcessingSteps.push({ type: 'upscale', scale: postUpscaleScale, model: 'realesrgan-x4plus' })
    }
    if (postInterpolate) {
      postProcessingSteps.push({ type: 'interpolate', target_fps: postInterpolateFps })
    }
    if (postAudio && postAudioFile) {
      formData.append('post_audio_file', postAudioFile)
      postProcessingSteps.push({ type: 'add_audio' })
    }
    if (postProcessingSteps.length > 0) {
      formData.append('post_processing', JSON.stringify(postProcessingSteps))
    }

    // LoRA parameters
    if (loraConfigs.length > 0) {
      formData.append('lora_configs', JSON.stringify(loraConfigs))
    }

    // Extend mode
    if (extendMode && clipCount > 1) {
      formData.append('extend_mode', 'true')
      formData.append('clip_count', String(clipCount))
    }

    try {
      if (DEBUG) console.debug('🎬 T2V request:', { prompt, modelType, numFrames: calcNumFrames, resolution, fps, duration })

      let t2vEndpoint = `${BACKEND_BASE}/generate-text`

      // Cloud Max uses its own endpoint with mode=t2v
      if (modelType === 'cloud_max') {
        t2vEndpoint = `${BACKEND_BASE}/generate-cloud-max-async`
        formData.append('mode', 't2v')
        formData.append('steps', String(steps))
        formData.append('cfg', String(cfg))
        formData.append('seed', String(seed))
        formData.append('shift', '8.0')
        formData.append('high_noise_steps', '12')
        formData.append('sampler_name', 'dpmpp_2m')
        formData.append('scheduler', 'beta')
      }

      const result = await postForm(t2vEndpoint, formData)

      if (!result.ok) {
        throw new Error(result.data?.detail || `Generation failed (status ${result.status})`)
      }

      const promptId = result.data?.prompt_id
      if (!promptId) {
        throw new Error('No prompt_id returned')
      }

      const isCloud = result.data?.compute_target === 'cloud'
      if (DEBUG) console.debug(`🐛 Job queued (${isCloud ? 'cloud' : 'local'}):`, result.data?.prompt_id || result.data?.runpod_job_id)

      // Show queued confirmation
      setLastQueued({
        promptId,
        prompt: prompt.substring(0, 40) + (prompt.length > 40 ? '...' : '')
      })

      // Notify queue indicator - job will be tracked in queue panel
      if (onJobSubmitted) onJobSubmitted({ prompt_id: promptId })

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
          availableFields={['positive', 'negative', 'steps', 'cfg', 'seed', 'loras']}
          onApply={handleApplyImport}
          onClose={() => setImportModal(null)}
        />
      )}

      {/* Model Selection Card */}
      <div className="grok-card">
        {/* ── Settings Profile Bar ─────────────────────────────────── */}
        {user && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            padding: '8px 12px', borderBottom: '1px solid var(--border-subtle, #333)',
            fontSize: '12px', color: 'var(--text-secondary, #999)',
          }}>
            <Settings2 size={14} />
            <span style={{ opacity: 0.7 }}>Profile:</span>
            <span style={{ color: 'var(--text-primary, #eee)', fontWeight: 500 }}>
              {activeProfileName || 'default'}
            </span>
            {profileSaving && (
              <span style={{ display: 'flex', alignItems: 'center', gap: '3px', color: 'var(--accent, #7c3aed)', fontSize: '11px' }}>
                <Loader2 size={11} className="animate-spin" /> saving...
              </span>
            )}
            {!profileSaving && profileLoaded && (
              <Check size={12} style={{ color: '#22c55e', opacity: 0.6 }} />
            )}
            <div style={{ marginLeft: 'auto', display: 'flex', gap: '6px' }}>
              <button
                type="button"
                onClick={() => {
                  setShowProfileMenu(!showProfileMenu)
                  if (!showProfileMenu) loadProfiles()
                }}
                style={{
                  background: 'none', border: '1px solid var(--border-subtle, #444)',
                  borderRadius: '4px', padding: '2px 8px', cursor: 'pointer',
                  color: 'var(--text-secondary, #999)', fontSize: '11px',
                }}
              >
                <Save size={11} style={{ marginRight: '3px', verticalAlign: '-1px' }} />
                Profiles
              </button>
            </div>
          </div>
        )}
        {/* Profile dropdown menu */}
        {showProfileMenu && user && (
          <div style={{
            padding: '8px 12px', borderBottom: '1px solid var(--border-subtle, #333)',
            background: 'var(--bg-secondary, #1a1a2e)', fontSize: '12px',
          }}>
            {/* Save As */}
            <div style={{ display: 'flex', gap: '6px', marginBottom: '8px' }}>
              <input
                type="text"
                placeholder="Profile name..."
                value={profileSaveInput}
                onChange={(e) => setProfileSaveInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && profileSaveInput.trim()) {
                    saveProfileAs(profileSaveInput.trim())
                      .then(() => { setProfileSaveInput(''); loadProfiles() })
                      .catch((err) => setError(err.message))
                  }
                }}
                style={{
                  flex: 1, background: 'var(--bg-tertiary, #111)', border: '1px solid var(--border-subtle, #444)',
                  borderRadius: '4px', padding: '4px 8px', color: 'var(--text-primary, #eee)',
                  fontSize: '12px', outline: 'none',
                }}
              />
              <button
                type="button"
                disabled={!profileSaveInput.trim()}
                onClick={() => {
                  if (profileSaveInput.trim()) {
                    saveProfileAs(profileSaveInput.trim())
                      .then(() => { setProfileSaveInput(''); loadProfiles() })
                      .catch((err) => setError(err.message))
                  }
                }}
                style={{
                  background: 'var(--accent, #7c3aed)', border: 'none', borderRadius: '4px',
                  padding: '4px 10px', cursor: 'pointer', color: '#fff', fontSize: '11px',
                  opacity: profileSaveInput.trim() ? 1 : 0.4,
                }}
              >
                Save As
              </button>
            </div>
            {/* Profile list */}
            {profileList.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {profileList.map((p) => (
                  <div
                    key={p.id}
                    style={{
                      display: 'flex', alignItems: 'center', gap: '8px',
                      padding: '4px 8px', borderRadius: '4px', cursor: 'pointer',
                      background: p.profile_name === activeProfileName ? 'var(--accent-bg, rgba(124,58,237,0.15))' : 'transparent',
                      border: p.profile_name === activeProfileName ? '1px solid var(--accent, #7c3aed)' : '1px solid transparent',
                    }}
                  >
                    <span
                      onClick={() => { switchProfile(p.profile_name); setShowProfileMenu(false) }}
                      style={{ flex: 1, color: 'var(--text-primary, #eee)' }}
                    >
                      {p.profile_name}
                      {p.profile_name === activeProfileName && <Check size={11} style={{ marginLeft: '4px', color: '#22c55e' }} />}
                    </span>
                    <span style={{ color: 'var(--text-tertiary, #666)', fontSize: '10px' }}>
                      {new Date(p.updated_at).toLocaleDateString()}
                    </span>
                    {p.profile_name !== 'default' && (
                      <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); deleteProfile(p.profile_name) }}
                        style={{
                          background: 'none', border: 'none', cursor: 'pointer',
                          color: 'var(--text-tertiary, #666)', padding: '0 2px',
                        }}
                        title="Delete profile"
                      >
                        <X size={12} />
                      </button>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ color: 'var(--text-tertiary, #666)', fontSize: '11px' }}>
                No saved profiles yet. Your settings auto-save to &quot;default&quot;.
              </div>
            )}
            {/* Factory Presets */}
            {factoryPresets?.length > 0 && (
              <div style={{ marginTop: '10px', borderTop: '1px solid var(--border-subtle, #333)', paddingTop: '8px' }}>
                <div style={{ fontSize: '11px', color: 'var(--text-secondary, #999)', marginBottom: '6px', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px' }}>
                  Factory Presets
                  <span style={{ fontWeight: 400, opacity: 0.6 }}>(best-tested settings)</span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  {factoryPresets.map((preset, idx) => (
                    <div
                      key={idx}
                      onClick={() => { applyPreset(preset); setShowProfileMenu(false) }}
                      style={{
                        display: 'flex', flexDirection: 'column', gap: '2px',
                        padding: '6px 8px', borderRadius: '4px', cursor: 'pointer',
                        border: '1px solid var(--border-subtle, #333)',
                        background: 'var(--bg-tertiary, #111)',
                        transition: 'border-color 0.15s',
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.borderColor = 'var(--accent, #7c3aed)'}
                      onMouseLeave={(e) => e.currentTarget.style.borderColor = 'var(--border-subtle, #333)'}
                    >
                      <span style={{ color: 'var(--text-primary, #eee)', fontSize: '12px', fontWeight: 500 }}>
                        {preset.name}
                      </span>
                      <span style={{ color: 'var(--text-tertiary, #666)', fontSize: '10px' }}>
                        {preset.description}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        <div className="grok-card-header">
          <div className="grok-card-title">Model Selection</div>
        </div>

        <div className="form-group">
          <label className="grok-section-label">Generation Mode</label>
          <div style={{ position: 'relative' }}>
            <select
              value={modelType}
              onChange={(e) => {
                const newMode = e.target.value
                setModelType(newMode)
                if (newMode === 'wan22') {
                  setResolution('480p')
                  setAspectRatio('9:16')
                  setDuration(5)
                  setSteps(6)
                  setCfg(1.0)
                  if (computeTarget === 'cloud') setComputeTarget('local')
                } else if (newMode === 'ltx2') {
                  setResolution('576p')
                  setAspectRatio('9:16')
                  setDuration(5)
                  setSteps(20)
                  setCfg(3.0)
                  if (computeTarget === 'cloud') setComputeTarget('local')
                } else if (newMode === 'cloud_max') {
                  setResolution('720p')
                  setAspectRatio('9:16')
                  setDuration(8)
                  setSteps(25)
                  setCfg(3.0)
                  setComputeTarget('cloud')
                }
              }}
              style={{
                width: '100%',
                padding: '12px 40px 12px 16px',
                backgroundColor: 'var(--bg-secondary, #1a1a1a)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px',
                color: 'var(--text-primary, #fff)',
                fontSize: '1rem',
                appearance: 'none',
                cursor: 'pointer',
              }}
            >
              {MODEL_MODES.map((mode) => (
                <option key={mode.value} value={mode.value} style={{ backgroundColor: '#1a1a1a', color: '#fff' }}>
                  {mode.label}
                </option>
              ))}
            </select>
            <ChevronDown size={20} style={{ position: 'absolute', right: '12px', top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', color: 'var(--text-muted)' }} />
          </div>

          {/* Model info badges */}
          {modelType === 'cloud_max' ? (
            <div className="info-badge" style={{ marginTop: '8px', borderColor: '#f472b6' }}>
              <span style={{ fontWeight: 600 }}>Cloud Max — bf16 Full Precision</span> | <span style={{ color: '#f9a8d4' }}>RunPod A6000/A40</span>
              <div style={{ marginTop: '4px', opacity: 0.8 }}>Unquantized bf16 | 48GB VRAM | 25 steps | Maximum quality</div>
              <div style={{ marginTop: '2px', opacity: 0.6, fontSize: '0.75rem' }}>~$1.22/hr | Cloud-only | Up to 720p 30s</div>
            </div>
          ) : modelType === 'wan22' ? (
            <div className="info-badge" style={{ marginTop: '8px' }}>
              <span style={{ fontWeight: 600 }}>Wan2.2 14B Q6</span> | <span style={{ color: '#93c5fd' }}>DisTorch2 Multi-GPU</span>
              <div style={{ marginTop: '4px', opacity: 0.8 }}>T2I first pass + I2V animation | All resolutions up to 30s</div>
            </div>
          ) : (
            <div className="info-badge" style={{ marginTop: '8px' }}>
              <span style={{ fontWeight: 600 }}>LTX-2 19B</span> | <span style={{ color: '#86efac' }}>Direct T2V</span>
              <div style={{ marginTop: '4px', opacity: 0.8 }}>Faster inference | No T2I pass | Uses Gemma 3 text encoder</div>
            </div>
          )}

          {/* Cloud GPU Toggle */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '8px' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-secondary, #888)', minWidth: '70px' }}>Compute:</span>
            <div style={{ display: 'flex', gap: '4px' }}>
              <button
                type="button"
                onClick={() => { if (modelType !== 'cloud_max') setComputeTarget('local') }}
                disabled={modelType === 'cloud_max'}
                style={{
                  padding: '4px 10px', fontSize: '11px', borderRadius: '4px', border: '1px solid',
                  borderColor: computeTarget === 'local' ? 'var(--accent-color, #6366f1)' : 'var(--border-color, #333)',
                  background: computeTarget === 'local' ? 'var(--accent-color, #6366f1)' : 'transparent',
                  color: computeTarget === 'local' ? '#fff' : 'var(--text-secondary, #888)',
                  cursor: modelType === 'cloud_max' ? 'not-allowed' : 'pointer',
                  opacity: modelType === 'cloud_max' ? 0.4 : 1, transition: 'all 0.15s ease',
                }}
              >
                Local
              </button>
              <button
                type="button"
                onClick={() => setComputeTarget('cloud')}
                style={{
                  padding: '4px 10px', fontSize: '11px', borderRadius: '4px', border: '1px solid',
                  borderColor: computeTarget === 'cloud' ? '#10b981' : 'var(--border-color, #333)',
                  background: computeTarget === 'cloud' ? '#10b981' : 'transparent',
                  color: computeTarget === 'cloud' ? '#fff' : 'var(--text-secondary, #888)',
                  cursor: 'pointer', transition: 'all 0.15s ease',
                }}
              >
                Cloud
              </button>
            </div>
            {computeTarget === 'cloud' && (
              <span style={{ fontSize: '10px', color: '#10b981' }}>RunPod GPU</span>
            )}
          </div>
        </div>

        {/* Unet Model Selection - Only for Wan2.2 */}
        {modelType === 'wan22' && (
          <div style={{ marginTop: '12px', paddingTop: '12px', borderTop: '1px solid var(--border-color)' }}>
            <div
              onClick={() => setShowUnetPanel(!showUnetPanel)}
              style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer', padding: '4px 0' }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Settings2 size={16} />
                <span style={{ fontSize: '0.9rem', fontWeight: 500 }}>Unet Model</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  ({unetHighNoise.replace('.gguf', '').replace('wan2.2_', '')})
                </span>
              </div>
              <span style={{ opacity: 0.5, fontSize: '0.8rem' }}>{showUnetPanel ? '▼' : '▶'}</span>
            </div>

            {showUnetPanel && (
              <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {/* Model Pair Selector */}
                <div>
                  <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
                    Model Pair (recommended)
                  </label>
                  <select
                    onChange={(e) => {
                      const pair = availableUnets.pairs?.find(p => p.name === e.target.value)
                      if (pair) {
                        setUnetHighNoise(pair.high.path)
                        setUnetLowNoise(pair.low?.path || '')
                      }
                    }}
                    style={{
                      width: '100%', padding: '8px 12px', backgroundColor: 'var(--bg-secondary)',
                      border: '1px solid var(--border-color)', borderRadius: '6px',
                      color: 'var(--text-primary)', fontSize: '0.85rem'
                    }}
                    value={availableUnets.pairs?.find(p => p.high.path === unetHighNoise)?.name || ''}
                  >
                    {availableUnets.pairs?.map((pair) => (
                      <option key={pair.name} value={pair.name}>
                        {pair.name} ({pair.high.size_gb}GB)
                      </option>
                    ))}
                  </select>
                </div>

                <details style={{ fontSize: '0.8rem' }}>
                  <summary style={{ cursor: 'pointer', color: 'var(--text-muted)', marginBottom: '8px' }}>
                    Advanced: Select model separately
                  </summary>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', paddingTop: '8px' }}>
                    <div>
                      <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
                        T2V Model
                      </label>
                      <select
                        value={unetHighNoise}
                        onChange={(e) => setUnetHighNoise(e.target.value)}
                        style={{
                          width: '100%', padding: '8px 12px', backgroundColor: 'var(--bg-secondary)',
                          border: '1px solid var(--border-color)', borderRadius: '6px',
                          color: 'var(--text-primary)', fontSize: '0.85rem'
                        }}
                      >
                        {availableUnets.high_noise?.map((model) => (
                          <option key={model.path} value={model.path}>
                            {model.name} ({model.size_gb}GB)
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                </details>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Prompt Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            Positive Prompt <span style={{ fontWeight: 400, color: 'var(--text-muted)', fontSize: '0.85rem' }}>(Describe the video)</span>
            <div style={{ position: 'relative', display: 'inline-block' }}>
              <button
                className="icon-btn"
                style={{ width: '20px', height: '20px', border: 'none', background: 'transparent', padding: 0, fontSize: '14px' }}
                onClick={() => setShowPromptTips(!showPromptTips)}
                title="Prompt tips"
              >
                {showPromptTips ? '💡' : '❓'}
              </button>
              {showPromptTips && (
                <div style={{
                  position: 'absolute', top: '100%', left: '50%', transform: 'translateX(-50%)',
                  marginTop: '8px', backgroundColor: '#1a1a1a', border: '1px solid #fbbf24',
                  borderRadius: '8px', padding: '12px', width: '280px', zIndex: 100,
                  fontSize: '0.8rem', color: '#fbbf24', boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
                }}>
                  <div style={{ fontWeight: 600, marginBottom: '8px' }}>Prompt Tips</div>
                  <ul style={{ margin: 0, paddingLeft: '16px', lineHeight: 1.6 }}>
                    <li>Structure: [subject + action] + [scene] + [camera]</li>
                    <li>Focus on motion — &quot;walking slowly&quot;, &quot;hair blowing&quot;</li>
                    <li>Add intensity — &quot;quickly&quot;, &quot;gently&quot;, &quot;dramatically&quot;</li>
                    <li>Camera moves — &quot;slow zoom in&quot;, &quot;pan left&quot;</li>
                    <li>Describe what you want, not what to avoid</li>
                  </ul>
                </div>
              )}
            </div>
          </div>
          <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
            <select
              value={enhanceModel}
              onChange={(e) => setEnhanceModel(e.target.value)}
              style={{ fontSize: '10px', height: '24px', padding: '0 4px',
                background: 'var(--bg-secondary, #1a1a1a)', border: '1px solid var(--border-color, #444)',
                borderRadius: '4px', color: 'var(--text-muted, #aaa)',
                cursor: 'pointer', maxWidth: '120px' }}
              title="LLM model for prompt enhancement"
            >
              {PROMPT_LLM_MODELS.map((m) => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
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
              onClick={() => handlePromptChange(getRandomPrompt(nsfwEnabled))}
              title="Generate random creative prompt"
            >
              ✨
            </button>
          </div>
        </div>

        {/* Refine Prompt - inline instruction input */}
        {showRefineInput && (
          <div style={{
            marginTop: '8px', padding: '8px 12px',
            backgroundColor: 'rgba(139, 92, 246, 0.08)',
            border: '1px solid rgba(139, 92, 246, 0.25)',
            borderRadius: '8px', display: 'flex', gap: '8px', alignItems: 'center',
          }}>
            <Pencil size={14} style={{ color: '#a78bfa', flexShrink: 0 }} />
            <input
              type="text"
              value={refineInstruction}
              onChange={(e) => setRefineInstruction(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && prompt.trim()) handleRefinePrompt() }}
              placeholder="What to improve? (e.g., add more motion, better lighting...) — leave empty for general polish"
              style={{
                flex: 1, background: 'var(--bg-input, #1a1a1a)', border: '1px solid var(--border-color, #444)',
                borderRadius: '6px', padding: '6px 10px', fontSize: '0.8rem',
                color: 'var(--text-primary, #eee)', outline: 'none',
              }}
            />
            <button
              className="icon-btn"
              style={{
                height: '28px', padding: '4px 12px', fontSize: '0.75rem',
                display: 'flex', alignItems: 'center', gap: '4px',
                background: isRefining ? 'var(--bg-input)' : 'linear-gradient(135deg, #8b5cf6, #6d28d9)',
                color: 'white', border: 'none', borderRadius: '6px', whiteSpace: 'nowrap',
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

        {/* Camera Motion Selector */}
        <CameraMotionSelector value={cameraMotion} onChange={setCameraMotion} />

        <div style={{ position: 'relative' }}>
          <textarea
            className="form-textarea"
            value={prompt}
            onChange={(e) => handlePromptChange(e.target.value)}
            rows={4}
            placeholder="Describe the video you want to generate... (e.g., 'a cat walking through a field of flowers, cinematic lighting')"
            style={{
              backgroundColor: '#0f0f0f', border: '1px solid var(--border-color)',
              borderRadius: '8px', resize: 'vertical', minHeight: '80px',
              padding: '12px', paddingBottom: '28px', width: '100%', boxSizing: 'border-box'
            }}
          />
          <div style={{ position: 'absolute', bottom: '8px', right: '8px', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            {prompt.length}/2048
          </div>
        </div>

        {/* Negative Prompt - Collapsible */}
        <div style={{ marginTop: '12px' }}>
          <div
            onClick={() => setShowNegativePrompt(!showNegativePrompt)}
            style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer', padding: '8px 0' }}
          >
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Negative Prompt
            </span>
            <span style={{ opacity: 0.5, fontSize: '0.8rem' }}>{showNegativePrompt ? '▼' : '▶'}</span>
          </div>

          {showNegativePrompt && (
            <div style={{ position: 'relative' }}>
              <textarea
                className="form-textarea"
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
                rows={3}
                placeholder="Things to avoid in the generation..."
                style={{
                  backgroundColor: '#0f0f0f', border: '1px solid var(--border-color)',
                  borderRadius: '8px', resize: 'vertical', minHeight: '60px',
                  padding: '12px', paddingBottom: '28px', width: '100%',
                  boxSizing: 'border-box', fontSize: '0.85rem'
                }}
              />
              <div style={{ position: 'absolute', bottom: '8px', right: '8px', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                {negativePrompt.length}/2048
              </div>
            </div>
          )}
        </div>

        {/* Prompt Strength / CFG Slider */}
        <div style={{ marginTop: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Prompt Strength / CFG
              </label>
              <span title="How strictly the video follows your prompt. Low = subtle, High = dramatic (may cause artifacts)" style={{ cursor: 'help', opacity: 0.5 }}>
                <HelpCircle size={12} />
              </span>
            </div>
            <span style={{
              fontSize: '0.8rem', fontWeight: 600,
              color: cfg <= 1.5 ? '#fbbf24' : cfg <= 3 ? '#34d399' : '#f87171',
              padding: '2px 8px', borderRadius: '4px',
              backgroundColor: cfg <= 1.5 ? 'rgba(251,191,36,0.1)' : cfg <= 3 ? 'rgba(52,211,153,0.1)' : 'rgba(248,113,113,0.1)',
            }}>
              {cfg <= 1.5 ? 'Subtle' : cfg <= 3 ? 'Balanced' : 'Strong'}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <input
              type="range" min="1" max="5" step="0.1" value={cfg}
              onChange={(e) => setCfg(parseFloat(e.target.value))}
              style={{ flex: 1 }}
            />
            <span style={{ fontSize: '0.85rem', fontWeight: 600, minWidth: '32px', textAlign: 'right', color: 'var(--text-primary)' }}>
              {cfg.toFixed(1)}
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            <span>Subtle motion</span>
            <span>Strong action</span>
          </div>
        </div>
      </div>

      {/* Settings Card */}
      <div className="grok-card">
        {/* Resolution */}
        <div className="form-group">
          <label className="grok-section-label">
            Resolution
            <span className="text-muted" style={{ fontWeight: 400 }}>{' (Higher = Better Quality, more VRAM)'}</span>
          </label>
          <div className="grok-toggle-group">
            {Object.entries(RESOLUTION_PRESETS).map(([key, preset]) => (
              <button
                key={key}
                className={`grok-toggle-btn ${resolution === key ? 'active' : ''}`}
                onClick={() => setResolution(key)}
                type="button"
              >
                {preset.label}
                <span style={{ fontSize: '0.7rem', opacity: 0.7, display: 'block' }}>
                  {preset.dimensions[aspectRatio] || preset.dimensions['1:1']}
                </span>
              </button>
            ))}
          </div>

          {/* Upscale Output */}
          <div style={{ marginTop: '10px', display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', fontSize: '0.85rem' }}>
              <input type="checkbox" checked={postUpscale} onChange={(e) => setPostUpscale(e.target.checked)} style={{ width: '16px', height: '16px' }} />
              <span>Upscale</span>
            </label>
            {postUpscale && (
              <>
                <div className="grok-toggle-group" style={{ width: 'auto' }}>
                  <button className={`grok-toggle-btn ${postUpscaleScale === 2 ? 'active' : ''}`} onClick={() => setPostUpscaleScale(2)} style={{ padding: '4px 10px', fontSize: '0.8rem' }}>2x</button>
                  <button className={`grok-toggle-btn ${postUpscaleScale === 4 ? 'active' : ''}`} onClick={() => setPostUpscaleScale(4)} style={{ padding: '4px 10px', fontSize: '0.8rem' }}>4x</button>
                </div>
                <span style={{ fontSize: '0.75rem', color: 'var(--accent-color)', fontWeight: 600 }}>
                  → {(() => {
                    const preset = RESOLUTION_PRESETS[resolution]
                    const dimStr = preset?.dimensions?.[aspectRatio] || preset?.dimensions?.['1:1'] || '480×848'
                    const [w, h] = dimStr.split('×').map(Number)
                    return `${w * postUpscaleScale}×${h * postUpscaleScale}`
                  })()}
                </span>
              </>
            )}
          </div>
        </div>

        {/* Aspect Ratio */}
        <div className="form-group">
          <label className="grok-section-label">Aspect Ratio</label>
          <div className="grok-toggle-group">
            {ASPECT_RATIOS.map((ratio) => (
              <button key={ratio} className={`grok-toggle-btn ${aspectRatio === ratio ? 'active' : ''}`} onClick={() => setAspectRatio(ratio)} type="button">
                {ratio}
              </button>
            ))}
          </div>
        </div>

        {/* Duration (seconds-based, aligned with I2V) */}
        <div className="form-group">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <label className="grok-section-label" style={{ marginBottom: 0 }}>Duration</label>
            <span className="nav-badge" style={{ fontSize: '0.8rem' }}>{duration}s ({duration * fps}f)</span>
          </div>
          <div style={{ position: 'relative', height: '24px', marginBottom: '8px' }}>
            <input
              type="range" min="3" max={maxDuration} step="1" value={duration}
              onChange={(e) => setDuration(parseInt(e.target.value, 10))}
              style={{ width: '100%', opacity: 0, position: 'absolute', zIndex: 2, cursor: 'pointer' }}
            />
            <div style={{ position: 'absolute', top: '10px', left: 0, right: 0, height: '4px', backgroundColor: '#333', borderRadius: '2px' }}>
              <div style={{
                width: `${((duration - 3) / (maxDuration - 3)) * 100}%`,
                height: '100%', backgroundColor: 'var(--accent-color, #a855f7)', borderRadius: '2px'
              }} />
            </div>
            <div style={{
              position: 'absolute', top: '2px',
              left: `calc(${((duration - 3) / (maxDuration - 3)) * 100}% - 10px)`,
              width: '20px', height: '20px', backgroundColor: 'white',
              borderRadius: '50%', boxShadow: '0 2px 4px rgba(0,0,0,0.3)'
            }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            <span>3s</span>
            <span>{Math.floor((3 + maxDuration) / 2)}s</span>
            <span>{maxDuration}s (max)</span>
          </div>
        </div>

        {/* FPS */}
        <div className="form-group">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <label className="grok-section-label" style={{ marginBottom: 0 }}>Frame Rate (FPS)</label>
            <span className="nav-badge" style={{ fontSize: '0.8rem' }}>{fps} fps</span>
          </div>
          <div className="grok-toggle-group">
            {FPS_OPTIONS.map((f) => (
              <button key={f} className={`grok-toggle-btn ${fps === f ? 'active' : ''}`} onClick={() => setFps(f)} type="button">
                {f}
              </button>
            ))}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '8px' }}>
            Higher FPS = smoother motion, more VRAM required
          </div>
        </div>

        {/* Extend Duration - Sequential Clips */}
        <div className="form-group" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div className="grok-section-label" style={{ marginBottom: '4px' }}>Extend Duration</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Chain multiple clips sequentially</div>
          </div>
          <label className="grok-switch">
            <input type="checkbox" checked={extendMode} onChange={(e) => { setExtendMode(e.target.checked); if (!e.target.checked) setClipCount(1) }} />
            <span className="grok-slider"></span>
          </label>
        </div>

        {/* Clip Count Slider */}
        {extendMode && (
          <div className="form-group" style={{
            background: 'linear-gradient(135deg, rgba(233, 69, 96, 0.1) 0%, rgba(233, 69, 96, 0.05) 100%)',
            borderRadius: '8px', padding: '12px', marginTop: '-8px',
            border: '1px solid rgba(233, 69, 96, 0.2)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <div className="grok-section-label">Number of Clips: {clipCount}</div>
              <div style={{
                fontSize: '0.75rem', color: '#e94560', background: 'rgba(233, 69, 96, 0.15)',
                padding: '2px 8px', borderRadius: '10px', fontWeight: '600'
              }}>
                ≈ {(duration * clipCount).toFixed(0)}s total
              </div>
            </div>
            <input type="range" min="1" max="5" value={clipCount} onChange={(e) => setClipCount(parseInt(e.target.value))} style={{ width: '100%', accentColor: '#e94560' }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              <span>1</span><span>2</span><span>3</span><span>4</span><span>5</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '8px', fontStyle: 'italic' }}>
              Each clip continues from the last frame of the previous clip
            </div>
          </div>
        )}
      </div>

      {/* Advanced Settings */}
      <div className="grok-card" style={{ padding: 0 }}>
        <div
          onClick={() => setShowAdvanced(!showAdvanced)}
          style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '16px 20px', cursor: 'pointer', color: 'var(--text-secondary)' }}
        >
          <Settings size={16} />
          <span style={{ fontSize: '0.85rem', fontWeight: 500 }}>Advanced Settings</span>
          <ChevronDown size={16} style={{ marginLeft: 'auto', transform: showAdvanced ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }} />
        </div>

        {showAdvanced && (
          <div style={{ padding: '0 20px 20px', borderTop: '1px solid var(--border-color)' }}>
            <div style={{ display: 'flex', gap: '16px', marginTop: '16px' }}>
              <div className="form-group" style={{ flex: 1 }}>
                <label className="grok-section-label">Video Steps</label>
                <input className="form-input" type="number" value={steps} onChange={(e) => setSteps(parseInt(e.target.value) || 6)} min="1" max="30" />
              </div>
              <div className="form-group" style={{ flex: 1 }}>
                <label className="grok-section-label">Seed</label>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <input className="form-input" type="number" value={seed} onChange={(e) => setSeed(parseInt(e.target.value) || -1)} placeholder="-1 for random" style={{ flex: 1 }} />
                  <button className="icon-btn" onClick={() => setSeed(-1)} style={{ whiteSpace: 'nowrap', width: 'auto', padding: '0 12px', fontSize: '0.8rem' }}>Random</button>
                </div>
              </div>
            </div>

            {/* T2I settings (only for Wan2.2 which does T2I→I2V) */}
            {modelType === 'wan22' && (
              <div style={{ display: 'flex', gap: '16px' }}>
                <div className="form-group" style={{ flex: 1 }}>
                  <label className="grok-section-label">T2I Steps</label>
                  <input className="form-input" type="number" value={t2iSteps} onChange={(e) => setT2iSteps(parseInt(e.target.value) || 20)} min="1" max="50" />
                </div>
                <div className="form-group" style={{ flex: 1 }}>
                  <label className="grok-section-label">T2I CFG</label>
                  <input className="form-input" type="number" value={t2iCfg} onChange={(e) => setT2iCfg(parseFloat(e.target.value) || 6.0)} min="1" max="20" step="0.5" />
                </div>
              </div>
            )}

            {/* LoRA Settings */}
            <div style={{ marginTop: '16px', paddingTop: '16px', borderTop: '1px solid var(--border-color)' }}>
              <div
                onClick={() => setShowLoraPanel(!showLoraPanel)}
                style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer', marginBottom: showLoraPanel ? '12px' : 0 }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Layers size={16} />
                  <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>LoRA Models</span>
                  {loraConfigs.length > 0 && (
                    <span style={{ fontSize: '0.7rem', backgroundColor: 'var(--accent-color)', color: 'white', padding: '2px 6px', borderRadius: '4px' }}>
                      {loraConfigs.length} active
                    </span>
                  )}
                </div>
                <span style={{ opacity: 0.5, fontSize: '0.8rem' }}>{showLoraPanel ? '▼' : '▶'}</span>
              </div>

              {showLoraPanel && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {loraConfigs.map((config, idx) => (
                    <div key={idx} style={{
                      backgroundColor: 'var(--bg-input)', borderRadius: '8px', padding: '12px', border: '1px solid var(--border-color)'
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                        <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>LoRA #{idx + 1}</span>
                        <button onClick={() => setLoraConfigs(loraConfigs.filter((_, i) => i !== idx))} style={{ background: 'transparent', border: 'none', color: '#ef4444', cursor: 'pointer', padding: '2px 6px', fontSize: '0.8rem' }}>
                          Remove
                        </button>
                      </div>
                      {/* High Noise LoRA */}
                      <div style={{ marginBottom: '8px' }}>
                        <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
                          High Noise (steps 0-3)
                        </label>
                        <select
                          value={config.high || ''}
                          onChange={(e) => { const nc = [...loraConfigs]; nc[idx] = { ...config, high: e.target.value }; setLoraConfigs(nc) }}
                          style={{ width: '100%', padding: '6px 10px', backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: '4px', color: 'var(--text-primary)', fontSize: '0.8rem' }}
                        >
                          <option value="">None</option>
                          {filteredLoras.by_category && Object.keys(filteredLoras.by_category).sort().map((category) => (
                            <optgroup key={category} label={category === 'root' ? 'Root' : category}>
                              {filteredLoras.by_category[category].map((lora) => (
                                <option key={lora.path} value={lora.path}>{lora.name} ({lora.size_mb}MB)</option>
                              ))}
                            </optgroup>
                          ))}
                        </select>
                      </div>
                      {/* Low Noise LoRA */}
                      <div style={{ marginBottom: '8px' }}>
                        <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
                          Low Noise (steps 3+)
                        </label>
                        <select
                          value={config.low || ''}
                          onChange={(e) => { const nc = [...loraConfigs]; nc[idx] = { ...config, low: e.target.value }; setLoraConfigs(nc) }}
                          style={{ width: '100%', padding: '6px 10px', backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: '4px', color: 'var(--text-primary)', fontSize: '0.8rem' }}
                        >
                          <option value="">None (uses High Noise)</option>
                          {filteredLoras.by_category && Object.keys(filteredLoras.by_category).sort().map((category) => (
                            <optgroup key={category} label={category === 'root' ? 'Root' : category}>
                              {filteredLoras.by_category[category].map((lora) => (
                                <option key={lora.path} value={lora.path}>{lora.name} ({lora.size_mb}MB)</option>
                              ))}
                            </optgroup>
                          ))}
                        </select>
                      </div>
                      {/* Strength slider */}
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
                          <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Strength</label>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{(config.strength || 1.0).toFixed(2)}</span>
                        </div>
                        <input
                          type="range" min="0" max="2" step="0.05" value={config.strength || 1.0}
                          onChange={(e) => { const nc = [...loraConfigs]; nc[idx] = { ...config, strength: parseFloat(e.target.value) }; setLoraConfigs(nc) }}
                          style={{ width: '100%', cursor: 'pointer' }}
                        />
                      </div>
                    </div>
                  ))}

                  <button
                    onClick={() => setLoraConfigs([...loraConfigs, { high: '', low: '', strength: 1.0 }])}
                    style={{
                      padding: '8px 12px', backgroundColor: 'transparent', border: '1px dashed var(--border-color)',
                      borderRadius: '6px', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '0.85rem',
                      display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px'
                    }}
                  >
                    + Add LoRA
                  </button>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                    Stack multiple LoRAs for combined effects. Each LoRA has its own strength.
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Post-Processing Options */}
      <div className="grok-card" style={{ padding: 0 }}>
        <div
          onClick={() => setShowPostProcessing(!showPostProcessing)}
          style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '16px 20px', cursor: 'pointer', color: 'var(--text-secondary)' }}
        >
          <Zap size={16} />
          <span style={{ fontSize: '0.85rem', fontWeight: 500 }}>Post-Processing</span>
          {(postInterpolate || postAudio) && (
            <span style={{ fontSize: '0.7rem', backgroundColor: 'var(--success-color)', color: 'white', padding: '2px 6px', borderRadius: '4px' }}>
              {[postInterpolate && 'RIFE', postAudio && 'Audio'].filter(Boolean).join(' + ')}
            </span>
          )}
          <ChevronDown size={16} style={{ marginLeft: 'auto', transform: showPostProcessing ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }} />
        </div>

        {showPostProcessing && (
          <div style={{ padding: '0 20px 20px', borderTop: '1px solid var(--border-color)' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '12px' }}>
              {/* Frame Interpolation option */}
              <div style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '8px 12px',
                backgroundColor: postInterpolate ? 'rgba(var(--success-rgb), 0.1)' : 'var(--bg-secondary)',
                borderRadius: '8px',
                border: postInterpolate ? '1px solid var(--success-color)' : '1px solid var(--border-color)'
              }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', flex: 1 }}>
                  <input type="checkbox" checked={postInterpolate} onChange={(e) => setPostInterpolate(e.target.checked)} style={{ width: '16px', height: '16px' }} />
                  <span>Smooth Motion (RIFE)</span>
                </label>
                {postInterpolate && (
                  <select
                    value={postInterpolateFps}
                    onChange={(e) => setPostInterpolateFps(parseInt(e.target.value))}
                    style={{
                      padding: '4px 8px', backgroundColor: 'var(--bg-tertiary)',
                      border: '1px solid var(--border-color)', borderRadius: '4px', color: 'var(--text-primary)'
                    }}
                  >
                    <option value={30}>30 fps</option>
                    <option value={60}>60 fps</option>
                  </select>
                )}
              </div>

              {/* Add Audio option */}
              <div style={{
                display: 'flex', flexDirection: 'column', gap: '8px', padding: '8px 12px',
                backgroundColor: postAudio ? 'rgba(var(--success-rgb), 0.1)' : 'var(--bg-secondary)',
                borderRadius: '8px',
                border: postAudio ? '1px solid var(--success-color)' : '1px solid var(--border-color)'
              }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                  <input type="checkbox" checked={postAudio} onChange={(e) => { setPostAudio(e.target.checked); if (!e.target.checked) setPostAudioFile(null) }} style={{ width: '16px', height: '16px' }} />
                  <span>Add Audio Track</span>
                </label>
                {postAudio && (
                  <input type="file" accept="audio/*" onChange={(e) => setPostAudioFile(e.target.files?.[0] || null)} style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }} />
                )}
              </div>

              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                Post-processing runs as chained jobs after video generation completes
              </div>
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
          Job queued! Check the Queue panel for progress.
        </div>
      )}

      {error && (
        <div style={{
          padding: '12px', backgroundColor: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '8px',
          color: '#ef4444', marginBottom: '12px', fontSize: '0.85rem',
        }}>
          {error}
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
        style={{
          height: '48px', fontSize: '1rem',
          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
          backgroundColor: '#e5e5e5', color: 'black'
        }}
      >
        {submitting ? (
          <>
            <Loader2 size={18} className="animate-spin" />
            Queueing...
          </>
        ) : (
          <>
            <Sparkles size={18} />
            Generate Video
          </>
        )}
      </button>

      <div className="info-badge" style={{ marginTop: '12px', textAlign: 'center' }}>
        {modelType === 'ltx2'
          ? 'LTX-2 generates video directly from text (faster)'
          : modelType === 'cloud_max'
            ? 'Cloud Max uses full bf16 precision on cloud GPU (highest quality)'
            : 'Wan2.2 first generates an image, then animates it (higher quality)'}
      </div>
    </div>
  )
}
