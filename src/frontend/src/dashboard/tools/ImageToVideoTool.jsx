import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Upload, X, Film, Type, Settings2, Image as ImageIcon, Link, FolderOpen, Sparkles, Info, ChevronDown, Layers, FileSearch, Sliders, Clock, HelpCircle, Wand2, Loader2, Save, Check, Grid, Trash2, Pencil } from 'lucide-react'
import { BACKEND_BASE, DEBUG, getMediaUrl } from '../../config'
import { postForm, uploadUserMedia, apiFetch, getUserMediaUrl } from '../../api'
import { sendClientLog } from '../../logging'
import { useNSFW } from '../../contexts/NSFWContext'
import { useAuth } from '../../contexts/AuthContext'
import { getDefaultPrompt, getRandomPrompt } from '../../data/defaultPrompts'
import { estimateI2VTime } from '../../utils/timeEstimates'
import PresetSelector from '../../components/PresetSelector'
import CameraMotionSelector, { getCameraMotionPrefix } from '../../components/CameraMotionSelector'
import MediaImportModal from '../../components/MediaImportModal'
import { parseComfyWorkflow } from '../../utils/parseComfyMetadata'
import { useToolProfile } from '../../hooks/useToolProfile'
import useLLMEnhance from '../../hooks/useLLMEnhance'
import LLMQueueIndicator from '../../components/LLMQueueIndicator'
import '../../components/PresetSelector.css'

const FPS_OPTIONS = [8, 12, 16, 24]

// Model mode options for I2V
const MODEL_MODES = [
  { value: 'wan2.2', label: '🎬 Wan2.2 14B Q6 DisTorch2', desc: 'High quality dual-pass via ComfyUI' },
  { value: 'blockswap_q8', label: '🧪 BlockSwap Q8 Experimental', desc: 'Q8 quality • Single-GPU BlockSwap • Lightning LoRA + NAG + TorchCompile' },
  { value: 'distorch2_q8', label: '🧪 DisTorch2 Q8 Experimental', desc: 'Q8 quality + DisTorch2 Multi-GPU + Selectable LoRAs' },
  { value: 'ltx2', label: '⚡ LTX-2 19B Distilled', desc: 'Fast single-pass, lower VRAM' },
]

// Resolution presets with dimensions per aspect ratio
// Includes max_duration based on tested VRAM limits (28GB dual GPU)
const RESOLUTION_PRESETS = {
  '480p': {
    label: '480p',
    dimensions: {
      '16:9': '848×480',
      '9:16': '480×848',
      '1:1': '480×480',
      '4:3': '640×480',
      '3:4': '480×640',
    },
    // Max 30s for all models (VRAM will be the practical limit)
    max_duration_wan22: 30,
    max_duration_ltx2: 30,
    max_duration_blockswap_q8: 30,
    max_duration_distorch2_q8: 30,
  },
  '576p': {
    label: '576p',
    dimensions: {
      '16:9': '1024×576',
      '9:16': '576×1024',
      '1:1': '576×576',
      '4:3': '768×576',
      '3:4': '576×768',
    },
    // Max 30s for all models (VRAM will be the practical limit)
    max_duration_wan22: 30,
    max_duration_ltx2: 30,
    max_duration_blockswap_q8: 30,
    max_duration_distorch2_q8: 30,
  },
  '720p': {
    label: '720p',
    dimensions: {
      '16:9': '1280×720',
      '9:16': '720×1280',
      '1:1': '720×720',
      '4:3': '960×720',
      '3:4': '720×960',
    },
    // Max 30s for all models (VRAM will be the practical limit)
    max_duration_wan22: 30,
    max_duration_ltx2: 30,
    max_duration_blockswap_q8: 30,
    max_duration_distorch2_q8: 30,
  },
}

// Aspect ratio options
const ASPECT_RATIOS = ['16:9', '9:16', '1:1', '4:3', '3:4']

// Default settings for profile persistence
const I2V_DEFAULT_SETTINGS = {
  prompt: '',
  negativePrompt: 'low quality, blurry, out of focus, unstable camera, artifacts, distortion, low resolution, overexposed, underexposed, color banding, missing details, unrealistic lighting, flickering shadows, frame stutter, ghosting, bad reflections, unrealistic motion, pixelated textures, wrong physics, broken animation, rendering artifacts, compression noise, jitter, unnatural sand behavior, visual glitches',
  duration: 8,
  resolution: '480p',
  modelMode: 'wan2.2',
  modelVersion: 'v2',
  aspectRatio: '9:16',
  fps: 16,
  steps: 6,
  cfg: 3.0,
  seed: -1,
  cameraMotion: '',
  bsShift: 9.0,
  bsNagScale: 11.0,
  bsEnableFlorence2: true,
  bsEnableUpscale: false,
  bsEnableInterpolation: false,
  bsHighNoiseSteps: 4,
  loraConfigs: [],
  unetHighNoise: 'wan2.2_i2v_high_noise_14B_Q6_K.gguf',
  unetLowNoise: 'wan2.2_i2v_low_noise_14B_Q6_K.gguf',
  extendMode: false,
  clipCount: 1,
  postUpscale: false,
  postUpscaleScale: 2,
  postInterpolate: false,
  postInterpolateFps: 60,
  enhanceModel: 'GLM-4.7-Flash-Claude-Opus-Reasoning',
  sourceImageName: null,
}

export default function ImageToVideoTool({ onOutput, onRefreshHistory, onCreationsModeChange, onParamsChange, onJobSubmitted, pendingImport = null, onImportConsumed = null }) {
  const { nsfwEnabled } = useNSFW()
  const { user, requestLogin } = useAuth()
  const fileInputRef = useRef(null)
  const pendingImageRestore = useRef(null)

  // ── Profile persistence (auto-save on every change) ──────────────────
  const applyProfileSettings = useCallback((s) => {
    if (s.prompt) setPrompt(s.prompt)
    if (s.negativePrompt !== undefined) setNegativePrompt(s.negativePrompt)
    if (s.duration !== undefined) setDuration(s.duration)
    if (s.resolution) setResolution(s.resolution)
    if (s.modelMode) setModelMode(s.modelMode)
    if (s.modelVersion) setModelVersion(s.modelVersion)
    if (s.aspectRatio) setAspectRatio(s.aspectRatio)
    if (s.fps !== undefined) setFps(s.fps)
    if (s.steps !== undefined) setSteps(s.steps)
    if (s.cfg !== undefined) setCfg(s.cfg)
    if (s.seed !== undefined) setSeed(s.seed)
    if (s.cameraMotion !== undefined) setCameraMotion(s.cameraMotion)
    if (s.bsShift !== undefined) setBsShift(s.bsShift)
    if (s.bsNagScale !== undefined) setBsNagScale(s.bsNagScale)
    if (s.bsEnableFlorence2 !== undefined) setBsEnableFlorence2(s.bsEnableFlorence2)
    if (s.bsEnableUpscale !== undefined) setBsEnableUpscale(s.bsEnableUpscale)
    if (s.bsEnableInterpolation !== undefined) setBsEnableInterpolation(s.bsEnableInterpolation)
    if (s.bsHighNoiseSteps !== undefined) setBsHighNoiseSteps(s.bsHighNoiseSteps)
    if (s.loraConfigs !== undefined) setLoraConfigs(s.loraConfigs)
    if (s.unetHighNoise) setUnetHighNoise(s.unetHighNoise)
    if (s.unetLowNoise) setUnetLowNoise(s.unetLowNoise)
    if (s.extendMode !== undefined) setExtendMode(s.extendMode)
    if (s.clipCount !== undefined) setClipCount(s.clipCount)
    if (s.postUpscale !== undefined) setPostUpscale(s.postUpscale)
    if (s.postUpscaleScale !== undefined) setPostUpscaleScale(s.postUpscaleScale)
    if (s.postInterpolate !== undefined) setPostInterpolate(s.postInterpolate)
    if (s.postInterpolateFps !== undefined) setPostInterpolateFps(s.postInterpolateFps)
    if (s.enhanceModel) setEnhanceModel(s.enhanceModel)
    // Queue source image restore (handled by useEffect below)
    if (s.sourceImageName) {
      pendingImageRestore.current = s.sourceImageName
    }
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
  } = useToolProfile('image_to_video', I2V_DEFAULT_SETTINGS, {
    onLoad: applyProfileSettings,
  })
  const [showProfileMenu, setShowProfileMenu] = useState(false)
  const [profileSaveInput, setProfileSaveInput] = useState('')

  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [uploadTab, setUploadTab] = useState('file') // 'file', 'url', 'creations', 'library'
  const [restoringImage, setRestoringImage] = useState(false)
  const [userUploads, setUserUploads] = useState([])
  const [uploadsLoading, setUploadsLoading] = useState(false)

  const [prompt, setPrompt] = useState(() => {
    // Load saved prompt or generate a random default for new users
    return getDefaultPrompt(false) // nsfwEnabled starts false
  })
  const [negativePrompt, setNegativePrompt] = useState('low quality, blurry, out of focus, unstable camera, artifacts, distortion, low resolution, overexposed, underexposed, color banding, missing details, unrealistic lighting, flickering shadows, frame stutter, ghosting, bad reflections, unrealistic motion, pixelated textures, wrong physics, broken animation, rendering artifacts, compression noise, jitter, unnatural sand behavior, visual glitches')
  const [showNegativePrompt, setShowNegativePrompt] = useState(false)
  const [showPromptTips, setShowPromptTips] = useState(false)
  const [duration, setDuration] = useState(8) // seconds, 3-15 range
  const [resolution, setResolution] = useState('480p')
  const [modelMode, setModelMode] = useState('wan2.2')  // default to Wan2.2 for quality
  const [modelVersion, setModelVersion] = useState('v2')
  const [usePose, setUsePose] = useState(false)
  const [aspectRatio, setAspectRatio] = useState('9:16')
  const [fps, setFps] = useState(16)
  const [steps, setSteps] = useState(6)
  const [cfg, setCfg] = useState(3.0)  // Default to balanced prompt strength
  const [seed, setSeed] = useState(-1)
  const [showAdvanced, setShowAdvanced] = useState(false)  // Sampling settings collapsed by default

  // BlockSwap Q8 Experimental mode settings
  const [bsShift, setBsShift] = useState(9.0)
  const [bsNagScale, setBsNagScale] = useState(11.0)
  const [bsEnableFlorence2, setBsEnableFlorence2] = useState(true)
  const [bsEnableUpscale, setBsEnableUpscale] = useState(false)
  const [bsEnableInterpolation, setBsEnableInterpolation] = useState(false)
  const [bsHighNoiseSteps, setBsHighNoiseSteps] = useState(4)

  // Camera motion preset
  const [cameraMotion, setCameraMotion] = useState('')
  const [isEnhancing, setIsEnhancing] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isRefining, setIsRefining] = useState(false)
  const [showRefineInput, setShowRefineInput] = useState(false)
  const [refineInstruction, setRefineInstruction] = useState('')
  const [enhanceModel, setEnhanceModel] = useState('GLM-4.7-Flash-Claude-Opus-Reasoning')

  // Pending import modal state
  const [importModal, setImportModal] = useState(null)  // { item, workflow }

  // When Dashboard sends a new pendingImport (from MyMedia "Use in tool")
  useEffect(() => {
    if (!pendingImport) return
    setImportModal(pendingImport)
    if (onImportConsumed) onImportConsumed()
  }, [pendingImport])

  const handleApplyImport = (selected) => {
    if (selected.image && importModal?.item) {
      // The item reference is in importModal; Dashboard already navigated here,
      // image loading happens via selectCreation below or caller side.
      // If we have the item, load it as input image via fetch.
      const item = importModal.item

      // If item is a video, use the corresponding .png thumbnail instead
      let imageUrl, imageFilename
      if (item.type === 'video' && item.filename?.match(/\.(mp4|webm|mov)$/i)) {
        const pngFilename = item.filename.replace(/\.(mp4|webm|mov)$/i, '.png')
        // Use relative path — Vite proxy handles /comfyui-output/ in dev,
        // avoids CORS issues with StaticFiles mount not having CORS headers
        const pngUrl = item.url?.replace(/\.(mp4|webm|mov)$/i, '.png')
        imageUrl = pngUrl  // relative path, proxied by Vite
        imageFilename = pngFilename
        console.debug('🎬 Use in tool: video detected, using companion image:', pngFilename)
      } else {
        imageUrl = getMediaUrl(item.url, item.signed_url)
        imageFilename = item.filename || imageUrl.split('/').pop()
      }

      apiFetch(imageUrl)
        .then(r => {
          if (!r.ok) throw new Error(`Failed to fetch image: ${r.status}`)
          return r.blob()
        })
        .then(blob => {
          const filename = imageFilename || imageUrl.split('/').pop()
          const fileObj = new File([blob], filename, { type: blob.type || 'image/png' })
          setFile(fileObj)
          setPreviewUrl(imageUrl)
          setUploadTab('file')
        })
        .catch((err) => {
          console.warn('⚠️ Use in tool: failed to load image', err)
        })
    }
    // Coerce all values to proper types — metadata may have numbers instead of strings
    if (selected.positive)  setPrompt(String(selected.positive))
    if (selected.negative)  setNegativePrompt(String(selected.negative))
    if (selected.steps)     setSteps(Number(selected.steps) || selected.steps)
    if (selected.cfg)       setCfg(Number(selected.cfg) || selected.cfg)
    if (selected.seed)      setSeed(String(selected.seed))
    if (selected.loras && Array.isArray(selected.loras)) {
      setLoraConfigs(selected.loras.map(l => ({
        high: l.high || '',
        low: l.low || '',
        strength: l.strength ?? 1.0,
      })))
    }
    setImportModal(null)
  }
  const [imageDescription, setImageDescription] = useState('')  // Store vision analysis result

  // LoRA state - now supports multiple LoRAs with individual strengths
  const [availableLoras, setAvailableLoras] = useState({ high_noise: [], low_noise: [], general: [] })
  // Array of {high: string, low: string, strength: number}
  const [loraConfigs, setLoraConfigs] = useState([])
  const [showLoraPanel, setShowLoraPanel] = useState(false)
  const [loraSearchHigh, setLoraSearchHigh] = useState({})  // {idx: string} per LoRA slot
  const [loraSearchLow, setLoraSearchLow] = useState({})    // {idx: string} per LoRA slot
  const [loraDropdownOpen, setLoraDropdownOpen] = useState(null)  // 'high-0', 'low-2', etc.

  // Unet model state
  const [availableUnets, setAvailableUnets] = useState({ high_noise: [], low_noise: [], pairs: [] })
  const [unetHighNoise, setUnetHighNoise] = useState('wan2.2_i2v_high_noise_14B_Q6_K.gguf')
  const [unetLowNoise, setUnetLowNoise] = useState('wan2.2_i2v_low_noise_14B_Q6_K.gguf')
  const [showUnetPanel, setShowUnetPanel] = useState(false)

  // Extend Duration - Sequential clip generation
  const [extendMode, setExtendMode] = useState(false)
  const [clipCount, setClipCount] = useState(1)

  // Post-processing options (chained jobs)
  const [postUpscale, setPostUpscale] = useState(false)
  const [postUpscaleScale, setPostUpscaleScale] = useState(2)
  const [postInterpolate, setPostInterpolate] = useState(false)
  const [postInterpolateFps, setPostInterpolateFps] = useState(60)
  const [postAudio, setPostAudio] = useState(false)
  const [postAudioFile, setPostAudioFile] = useState(null)
  const [showPostProcessing, setShowPostProcessing] = useState(false)

  // Preset mode
  const [usePresets, setUsePresets] = useState(false)
  const [selectedPreset, setSelectedPreset] = useState(null)
  const [presetParameters, setPresetParameters] = useState({})

  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  // Selected creation from MyMediaTool picker
  const [selectedCreation, setSelectedCreation] = useState(null)

  const canSubmit = useMemo(() => !!file && !busy, [file, busy])

  // ── Restore source image from profile on load ─────────────────────────
  useEffect(() => {
    if (!pendingImageRestore.current || !profileLoaded) return
    const imageName = pendingImageRestore.current
    pendingImageRestore.current = null

    async function restore() {
      setRestoringImage(true)
      try {
        const resp = await apiFetch(`/user/media/uploads/${encodeURIComponent(imageName)}`)
        if (!resp.ok) throw new Error('Image not found in storage')
        const blob = await resp.blob()
        const fileObj = new File([blob], imageName, { type: blob.type || 'image/png' })
        setFile(fileObj)
        setPreviewUrl(URL.createObjectURL(blob))
        if (DEBUG) console.debug('📁 Restored source image from profile:', imageName)
      } catch (err) {
        if (DEBUG) console.debug('📁 Failed to restore source image:', imageName, err.message)
        // Non-fatal: profile loads without image, user can re-upload
      } finally {
        setRestoringImage(false)
      }
    }
    restore()
  }, [profileLoaded])

  // ── Auto-save settings to profile on every change ─────────────────────
  const settingsSnapshot = useMemo(() => ({
    prompt, negativePrompt, duration, resolution, modelMode, modelVersion,
    aspectRatio, fps, steps, cfg, seed, cameraMotion,
    bsShift, bsNagScale, bsEnableFlorence2, bsEnableUpscale,
    bsEnableInterpolation, bsHighNoiseSteps,
    loraConfigs, unetHighNoise, unetLowNoise,
    extendMode, clipCount,
    postUpscale, postUpscaleScale, postInterpolate, postInterpolateFps,
    enhanceModel,
    sourceImageName: file?.name || null,
  }), [
    prompt, negativePrompt, duration, resolution, modelMode, modelVersion,
    aspectRatio, fps, steps, cfg, seed, cameraMotion,
    bsShift, bsNagScale, bsEnableFlorence2, bsEnableUpscale,
    bsEnableInterpolation, bsHighNoiseSteps,
    loraConfigs, unetHighNoise, unetLowNoise,
    extendMode, clipCount,
    postUpscale, postUpscaleScale, postInterpolate, postInterpolateFps,
    enhanceModel, file,
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
      setPrompt(result.prompt)
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
      setPrompt(result.prompt)
      if (result.negative_prompt) setNegativePrompt(result.negative_prompt)
      setRefineInstruction('')
      setShowRefineInput(false)
    } else if (llm.error) {
      setError(`Refine failed: ${llm.error}`)
    }
    setIsRefining(false)
  }

  // Analyze image with vision model and generate creative video prompts
  const handleAnalyzeAndGenerate = async (useNsfw = false) => {
    if (!previewUrl || isAnalyzing) return
    setIsAnalyzing(true)
    setError('')

    try {
      // Get image as base64
      let imageBase64 = ''

      // If previewUrl is a blob or data URL, fetch it
      if (previewUrl.startsWith('blob:') || previewUrl.startsWith('data:')) {
        const response = await fetch(previewUrl)
        const blob = await response.blob()
        imageBase64 = await new Promise((resolve) => {
          const reader = new FileReader()
          reader.onloadend = () => resolve(reader.result)
          reader.readAsDataURL(blob)
        })
      } else {
        // For remote URLs, fetch through backend or directly
        const response = await fetch(previewUrl)
        const blob = await response.blob()
        imageBase64 = await new Promise((resolve) => {
          const reader = new FileReader()
          reader.onloadend = () => resolve(reader.result)
          reader.readAsDataURL(blob)
        })
      }

      if (DEBUG) console.log('🔮 Analyzing image and generating prompts...')

      const res = await fetch(`${BACKEND_BASE}/api/analyze-and-generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: imageBase64,
          nsfw: useNsfw,
        }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Analysis failed')
      }

      const data = await res.json()
      if (DEBUG) console.log('🎬 Generated prompts:', data)

      // Store the image description for reference
      setImageDescription(data.description)

      // Update prompts
      setPrompt(data.prompt)
      if (data.negative_prompt) {
        setNegativePrompt(data.negative_prompt)
      }

    } catch (err) {
      console.error('Analyze error:', err)
      setError(`Analysis failed: ${err.message}`)
    } finally {
      setIsAnalyzing(false)
    }
  }

  // Calculate max duration based on resolution and model mode
  const maxDuration = useMemo(() => {
    const preset = RESOLUTION_PRESETS[resolution]
    if (!preset) return 30
    if (modelMode === 'ltx2') {
      return preset.max_duration_ltx2 || 30
    }
    if (modelMode === 'blockswap_q8') {
      return preset.max_duration_blockswap_q8 || 30
    }
    if (modelMode === 'distorch2_q8') {
      return preset.max_duration_distorch2_q8 || 30
    }
    return preset.max_duration_wan22 || 30
  }, [resolution, modelMode])

  // Clamp duration when max changes
  useEffect(() => {
    if (duration > maxDuration) {
      setDuration(maxDuration)
    }
  }, [maxDuration, duration])

  // Calculate estimated generation time
  const timeEstimate = useMemo(() => {
    return estimateI2VTime({ resolution, duration, steps })
  }, [resolution, duration, steps])

  // Fetch available LoRAs on mount
  useEffect(() => {
    const fetchLoras = async () => {
      try {
        const res = await fetch(`${BACKEND_BASE}/loras`)
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

  // Filter LoRAs based on NSFW setting
  const filteredLoras = useMemo(() => {
    if (nsfwEnabled) return availableLoras

    // Filter each category
    const filterList = (list) => (list || []).filter(l => !l.nsfw)

    // Filter by_category object
    const filteredByCategory = {}
    if (availableLoras.by_category) {
      Object.keys(availableLoras.by_category).forEach(cat => {
        const filtered = filterList(availableLoras.by_category[cat])
        if (filtered.length > 0) {
          filteredByCategory[cat] = filtered
        }
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
          if (DEBUG) console.debug('🐛 loaded Unet models:', data.count)
        }
      } catch (e) {
        console.error('Failed to fetch Unet models:', e)
      }
    }
    fetchUnets()
  }, [])

  // Persist prompt to localStorage on change
  useEffect(() => {
    if (prompt) {
      try {
        localStorage.setItem('oelala_last_prompt', prompt)
      } catch { /* ignore storage errors */ }
    }
  }, [prompt])

  // Expose current params to parent for JSON download
  useEffect(() => {
    if (onParamsChange) {
      onParamsChange({
        tool: 'ImageToVideo',
        prompt,
        duration,
        resolution,
        modelMode,
        modelVersion,
        aspectRatio,
        fps,
        steps,
        cfg,
        seed,
        usePose,
        loraConfigs,
        filename: file?.name || null,
      })
    }
  }, [prompt, duration, resolution, modelMode, modelVersion, aspectRatio, fps, steps, cfg, seed, usePose, loraConfigs, file, onParamsChange])

  // Select an image from My Creations (called by MyMediaTool in output panel)
  const selectCreation = useCallback(async (item) => {
    setSelectedCreation(item)
    setError('')

    try {
      // Fetch the image and convert to File object
      // Use getMediaUrl to handle both signed URLs and relative paths
      const imageUrl = getMediaUrl(item.url, item.signed_url)
      const response = await apiFetch(imageUrl)
      const blob = await response.blob()
      const filename = item.filename || item.url.split('/').pop()
      const fileObj = new File([blob], filename, { type: blob.type || 'image/png' })

      setFile(fileObj)
      setPreviewUrl(imageUrl)
      setUploadTab('file') // Switch back to file tab to show the selection

      // Show in output panel
      onOutput({
        kind: 'image',
        url: imageUrl,
        backendUrl: imageUrl,
        filename: filename,
        meta: { source: 'my-creations', originalItem: item },
      })

      if (DEBUG) console.debug('🐛 selected creation:', filename)

      // Fetch metadata and show import modal so user can optionally reuse prompts
      try {
        const metaRes = await fetch(`${BACKEND_BASE}/comfyui-metadata/${filename}`)
        if (metaRes.ok) {
          const metaJson = await metaRes.json()
          const workflowData = parseComfyWorkflow(metaJson.metadata || {})
          // Only show modal if there's something useful to import
          if (workflowData.positive || workflowData.steps || workflowData.loras?.length) {
            setImportModal({ item, workflow: workflowData })
          }
        }
      } catch (_) { /* no metadata — silently skip */ }
    } catch (e) {
      setError('Failed to load selected image')
      console.error('Error selecting creation:', e)
    }
  }, [onOutput])

  // Notify Dashboard when creations tab is active/inactive
  useEffect(() => {
    if (onCreationsModeChange) {
      onCreationsModeChange(uploadTab === 'creations' && !file, selectCreation)
    }
    // Cleanup: disable creations mode when component unmounts
    return () => {
      if (onCreationsModeChange) {
        onCreationsModeChange(false, null)
      }
    }
  }, [uploadTab, file, onCreationsModeChange, selectCreation])

  const onPickFile = async (picked) => {
    if (!picked) return
    setFile(picked)
    setError('')
    setSelectedCreation(null) // Clear selection when manually picking

    const url = URL.createObjectURL(picked)
    setPreviewUrl(url)

    // Try to extract metadata from uploaded file (T2I/I2V generated images have embedded prompts)
    try {
      const formData = new FormData()
      formData.append('file', picked)
      const res = await fetch(`${BACKEND_BASE}/extract-metadata`, { method: 'POST', body: formData })
      if (res.ok) {
        const meta = await res.json()
        if (DEBUG) console.debug('🐛 extracted metadata:', meta)
        // Auto-fill prompts from metadata if found
        if (meta.prompt && !prompt) {
          setPrompt(meta.prompt)
        }
        if (meta.negative_prompt && negativePrompt === 'low quality, blurry, out of focus, unstable camera, artifacts, distortion, low resolution, overexposed, underexposed, color banding, missing details, unrealistic lighting, flickering shadows, frame stutter, ghosting, bad reflections, unrealistic motion, pixelated textures, wrong physics, broken animation, rendering artifacts, compression noise, jitter, unnatural sand behavior, visual glitches') {
          setNegativePrompt(meta.negative_prompt)
        }
      }
    } catch (e) {
      // Metadata extraction is optional, don't fail the upload
      if (DEBUG) console.debug('🐛 no metadata extracted:', e.message)
    }

    // Sidecar: persist to user's upload library (fire-and-forget)
    if (user && picked) {
      uploadUserMedia('uploads', picked).then(() => {
        if (DEBUG) console.debug('📁 Image persisted to upload library:', picked.name)
        // Refresh library if it's loaded
        loadUserUploads()
      }).catch(err => {
        if (DEBUG) console.debug('📁 Sidecar upload failed (non-blocking):', err.message)
      })
    }
  }

  // ── User Uploads Library ────────────────────────────────────────────
  const loadUserUploads = useCallback(async () => {
    if (!user) return
    setUploadsLoading(true)
    try {
      const resp = await apiFetch('/user/media?type=uploads')
      if (resp.ok) {
        const data = await resp.json()
        const items = data.media || []
        // Pre-fetch authenticated thumbnails as blob URLs
        const enriched = await Promise.all(items.map(async (item) => {
          try {
            const imgResp = await apiFetch(item.url)
            if (imgResp.ok) {
              const blob = await imgResp.blob()
              return { ...item, blobUrl: URL.createObjectURL(blob) }
            }
          } catch { /* non-blocking */ }
          return item
        }))
        setUserUploads(enriched)
      }
    } catch (err) {
      if (DEBUG) console.debug('📁 Failed to load uploads library:', err.message)
    } finally {
      setUploadsLoading(false)
    }
  }, [user])

  // Load library when tab is selected
  useEffect(() => {
    if (uploadTab === 'library' && user && userUploads.length === 0) {
      loadUserUploads()
    }
  }, [uploadTab, user, userUploads.length, loadUserUploads])

  const selectFromLibrary = useCallback(async (item) => {
    setError('')
    try {
      const imageUrl = getMediaUrl(item.url, item.signed_url)
      const response = await apiFetch(imageUrl)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const blob = await response.blob()
      const filename = item.name || item.url.split('/').pop()
      const fileObj = new File([blob], filename, { type: blob.type || 'image/png' })

      setFile(fileObj)
      setPreviewUrl(URL.createObjectURL(blob))
      setUploadTab('file')
      setSelectedCreation(null)

      if (DEBUG) console.debug('📁 Selected from library:', filename)
    } catch (err) {
      setError('Failed to load image from library')
      if (DEBUG) console.debug('📁 Library select failed:', err.message)
    }
  }, [])

  const deleteFromLibrary = useCallback(async (item, e) => {
    e.stopPropagation()
    if (!user) return
    try {
      const filename = item.name || item.url.split('/').pop()
      await apiFetch(`/user/media/uploads/${encodeURIComponent(filename)}`, { method: 'DELETE' })
      setUserUploads(prev => prev.filter(u => u.name !== item.name))
      if (DEBUG) console.debug('📁 Deleted from library:', filename)
    } catch (err) {
      if (DEBUG) console.debug('📁 Delete failed:', err.message)
    }
  }, [user])

  const clearFile = () => {
    setFile(null)
    setPreviewUrl('')
    setSelectedCreation(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const handleSubmit = async () => {
    // Check if user is logged in
    if (!user) {
      requestLogin('Log in om video\'s te genereren')
      return
    }

    if (!file) {
      setError('Image is required')
      return
    }

    setBusy(true)
    setError('')

    const numFrames = duration * fps
    const formData = new FormData()
    formData.append('file', file)
    formData.append('num_frames', String(numFrames))
    formData.append('resolution', resolution)
    formData.append('fps', String(fps))
    formData.append('aspect_ratio', aspectRatio)

    // Build prompt with camera motion prefix
    if (!usePose) {
      const motionPrefix = getCameraMotionPrefix(cameraMotion)
      const finalPrompt = motionPrefix + (prompt || 'Motion, subject moving naturally')
      formData.append('prompt', finalPrompt)
    }

    // Choose endpoint
    let endpoint
    let useAsync = true  // Default to async mode for non-blocking generation

    if (usePose) {
      endpoint = `${BACKEND_BASE}/generate-pose`
      useAsync = false  // Pose generation is not async yet
    } else if (modelMode === 'blockswap_q8') {
      // BlockSwap Q8 Experimental endpoint
      endpoint = `${BACKEND_BASE}/generate-blockswap-q8-async`
      formData.append('steps', String(steps))
      formData.append('cfg', String(cfg))
      formData.append('seed', String(seed))
      formData.append('shift', String(bsShift))
      formData.append('nag_scale', String(bsNagScale))
      formData.append('high_noise_steps', String(bsHighNoiseSteps))
      formData.append('enable_florence2', String(bsEnableFlorence2))
      formData.append('enable_upscale', String(bsEnableUpscale))
      formData.append('enable_interpolation', String(bsEnableInterpolation))
      // LoRA parameters - send as JSON array
      if (loraConfigs.length > 0) {
        formData.append('lora_configs', JSON.stringify(loraConfigs))
      }
    } else if (modelMode === 'distorch2_q8') {
      // DisTorch2 Q8 Experimental endpoint
      endpoint = `${BACKEND_BASE}/generate-distorch2-q8-async`
      formData.append('steps', String(steps))
      formData.append('cfg', String(cfg))
      formData.append('seed', String(seed))
      formData.append('shift', String(bsShift))
      formData.append('nag_scale', String(bsNagScale))
      formData.append('high_noise_steps', String(bsHighNoiseSteps))
      formData.append('enable_florence2', String(bsEnableFlorence2))
      formData.append('enable_upscale', String(bsEnableUpscale))
      formData.append('enable_interpolation', String(bsEnableInterpolation))
      // LoRA parameters - send as JSON array
      if (loraConfigs.length > 0) {
        formData.append('lora_configs', JSON.stringify(loraConfigs))
      }
    } else if (modelMode === 'ltx2') {
      // LTX-2 endpoint
      endpoint = `${BACKEND_BASE}/generate-ltx2-i2v-async`
      formData.append('steps', String(steps))
      formData.append('cfg', String(cfg))
      formData.append('seed', String(seed))
      // Post-processing chain (LTX-2 supports the same post-processing)
      const postProcessing = []
      if (postUpscale) {
        postProcessing.push({ type: 'upscale', scale: postUpscaleScale })
      }
      if (postInterpolate) {
        postProcessing.push({ type: 'interpolate', target_fps: postInterpolateFps })
      }
      if (postAudio && postAudioFile) {
        formData.append('post_audio_file', postAudioFile)
        postProcessing.push({ type: 'add_audio' })
      }
      if (postProcessing.length > 0) {
        formData.append('post_processing', JSON.stringify(postProcessing))
      }
    } else {
      // Use async ComfyUI endpoint for Wan2.2 Q6 - returns immediately with prompt_id
      endpoint = `${BACKEND_BASE}/generate-wan22-async`
      formData.append('steps', String(steps))
      formData.append('cfg', String(cfg))
      formData.append('seed', String(seed))
      // Extend mode parameters
      if (extendMode && clipCount > 1) {
        formData.append('extend_mode', 'true')
        formData.append('clip_count', String(clipCount))
      }
      // Unet parameters
      if (unetHighNoise) formData.append('unet_high_noise', unetHighNoise)
      if (unetLowNoise) formData.append('unet_low_noise', unetLowNoise)
      // LoRA parameters - send as JSON array
      if (loraConfigs.length > 0) {
        formData.append('lora_configs', JSON.stringify(loraConfigs))
      }
      // Post-processing chain
      const postProcessing = []
      if (postUpscale) {
        postProcessing.push({ type: 'upscale', scale: postUpscaleScale })
      }
      if (postInterpolate) {
        postProcessing.push({ type: 'interpolate', target_fps: postInterpolateFps })
      }
      if (postAudio && postAudioFile) {
        formData.append('post_audio_file', postAudioFile)
        postProcessing.push({ type: 'add_audio' })
      }
      if (postProcessing.length > 0) {
        formData.append('post_processing', JSON.stringify(postProcessing))
      }
    }

    try {
      if (DEBUG) console.debug('🐛 submit image-to-video', { duration, numFrames, usePose, resolution, fps, modelMode, useAsync })
      const result = await postForm(endpoint, formData)
      if (!result.ok) {
        setError(result.data?.detail || `Generation failed (status ${result.status})`)
        return
      }

      if (useAsync) {
        // Async mode - job was queued, notify parent to refresh queue panel
        if (DEBUG) console.debug('🐛 Job queued:', result.data?.prompt_id)
        if (onJobSubmitted) {
          onJobSubmitted(result.data)
        }
        // Don't wait - job will appear in queue and output when done
        // Clear busy state immediately so user can queue more jobs
      } else {
        // Sync mode - result contains the video
        // Use getMediaUrl helper for signed URL support
        const videoUrl = result.data?.video_url || result.data?.url
        const outputVideo = result.data?.output_video
        const url = getMediaUrl(videoUrl, result.data?.signed_url)

        onOutput({
          kind: 'video',
          url,
          backendUrl: url,
          filename: outputVideo,
          meta: result.data,
        })
        onRefreshHistory()
      }
    } catch (e) {
      const message = e?.message || 'Failed to generate video'
      setError(message)
      await sendClientLog({
        level: 'error',
        message: 'Image-to-video failed',
        timestamp: new Date().toISOString(),
        meta: { message, modelMode },
      })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="tool-container">
      {/* Import from previous generation modal */}
      {importModal && (
        <MediaImportModal
          item={importModal.item}
          parsedData={importModal.workflow}
          availableFields={['image', 'positive', 'negative', 'steps', 'cfg', 'seed', 'loras']}
          onApply={handleApplyImport}
          onClose={() => setImportModal(null)}
        />
      )}

      <style>{`
        .animate-spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>

      {/* Mode Selection */}
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
                No saved profiles yet. Your settings auto-save to "default".
              </div>
            )}
            {/* Factory Presets */}
            {factoryPresets?.length > 0 && (
              <div style={{ marginTop: '10px', borderTop: '1px solid var(--border-subtle, #333)', paddingTop: '8px' }}>
                <div style={{ fontSize: '11px', color: 'var(--text-secondary, #999)', marginBottom: '6px', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px' }}>
                  ⚡ Factory Presets
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
              value={modelMode}
              onChange={(e) => {
                const newMode = e.target.value
                setModelMode(newMode)
                // Adjust defaults per model
                if (newMode === 'wan2.2') {
                  setResolution('480p')  // Best quality/length ratio for Wan2.2
                  setAspectRatio('9:16')
                  setDuration(8)
                  setSteps(6)
                  setCfg(1.0)
                } else if (newMode === 'ltx2') {
                  setResolution('576p')  // LTX-2 handles higher res better
                  setAspectRatio('9:16')
                  setDuration(5)
                  setSteps(20)  // LTX-2 needs more steps
                  setCfg(3.0)
                } else if (newMode === 'blockswap_q8' || newMode === 'distorch2_q8') {
                  setResolution('720p')
                  setAspectRatio('9:16')
                  setDuration(7)
                  setSteps(8)
                  setCfg(1.0)
                  setBsShift(9.0)
                  setBsHighNoiseSteps(4)
                  setBsNagScale(11.0)
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
                <option
                  key={mode.value}
                  value={mode.value}
                  style={{ backgroundColor: '#1a1a1a', color: '#fff' }}
                >
                  {mode.label}
                </option>
              ))}
            </select>
            <ChevronDown
              size={20}
              style={{
                position: 'absolute',
                right: '12px',
                top: '50%',
                transform: 'translateY(-50%)',
                pointerEvents: 'none',
                color: 'var(--text-muted)'
              }}
            />
          </div>
          {modelMode === 'wan2.2' ? (
            <div className="info-badge" style={{ marginTop: '8px' }}>
              <span style={{ fontWeight: 600 }}>🎬 Wan2.2 14B Q6</span> • <span style={{ color: '#93c5fd' }}>DisTorch2 Multi-GPU</span>
              <div style={{ marginTop: '4px', opacity: 0.8 }}>
                Dual-pass (high/low noise) • All resolutions up to 30s
              </div>
            </div>
          ) : modelMode === 'blockswap_q8' ? (
            <div className="info-badge" style={{ marginTop: '8px', borderColor: '#f59e0b' }}>
              <span style={{ fontWeight: 600 }}>🧪 BlockSwap Q8 Experimental</span> • <span style={{ color: '#fbbf24' }}>Q8_0 Single-GPU</span>
              <div style={{ marginTop: '4px', opacity: 0.8 }}>
                Lightning LoRA + NAG + TorchCompile + EnhanceAVideo • Florence2 captioning • BlockSwap VRAM swap
              </div>
              <div style={{ marginTop: '2px', opacity: 0.6, fontSize: '0.75rem' }}>
                Single GPU • All resolutions up to 30s
              </div>
            </div>
          ) : modelMode === 'distorch2_q8' ? (
            <div className="info-badge" style={{ marginTop: '8px', borderColor: '#a78bfa' }}>
              <span style={{ fontWeight: 600 }}>🧪 DisTorch2 Q8 Experimental</span> • <span style={{ color: '#c4b5fd' }}>Q8_0 Multi-GPU</span>
              <div style={{ marginTop: '4px', opacity: 0.8 }}>
                DisTorch2 Dual-GPU • NAG + EnhanceAVideo • Selectable LoRAs • Florence2 captioning
              </div>
              <div style={{ marginTop: '2px', opacity: 0.6, fontSize: '0.75rem' }}>
                All resolutions up to 30s
              </div>
            </div>
          ) : (
            <div className="info-badge" style={{ marginTop: '8px' }}>
              <span style={{ fontWeight: 600 }}>⚡ LTX-2 19B</span> • <span style={{ color: '#86efac' }}>Single Model</span>
              <div style={{ marginTop: '4px', opacity: 0.8 }}>
                Faster inference • No high/low noise • Uses Gemma 3 text encoder
              </div>
            </div>
          )}
        </div>

        {/* Unet Model Selection - Only for Wan2.2 */}
        {modelMode === 'wan2.2' && (
        <div style={{ marginTop: '12px', paddingTop: '12px', borderTop: '1px solid var(--border-color)' }}>
          <div
            onClick={() => setShowUnetPanel(!showUnetPanel)}
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              cursor: 'pointer',
              padding: '4px 0'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Settings2 size={16} />
              <span style={{ fontSize: '0.9rem', fontWeight: 500 }}>Unet Model</span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                ({unetHighNoise.replace('.gguf', '').replace('wan2.2_i2v_', '')})
              </span>
            </div>
            <span style={{ opacity: 0.5, fontSize: '0.8rem' }}>{showUnetPanel ? '▼' : '▶'}</span>
          </div>

          {showUnetPanel && (
            <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {/* Model Pair Selector - Easy mode */}
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
                  Model Pair (recommended)
                </label>
                <select
                  onChange={(e) => {
                    const pair = availableUnets.pairs?.find(p => p.name === e.target.value)
                    if (pair) {
                      setUnetHighNoise(pair.high.path)
                      setUnetLowNoise(pair.low.path)
                    }
                  }}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    backgroundColor: 'var(--bg-secondary)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '6px',
                    color: 'var(--text-primary)',
                    fontSize: '0.85rem'
                  }}
                  value={availableUnets.pairs?.find(p => p.high.path === unetHighNoise && p.low.path === unetLowNoise)?.name || ''}
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
                  ⚙️ Advanced: Select models separately
                </summary>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', paddingTop: '8px' }}>
                  {/* High Noise Model */}
                  <div>
                    <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
                      High Noise Model (steps 0-3)
                    </label>
                    <select
                      value={unetHighNoise}
                      onChange={(e) => setUnetHighNoise(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '8px 12px',
                        backgroundColor: 'var(--bg-secondary)',
                        border: '1px solid var(--border-color)',
                        borderRadius: '6px',
                        color: 'var(--text-primary)',
                        fontSize: '0.85rem'
                      }}
                    >
                      {availableUnets.high_noise?.map((model) => (
                        <option key={model.path} value={model.path}>
                          {model.name} ({model.size_gb}GB)
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Low Noise Model */}
                  <div>
                    <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
                      Low Noise Model (steps 3+)
                    </label>
                    <select
                      value={unetLowNoise}
                      onChange={(e) => setUnetLowNoise(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '8px 12px',
                        backgroundColor: 'var(--bg-secondary)',
                        border: '1px solid var(--border-color)',
                        borderRadius: '6px',
                        color: 'var(--text-primary)',
                        fontSize: '0.85rem'
                      }}
                    >
                      {availableUnets.low_noise?.map((model) => (
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

      {/* Positive Prompt */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            Positive Prompt <span style={{ fontWeight: 400, color: 'var(--text-muted)', fontSize: '0.85rem' }}>(Describe the motion)</span>
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
                  position: 'absolute',
                  top: '100%',
                  left: '50%',
                  transform: 'translateX(-50%)',
                  marginTop: '8px',
                  backgroundColor: '#1a1a1a',
                  border: '1px solid #fbbf24',
                  borderRadius: '8px',
                  padding: '12px',
                  width: '280px',
                  zIndex: 100,
                  fontSize: '0.8rem',
                  color: '#fbbf24',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
                }}>
                  <div style={{ fontWeight: 600, marginBottom: '8px' }}>💡 Prompt Tips</div>
                  <ul style={{ margin: 0, paddingLeft: '16px', lineHeight: 1.6 }}>
                    <li>Structure: [subject + motion] + [scene] + [camera]</li>
                    <li>Focus on motion - "walking slowly", "hair blowing"</li>
                    <li>Add intensity - "quickly", "gently", "dramatically"</li>
                    <li>Camera moves - "slow zoom in", "pan left"</li>
                    <li>Describe what you want, not what to avoid</li>
                  </ul>
                </div>
              )}
            </div>
          </div>
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
              onClick={async () => {
                if (!previewUrl) return
                try {
                  const res = await fetch(`${BACKEND_BASE}/extract-metadata-url`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_url: previewUrl })
                  })
                  const data = await res.json()
                  if (data.positive_prompt) setPrompt(data.positive_prompt)
                  if (data.negative_prompt) setNegPrompt(data.negative_prompt)
                } catch (e) {
                  console.error('Extract metadata failed:', e)
                }
              }}
              title="Extract prompt from selected image"
              disabled={!previewUrl}
            >
              🔍
            </button>
            <button
              className="icon-btn"
              style={{ width: '24px', height: '24px', fontSize: '14px' }}
              onClick={() => setPrompt(getRandomPrompt(nsfwEnabled))}
              title="Generate random creative prompt"
            >
              ✨
            </button>
            {/* Divider */}
            <div style={{ width: '1px', height: '20px', backgroundColor: 'var(--border-color)', margin: '0 2px' }} />
            {/* Analyze & Generate SFW */}
            <button
              className="icon-btn"
              style={{
                width: 'auto',
                height: '24px',
                padding: '4px 8px',
                fontSize: '0.7rem',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                background: isAnalyzing ? 'var(--bg-input)' : 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
                color: 'white',
                border: 'none',
              }}
              onClick={() => handleAnalyzeAndGenerate(false)}
              disabled={isAnalyzing || !previewUrl}
              title="Analyze image with AI and generate creative SFW video prompt"
            >
              {isAnalyzing ? <Loader2 size={12} className="spin" /> : '🔮'}
              <span>SFW</span>
            </button>
            {/* Analyze & Generate NSFW - only show if NSFW enabled */}
            {nsfwEnabled && (
              <button
                className="icon-btn"
                style={{
                  width: 'auto',
                  height: '24px',
                  padding: '4px 8px',
                  fontSize: '0.7rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  background: isAnalyzing ? 'var(--bg-input)' : 'linear-gradient(135deg, #ec4899, #f43f5e)',
                  color: 'white',
                  border: 'none',
                }}
                onClick={() => handleAnalyzeAndGenerate(true)}
                disabled={isAnalyzing || !previewUrl}
                title="Analyze image with AI and generate creative NSFW video prompt"
              >
                {isAnalyzing ? <Loader2 size={12} className="spin" /> : '🔥'}
                <span>NSFW</span>
              </button>
            )}
          </div>
        </div>

        {/* Refine Prompt - inline instruction input */}
        {showRefineInput && (
          <div style={{
            marginTop: '8px',
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

        {/* Image Description - show if we have one from analysis */}
        {imageDescription && (
          <div style={{
            marginTop: '8px',
            padding: '8px 12px',
            backgroundColor: 'rgba(139, 92, 246, 0.1)',
            border: '1px solid rgba(139, 92, 246, 0.3)',
            borderRadius: '8px',
            fontSize: '0.75rem',
            color: 'var(--text-secondary)',
          }}>
            <div style={{ fontWeight: 600, marginBottom: '4px', color: '#a78bfa' }}>📝 Image Analysis:</div>
            <div style={{ lineHeight: 1.5 }}>{imageDescription}</div>
          </div>
        )}

        {/* Camera Motion Selector */}
        <CameraMotionSelector value={cameraMotion} onChange={setCameraMotion} />

        <div style={{ position: 'relative' }}>
          <textarea
            className="form-textarea"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={4}
            placeholder="Describe how you want the image to move or animate... (Optional for image-to-video)"
            style={{
              backgroundColor: '#0f0f0f',
              border: '1px solid var(--border-color)',
              borderRadius: '8px',
              resize: 'vertical',
              minHeight: '80px',
              padding: '12px',
              paddingBottom: '28px',
              width: '100%',
              boxSizing: 'border-box'
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
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              cursor: 'pointer',
              padding: '8px 0'
            }}
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
                  backgroundColor: '#0f0f0f',
                  border: '1px solid var(--border-color)',
                  borderRadius: '8px',
                  resize: 'vertical',
                  minHeight: '60px',
                  padding: '12px',
                  paddingBottom: '28px',
                  width: '100%',
                  boxSizing: 'border-box',
                  fontSize: '0.85rem'
                }}
              />
              <div style={{ position: 'absolute', bottom: '8px', right: '8px', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                {negativePrompt.length}/2048
              </div>
            </div>
          )}
        </div>

        {/* Prompt Strength Slider */}
        <div style={{ marginTop: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Prompt Strength / CFG
              </label>
              <span
                title="How strictly the video follows your prompt. Low = subtle movement, High = dramatic action (may cause artifacts)"
                style={{ cursor: 'help', opacity: 0.5 }}
              >
                <HelpCircle size={12} />
              </span>
            </div>
            <span style={{
              fontSize: '0.8rem',
              fontWeight: 600,
              color: cfg <= 1.5 ? '#fbbf24' : cfg <= 3 ? '#34d399' : '#f87171',
              padding: '2px 8px',
              borderRadius: '4px',
              backgroundColor: cfg <= 1.5 ? 'rgba(251,191,36,0.1)' : cfg <= 3 ? 'rgba(52,211,153,0.1)' : 'rgba(248,113,113,0.1)',
            }}>
              {cfg <= 1.5 ? '🌊 Subtle' : cfg <= 3 ? '⚡ Balanced' : '🔥 Strong'}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <input
              type="range"
              min="1"
              max="5"
              step="0.1"
              value={cfg}
              onChange={(e) => setCfg(parseFloat(e.target.value))}
              style={{ flex: 1 }}
            />
            <span style={{
              fontSize: '0.85rem',
              fontWeight: 600,
              minWidth: '32px',
              textAlign: 'right',
              color: 'var(--text-primary)'
            }}>
              {cfg.toFixed(1)}
            </span>
          </div>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: '0.7rem',
            color: 'var(--text-muted)',
            marginTop: '4px'
          }}>
            <span>Subtle motion</span>
            <span>Strong action</span>
          </div>
        </div>
      </div>

      {/* Upload Photo */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Upload Photo</div>
        </div>

        <div className="grok-tabs">
          <button
            className={`grok-tab ${uploadTab === 'file' ? 'active' : ''}`}
            onClick={() => setUploadTab('file')}
          >
            <Upload size={14} /> Upload File
          </button>
          <button
            className={`grok-tab ${uploadTab === 'url' ? 'active' : ''}`}
            onClick={() => setUploadTab('url')}
          >
            <Link size={14} /> From URL
          </button>
          <button
            className={`grok-tab ${uploadTab === 'creations' ? 'active' : ''}`}
            onClick={() => setUploadTab('creations')}
          >
            <FolderOpen size={14} /> From My Creations
          </button>
          <button
            className={`grok-tab ${uploadTab === 'library' ? 'active' : ''}`}
            onClick={() => setUploadTab('library')}
          >
            <Grid size={14} /> My Uploads
            {userUploads.length > 0 && (
              <span style={{ fontSize: '10px', opacity: 0.7, marginLeft: '4px' }}>({userUploads.length})</span>
            )}
          </button>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={(e) => onPickFile(e.target.files?.[0])}
          style={{ display: 'none' }}
        />

        {/* Tab Content: File Upload */}
        {uploadTab === 'file' && !file && !restoringImage && (
          <div className="upload-box" onClick={() => fileInputRef.current?.click()} style={{ cursor: 'pointer', borderStyle: 'dashed', minHeight: '200px', justifyContent: 'center' }}>
            <Upload size={48} className="text-muted" style={{ opacity: 0.2 }} />
            <div style={{ fontSize: '1rem', fontWeight: 500, color: 'var(--text-secondary)' }}>
              Drag & drop an image here, or click to browse
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              JPEG, PNG, WebP, Max 20MB
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Minimum size: 300x300px
            </div>
          </div>
        )}

        {/* Restoring image from profile */}
        {restoringImage && !file && (
          <div style={{
            padding: '32px', textAlign: 'center', color: 'var(--text-muted)',
            backgroundColor: 'var(--bg-secondary)', borderRadius: '8px',
            border: '1px dashed var(--border-color)', minHeight: '200px',
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center'
          }}>
            <Loader2 size={32} className="animate-spin" style={{ marginBottom: '12px', opacity: 0.5 }} />
            <div style={{ fontSize: '0.9rem' }}>Restoring source image from profile...</div>
          </div>
        )}

        {/* Tab Content: URL */}
        {uploadTab === 'url' && !file && (
          <div style={{ padding: '16px 0' }}>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
              Enter image URL:
            </div>
            <input
              type="url"
              placeholder="https://example.com/image.jpg"
              style={{
                width: '100%',
                padding: '12px',
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                fontSize: '0.9rem'
              }}
              onKeyDown={async (e) => {
                if (e.key === 'Enter' && e.target.value) {
                  try {
                    const response = await fetch(e.target.value)
                    const blob = await response.blob()
                    const filename = e.target.value.split('/').pop() || 'image.jpg'
                    const file = new File([blob], filename, { type: blob.type })
                    onPickFile(file)
                  } catch {
                    setError('Failed to load image from URL')
                  }
                }
              }}
            />
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '8px' }}>
              Press Enter to load
            </div>
          </div>
        )}

        {/* Tab Content: My Creations - show instruction, picker is in output panel */}
        {uploadTab === 'creations' && !file && (
          <div style={{
            padding: '24px 16px',
            textAlign: 'center',
            color: 'var(--text-muted)',
            backgroundColor: 'var(--bg-secondary)',
            borderRadius: '8px',
            border: '1px dashed var(--border-color)'
          }}>
            <ImageIcon size={32} style={{ opacity: 0.5, marginBottom: '12px' }} />
            <div style={{ fontSize: '0.9rem', marginBottom: '8px' }}>
              Select an image from the panel on the right →
            </div>
            <div style={{ fontSize: '0.8rem', opacity: 0.7 }}>
              Browse your generated images
            </div>
          </div>
        )}

        {/* Tab Content: My Uploads Library */}
        {uploadTab === 'library' && !file && (
          <div style={{ padding: '8px 0' }}>
            {uploadsLoading ? (
              <div style={{ textAlign: 'center', padding: '32px', color: 'var(--text-muted)' }}>
                <Loader2 size={24} className="animate-spin" style={{ margin: '0 auto 8px' }} />
                <div style={{ fontSize: '0.85rem' }}>Loading your uploads...</div>
              </div>
            ) : userUploads.length === 0 ? (
              <div style={{
                padding: '24px 16px',
                textAlign: 'center',
                color: 'var(--text-muted)',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: '8px',
                border: '1px dashed var(--border-color)'
              }}>
                <ImageIcon size={32} style={{ opacity: 0.5, marginBottom: '12px' }} />
                <div style={{ fontSize: '0.9rem', marginBottom: '8px' }}>
                  No uploaded images yet
                </div>
                <div style={{ fontSize: '0.8rem', opacity: 0.7 }}>
                  Upload an image via the Upload File tab — it will appear here for reuse
                </div>
              </div>
            ) : (
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))',
                gap: '8px',
                maxHeight: '300px',
                overflowY: 'auto',
                padding: '4px'
              }}>
                {userUploads.map((item, idx) => (
                  <div
                    key={idx}
                    onClick={() => selectFromLibrary(item)}
                    style={{
                      position: 'relative',
                      aspectRatio: '1',
                      borderRadius: '8px',
                      overflow: 'hidden',
                      cursor: 'pointer',
                      border: '2px solid transparent',
                      transition: 'border-color 0.2s, transform 0.15s',
                    }}
                    onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--accent-color, #6366f1)'; e.currentTarget.style.transform = 'scale(1.03)'; e.currentTarget.querySelector('.lib-del-btn').style.opacity = '1' }}
                    onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'transparent'; e.currentTarget.style.transform = 'scale(1)'; e.currentTarget.querySelector('.lib-del-btn').style.opacity = '0' }}
                  >
                    <img
                      src={item.blobUrl || getMediaUrl(item.url, item.signed_url)}
                      alt={item.name}
                      loading="lazy"
                      style={{
                        width: '100%',
                        height: '100%',
                        objectFit: 'cover',
                        borderRadius: '6px',
                      }}
                    />
                    <button
                      className="lib-del-btn"
                      onClick={(e) => deleteFromLibrary(item, e)}
                      title="Remove from library"
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
                        opacity: 0,
                        transition: 'opacity 0.2s',
                        backdropFilter: 'blur(4px)',
                      }}
                    >
                      <Trash2 size={12} />
                    </button>
                    <div style={{
                      position: 'absolute',
                      bottom: 0, left: 0, right: 0,
                      background: 'linear-gradient(transparent, rgba(0,0,0,0.7))',
                      padding: '12px 6px 4px',
                      fontSize: '0.65rem',
                      color: '#fff',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                    }}>
                      {item.name}
                    </div>
                  </div>
                ))}
              </div>
            )}
            {userUploads.length > 0 && (
              <div style={{ textAlign: 'right', marginTop: '6px' }}>
                <button
                  onClick={loadUserUploads}
                  style={{
                    background: 'none', border: 'none', color: 'var(--text-muted)',
                    fontSize: '0.75rem', cursor: 'pointer', textDecoration: 'underline',
                  }}
                >
                  ↻ Refresh
                </button>
              </div>
            )}
          </div>
        )}

        {/* Preview when file is selected (any tab) */}
        {file && (
          <div className="relative" style={{ position: 'relative' }}>
            <img
              src={previewUrl}
              alt="Preview"
              style={{
                width: '100%',
                maxHeight: '400px',
                objectFit: 'contain',
                borderRadius: '8px',
                border: '1px solid var(--border-color)'
              }}
            />
            <button
              onClick={(e) => { e.stopPropagation(); clearFile(); }}
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
                backdropFilter: 'blur(4px)'
              }}
            >
              <X size={18} />
            </button>
          </div>
        )}
      </div>

      {/* Settings */}
      <div className="grok-card">
        <div className="form-group">
          <label className="grok-section-label">
            Resolution
            <span className="text-muted" style={{ fontWeight: 400 }}>
              {' (Higher = Better Quality, more VRAM)'}
            </span>
          </label>
          <div className="grok-toggle-group">
            {Object.entries(RESOLUTION_PRESETS).map(([key, preset]) => (
              <button
                key={key}
                className={`grok-toggle-btn ${resolution === key ? 'active' : ''}`}
                onClick={() => setResolution(key)}
              >
                {preset.label}
                <span style={{ fontSize: '0.7rem', opacity: 0.7, display: 'block' }}>
                  {preset.dimensions[aspectRatio] || preset.dimensions['1:1']}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Aspect Ratio */}
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
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <label className="grok-section-label">Duration</label>
            <span className="nav-badge" style={{ fontSize: '0.8rem' }}>{duration}s ({duration * fps}f)</span>
          </div>
          <div style={{ position: 'relative', height: '24px', marginBottom: '8px' }}>
            <input
              type="range"
              min="3"
              max={maxDuration}
              step="1"
              value={duration}
              onChange={(e) => setDuration(parseInt(e.target.value, 10))}
              style={{ width: '100%', opacity: 0, position: 'absolute', zIndex: 2, cursor: 'pointer' }}
            />
            <div style={{
              position: 'absolute',
              top: '10px',
              left: 0,
              right: 0,
              height: '4px',
              backgroundColor: '#333',
              borderRadius: '2px'
            }}>
              <div style={{
                width: `${((duration - 3) / (maxDuration - 3)) * 100}%`,
                height: '100%',
                backgroundColor: 'var(--accent-color, #a855f7)',
                borderRadius: '2px'
              }} />
            </div>
            <div style={{
              position: 'absolute',
              top: '2px',
              left: `calc(${((duration - 3) / (maxDuration - 3)) * 100}% - 10px)`,
              width: '20px',
              height: '20px',
              backgroundColor: 'white',
              borderRadius: '50%',
              boxShadow: '0 2px 4px rgba(0,0,0,0.3)'
            }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            <span>3s</span>
            <span>{Math.floor((3 + maxDuration) / 2)}s</span>
            <span>{maxDuration}s (max)</span>
          </div>
        </div>

        {/* FPS Control */}
        <div className="form-group">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <label className="grok-section-label">Frame Rate (FPS)</label>
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
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '8px' }}>
            Higher FPS = smoother motion, more VRAM required
          </div>
        </div>

        {/* Model Version - only for non-Wan2.2 modes */}
        {modelMode !== 'wan2.2' && (
          <div className="form-group">
            <label className="grok-section-label">Model Version</label>
            <div className="grok-toggle-group">
              <button
                className={`grok-toggle-btn ${modelVersion === 'v1' ? 'active' : ''}`}
                onClick={() => setModelVersion('v1')}
              >
                V1
              </button>
              <button
                className={`grok-toggle-btn ${modelVersion === 'v2' ? 'active' : ''}`}
                onClick={() => setModelVersion('v2')}
              >
                V2 (Enhanced)
              </button>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '8px' }}>
              V2 features improved video quality, motion, and optional audio generation
            </div>
          </div>
        )}

        {/* Workflow Presets - Quick configuration */}
        {modelMode === 'wan2.2' && (
          <div style={{
            backgroundColor: 'var(--bg-tertiary)',
            padding: '16px',
            borderRadius: '8px',
            marginTop: '8px'
          }}>
            <div
              onClick={() => setUsePresets(!usePresets)}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                cursor: 'pointer'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Sliders size={16} />
                <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>Workflow Presets</span>
                {selectedPreset && (
                  <span style={{
                    fontSize: '0.7rem',
                    backgroundColor: 'var(--accent-color)',
                    color: 'white',
                    padding: '2px 6px',
                    borderRadius: '4px',
                    marginLeft: '4px'
                  }}>
                    {selectedPreset.name}
                  </span>
                )}
              </div>
              <span style={{ opacity: 0.5, fontSize: '0.8rem' }}>{usePresets ? '▼' : '▶'}</span>
            </div>

            {usePresets && (
              <div style={{ marginTop: '12px' }}>
                <PresetSelector
                  onPresetChange={(preset) => {
                    setSelectedPreset(preset)
                    // Apply preset parameters to local state
                    if (preset?.parameters) {
                      const params = preset.parameters
                      if (params.steps?.default) setSteps(params.steps.default)
                      if (params.cfg?.default) setCfg(params.cfg.default)
                      if (params.seed?.default !== undefined) setSeed(params.seed.default)
                      if (params.frame_rate?.default) setFps(params.frame_rate.default)
                    }
                  }}
                  onParametersChange={(params) => {
                    setPresetParameters(params)
                    // Sync with local state
                    if (params.steps !== undefined) setSteps(params.steps)
                    if (params.cfg !== undefined) setCfg(params.cfg)
                    if (params.seed !== undefined) setSeed(params.seed)
                    if (params.frame_rate !== undefined) setFps(params.frame_rate)
                  }}
                  currentParameters={{ steps, cfg, seed, frame_rate: fps }}
                />
              </div>
            )}
          </div>
        )}

        {/* Advanced Settings for BlockSwap Q8 / DisTorch2 Q8 Experimental */}
        {(modelMode === 'blockswap_q8' || modelMode === 'distorch2_q8') && (
          <div style={{
            backgroundColor: 'var(--bg-tertiary)',
            padding: '16px',
            borderRadius: '8px',
            marginTop: '8px'
          }}>
            <div
              onClick={() => setShowAdvanced(!showAdvanced)}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                cursor: 'pointer'
              }}
            >
              <div style={{
                fontSize: '0.9rem',
                fontWeight: 600,
                color: 'var(--text-primary)'
              }}>
                {modelMode === 'blockswap_q8' ? '🧪 BlockSwap Q8 Settings' : '🧪 DisTorch2 Q8 Settings'}
              </div>
              <span style={{ opacity: 0.5, fontSize: '0.8rem' }}>{showAdvanced ? '▼' : '▶'}</span>
            </div>

            {showAdvanced && (
              <div style={{ marginTop: '12px' }}>
                {/* Steps */}
                <div className="form-group" style={{ marginBottom: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <label className="grok-section-label">Sampling Steps</label>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{steps}</span>
                  </div>
                  <input
                    type="range" min="4" max="20" step="1"
                    value={steps}
                    onChange={(e) => setSteps(parseInt(e.target.value, 10))}
                    style={{ width: '100%', cursor: 'pointer' }}
                  />
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                    <span>4 (fast)</span><span>8 (rec)</span><span>20 (quality)</span>
                  </div>
                </div>

                {/* High Noise Steps */}
                <div className="form-group" style={{ marginBottom: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <label className="grok-section-label">High Noise Steps</label>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{bsHighNoiseSteps} of {steps}</span>
                  </div>
                  <input
                    type="range" min="1" max={Math.max(steps - 1, 2)} step="1"
                    value={bsHighNoiseSteps}
                    onChange={(e) => setBsHighNoiseSteps(parseInt(e.target.value, 10))}
                    style={{ width: '100%', cursor: 'pointer' }}
                  />
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                    Steps using high-noise Q8 model before switching to low-noise
                  </div>
                </div>

                {/* Shift */}
                <div className="form-group" style={{ marginBottom: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <label className="grok-section-label">Model Shift</label>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{bsShift.toFixed(1)}</span>
                  </div>
                  <input
                    type="range" min="1.0" max="20.0" step="0.5"
                    value={bsShift}
                    onChange={(e) => setBsShift(parseFloat(e.target.value))}
                    style={{ width: '100%', cursor: 'pointer' }}
                  />
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                    <span>1.0</span><span>8.0 (rec)</span><span>20.0</span>
                  </div>
                </div>

                {/* NAG Scale */}
                <div className="form-group" style={{ marginBottom: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <label className="grok-section-label">NAG Scale</label>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{bsNagScale.toFixed(1)}</span>
                  </div>
                  <input
                    type="range" min="1.0" max="20.0" step="0.5"
                    value={bsNagScale}
                    onChange={(e) => setBsNagScale(parseFloat(e.target.value))}
                    style={{ width: '100%', cursor: 'pointer' }}
                  />
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                    Normalized Attention Guidance — higher = more prompt adherence
                  </div>
                </div>

                {/* Seed */}
                <div className="form-group" style={{ marginBottom: '16px' }}>
                  <label className="grok-section-label">Seed</label>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <input
                      type="number" value={seed}
                      onChange={(e) => setSeed(parseInt(e.target.value, 10))}
                      placeholder="-1 for random"
                      style={{
                        flex: 1, padding: '8px 12px',
                        backgroundColor: 'var(--bg-secondary)',
                        border: '1px solid var(--border-color)',
                        borderRadius: '6px', color: 'var(--text-primary)', fontSize: '0.9rem'
                      }}
                    />
                    <button className="btn ghost sm" onClick={() => setSeed(-1)} style={{ whiteSpace: 'nowrap' }}>Random</button>
                  </div>
                </div>

                {/* Feature Toggles */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', paddingTop: '12px', borderTop: '1px solid var(--border-color)' }}>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '4px' }}>Features</div>
                  {/* Florence2 auto-captioning */}
                  <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                    <input type="checkbox" checked={bsEnableFlorence2} onChange={(e) => setBsEnableFlorence2(e.target.checked)} style={{ width: '16px', height: '16px' }} />
                    <span>🔍 Florence2 Auto-Caption</span>
                    <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>(analyzes image for prompt)</span>
                  </label>
                  {/* Upscale */}
                  <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                    <input type="checkbox" checked={bsEnableUpscale} onChange={(e) => setBsEnableUpscale(e.target.checked)} style={{ width: '16px', height: '16px' }} />
                    <span>📈 4x Upscale (RealESRGAN)</span>
                  </label>
                  {/* RIFE Interpolation */}
                  <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                    <input type="checkbox" checked={bsEnableInterpolation} onChange={(e) => setBsEnableInterpolation(e.target.checked)} style={{ width: '16px', height: '16px' }} />
                    <span>🎞 RIFE 2x Frame Interpolation</span>
                  </label>
                </div>

                {/* LoRA Settings for BS/DT2 Q8 */}
                <div style={{
                  marginTop: '16px',
                  paddingTop: '16px',
                  borderTop: '1px solid var(--border-color)'
                }}>
                  <div
                    onClick={() => setShowLoraPanel(!showLoraPanel)}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      cursor: 'pointer',
                      marginBottom: showLoraPanel ? '12px' : 0
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ fontSize: '0.9rem', fontWeight: 500 }}>🎨 LoRA Stack</span>
                      {loraConfigs.length > 0 && (
                        <span style={{
                          fontSize: '0.7rem',
                          padding: '2px 6px',
                          backgroundColor: 'rgba(var(--accent-rgb), 0.2)',
                          borderRadius: '10px',
                          color: 'var(--accent-color)'
                        }}>
                          {loraConfigs.length} active
                        </span>
                      )}
                    </div>
                    <span style={{ opacity: 0.5, fontSize: '0.8rem' }}>{showLoraPanel ? '▼' : '▶'}</span>
                  </div>

                  {showLoraPanel && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      {loraConfigs.map((config, idx) => (
                        <div key={idx} style={{
                          padding: '10px',
                          backgroundColor: 'var(--bg-secondary)',
                          borderRadius: '6px',
                          border: '1px solid var(--border-color)'
                        }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                            <span style={{ fontSize: '0.8rem', fontWeight: 500 }}>LoRA #{idx + 1}</span>
                            <button
                              onClick={() => setLoraConfigs(loraConfigs.filter((_, i) => i !== idx))}
                              style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '1rem' }}
                            >×</button>
                          </div>

                          {/* High Noise LoRA */}
                          <div style={{ marginBottom: '8px', position: 'relative' }}>
                            <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
                              High Noise (first pass)
                            </label>
                            {config.high ? (
                              <div style={{
                                display: 'flex', alignItems: 'center', gap: '6px',
                                padding: '6px 10px', backgroundColor: 'var(--bg-secondary)',
                                border: '1px solid var(--border-color)', borderRadius: '4px',
                                fontSize: '0.8rem', color: 'var(--text-primary)'
                              }}>
                                <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{config.high}</span>
                                <button onClick={() => {
                                  const newConfigs = [...loraConfigs]
                                  newConfigs[idx] = { ...config, high: '' }
                                  setLoraConfigs(newConfigs)
                                }} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '0.9rem', padding: '0 2px' }}>×</button>
                              </div>
                            ) : (
                              <div style={{ position: 'relative' }}>
                                <input
                                  type="text"
                                  placeholder="🔍 Search LoRA..."
                                  value={loraSearchHigh[idx] || ''}
                                  onChange={(e) => setLoraSearchHigh({ ...loraSearchHigh, [idx]: e.target.value })}
                                  onFocus={() => setLoraDropdownOpen(`high-${idx}`)}
                                  onBlur={() => setTimeout(() => setLoraDropdownOpen((prev) => prev === `high-${idx}` ? null : prev), 200)}
                                  style={{
                                    width: '100%', padding: '6px 10px',
                                    backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--accent-color, #3b82f6)',
                                    borderRadius: '4px', color: 'var(--text-primary)', fontSize: '0.8rem',
                                    outline: 'none', boxSizing: 'border-box'
                                  }}
                                />
                                {loraDropdownOpen === `high-${idx}` && (
                                  <div style={{
                                    position: 'absolute', top: '100%', left: 0, right: 0, zIndex: 50,
                                    maxHeight: '200px', overflowY: 'auto',
                                    backgroundColor: 'var(--bg-secondary, #1a1a1a)',
                                    border: '1px solid var(--border-color)',
                                    borderRadius: '0 0 4px 4px', boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
                                  }}>
                                    {filteredLoras.by_category && (() => {
                                      const searchTerm = (loraSearchHigh[idx] || '').toLowerCase()
                                      let totalMatches = 0
                                      const elements = Object.keys(filteredLoras.by_category).sort().map((category) => {
                                        const matches = filteredLoras.by_category[category].filter(l =>
                                          !searchTerm || l.name.toLowerCase().includes(searchTerm)
                                        )
                                        if (matches.length === 0) return null
                                        totalMatches += matches.length
                                        return (
                                          <div key={category}>
                                            <div style={{ padding: '4px 10px', fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 600, backgroundColor: 'rgba(255,255,255,0.03)', position: 'sticky', top: 0 }}>
                                              📁 {category === 'root' ? 'Root' : category}
                                            </div>
                                            {matches.map((lora) => (
                                              <div key={lora.name}
                                                onMouseDown={(e) => {
                                                  e.preventDefault()
                                                  const newConfigs = [...loraConfigs]
                                                  newConfigs[idx] = { ...config, high: lora.name }
                                                  setLoraConfigs(newConfigs)
                                                  setLoraSearchHigh({ ...loraSearchHigh, [idx]: '' })
                                                  setLoraDropdownOpen(null)
                                                }}
                                                style={{
                                                  padding: '6px 14px', fontSize: '0.8rem', cursor: 'pointer',
                                                  color: 'var(--text-primary)', transition: 'background-color 0.1s',
                                                }}
                                                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'rgba(59,130,246, 0.15)'}
                                                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                                              >
                                                {lora.name}
                                              </div>
                                            ))}
                                          </div>
                                        )
                                      })
                                      if (totalMatches === 0) return <div style={{ padding: '10px 14px', fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>No matches</div>
                                      return elements
                                    })()}
                                  </div>
                                )}
                              </div>
                            )}
                          </div>

                          {/* Low Noise LoRA */}
                          <div style={{ marginBottom: '8px', position: 'relative' }}>
                            <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
                              Low Noise (steps 3+)
                            </label>
                            {config.low ? (
                              <div style={{
                                display: 'flex', alignItems: 'center', gap: '6px',
                                padding: '6px 10px', backgroundColor: 'var(--bg-secondary)',
                                border: '1px solid var(--border-color)', borderRadius: '4px',
                                fontSize: '0.8rem', color: 'var(--text-primary)'
                              }}>
                                <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{config.low}</span>
                                <button onClick={() => {
                                  const newConfigs = [...loraConfigs]
                                  newConfigs[idx] = { ...config, low: '' }
                                  setLoraConfigs(newConfigs)
                                }} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '0.9rem', padding: '0 2px' }}>×</button>
                              </div>
                            ) : (
                              <div style={{ position: 'relative' }}>
                                <input
                                  type="text"
                                  placeholder="🔍 Search LoRA... (optional)"
                                  value={loraSearchLow[idx] || ''}
                                  onChange={(e) => setLoraSearchLow({ ...loraSearchLow, [idx]: e.target.value })}
                                  onFocus={() => setLoraDropdownOpen(`low-${idx}`)}
                                  onBlur={() => setTimeout(() => setLoraDropdownOpen((prev) => prev === `low-${idx}` ? null : prev), 200)}
                                  style={{
                                    width: '100%', padding: '6px 10px',
                                    backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)',
                                    borderRadius: '4px', color: 'var(--text-primary)', fontSize: '0.8rem',
                                    outline: 'none', boxSizing: 'border-box'
                                  }}
                                />
                                {loraDropdownOpen === `low-${idx}` && (
                                  <div style={{
                                    position: 'absolute', top: '100%', left: 0, right: 0, zIndex: 50,
                                    maxHeight: '200px', overflowY: 'auto',
                                    backgroundColor: 'var(--bg-secondary, #1a1a1a)',
                                    border: '1px solid var(--border-color)',
                                    borderRadius: '0 0 4px 4px', boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
                                  }}>
                                    {filteredLoras.by_category && (() => {
                                      const searchTerm = (loraSearchLow[idx] || '').toLowerCase()
                                      let totalMatches = 0
                                      const elements = Object.keys(filteredLoras.by_category).sort().map((category) => {
                                        const matches = filteredLoras.by_category[category].filter(l =>
                                          !searchTerm || l.name.toLowerCase().includes(searchTerm)
                                        )
                                        if (matches.length === 0) return null
                                        totalMatches += matches.length
                                        return (
                                          <div key={category}>
                                            <div style={{ padding: '4px 10px', fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 600, backgroundColor: 'rgba(255,255,255,0.03)', position: 'sticky', top: 0 }}>
                                              📁 {category === 'root' ? 'Root' : category}
                                            </div>
                                            {matches.map((lora) => (
                                              <div key={lora.name}
                                                onMouseDown={(e) => {
                                                  e.preventDefault()
                                                  const newConfigs = [...loraConfigs]
                                                  newConfigs[idx] = { ...config, low: lora.name }
                                                  setLoraConfigs(newConfigs)
                                                  setLoraSearchLow({ ...loraSearchLow, [idx]: '' })
                                                  setLoraDropdownOpen(null)
                                                }}
                                                style={{
                                                  padding: '6px 14px', fontSize: '0.8rem', cursor: 'pointer',
                                                  color: 'var(--text-primary)', transition: 'background-color 0.1s',
                                                }}
                                                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'rgba(59,130,246, 0.15)'}
                                                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                                              >
                                                {lora.name}
                                              </div>
                                            ))}
                                          </div>
                                        )
                                      })
                                      if (totalMatches === 0) return <div style={{ padding: '10px 14px', fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>No matches</div>
                                      return elements
                                    })()}
                                  </div>
                                )}
                              </div>
                            )}
                          </div>

                          {/* Strength slider */}
                          <div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
                              <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Strength</label>
                              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{(config.strength || 1.0).toFixed(2)}</span>
                            </div>
                            <input
                              type="range" min="0" max="2" step="0.05"
                              value={config.strength || 1.0}
                              onChange={(e) => {
                                const newConfigs = [...loraConfigs]
                                newConfigs[idx] = { ...config, strength: parseFloat(e.target.value) }
                                setLoraConfigs(newConfigs)
                              }}
                              style={{ width: '100%', cursor: 'pointer' }}
                            />
                          </div>
                        </div>
                      ))}

                      {/* Add LoRA button */}
                      <button
                        onClick={() => setLoraConfigs([...loraConfigs, { high: '', low: '', strength: 1.0 }])}
                        style={{
                          padding: '8px 12px',
                          backgroundColor: 'transparent',
                          border: '1px dashed var(--border-color)',
                          borderRadius: '6px',
                          color: 'var(--text-secondary)',
                          cursor: 'pointer',
                          fontSize: '0.85rem',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          gap: '6px'
                        }}
                      >
                        + Add LoRA
                      </button>

                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                        💡 Stack multiple LoRAs for combined effects. Each LoRA has its own strength.
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Advanced Settings for Wan2.2 - Always visible, collapsible */}
        {modelMode === 'wan2.2' && (
          <div style={{
            backgroundColor: 'var(--bg-tertiary)',
            padding: '16px',
            borderRadius: '8px',
            marginTop: '8px'
          }}>
            <div
              onClick={() => setShowAdvanced(!showAdvanced)}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                cursor: 'pointer'
              }}
            >
              <div style={{
                fontSize: '0.9rem',
                fontWeight: 600,
                color: 'var(--text-primary)'
              }}>
                ⚙️ Sampling Settings
              </div>
              <span style={{ opacity: 0.5, fontSize: '0.8rem' }}>{showAdvanced ? '▼' : '▶'}</span>
            </div>

            {showAdvanced && (
              <div style={{ marginTop: '12px' }}>
                {/* Steps */}
                <div className="form-group" style={{ marginBottom: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <label className="grok-section-label">Sampling Steps</label>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{steps}</span>
                  </div>
                  <input
                    type="range"
                    min="4"
                    max="20"
                    step="1"
                    value={steps}
                    onChange={(e) => setSteps(parseInt(e.target.value, 10))}
                    style={{ width: '100%', cursor: 'pointer' }}
                  />
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                    <span>4 (fast)</span>
                    <span>6 (rec)</span>
                    <span>20 (quality)</span>
                  </div>
                </div>

                {/* Seed */}
                <div className="form-group">
                  <label className="grok-section-label">Seed</label>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <input
                      type="number"
                      value={seed}
                      onChange={(e) => setSeed(parseInt(e.target.value, 10))}
                      placeholder="-1 for random"
                      style={{
                        flex: 1,
                        padding: '8px 12px',
                        backgroundColor: 'var(--bg-secondary)',
                        border: '1px solid var(--border-color)',
                        borderRadius: '6px',
                        color: 'var(--text-primary)',
                        fontSize: '0.9rem'
                      }}
                    />
                    <button
                      className="btn ghost sm"
                      onClick={() => setSeed(-1)}
                      style={{ whiteSpace: 'nowrap' }}
                    >
                      Random
                    </button>
                  </div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                    -1 = random seed each generation
                  </div>
                </div>
              </div>
            )}

            {/* Post-Processing Settings */}
            <div style={{
              marginTop: '16px',
              paddingTop: '16px',
              borderTop: '1px solid var(--border-color)'
            }}>
              <div
                onClick={() => setShowPostProcessing(!showPostProcessing)}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  cursor: 'pointer',
                  marginBottom: showPostProcessing ? '12px' : 0
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Sparkles size={16} />
                  <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>Post-Processing</span>
                  {(postUpscale || postInterpolate || postAudio) && (
                    <span style={{
                      fontSize: '0.7rem',
                      backgroundColor: 'var(--success-color)',
                      color: 'white',
                      padding: '2px 6px',
                      borderRadius: '4px'
                    }}>
                      {[postUpscale && 'Upscale', postInterpolate && 'RIFE', postAudio && 'Audio'].filter(Boolean).join(' + ')}
                    </span>
                  )}
                </div>
                <span style={{ opacity: 0.5, fontSize: '0.8rem' }}>{showPostProcessing ? '▼' : '▶'}</span>
              </div>

              {showPostProcessing && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {/* Upscale option */}
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '8px 12px',
                    backgroundColor: postUpscale ? 'rgba(var(--success-rgb), 0.1)' : 'var(--bg-secondary)',
                    borderRadius: '8px',
                    border: postUpscale ? '1px solid var(--success-color)' : '1px solid var(--border-color)'
                  }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', flex: 1 }}>
                      <input
                        type="checkbox"
                        checked={postUpscale}
                        onChange={(e) => setPostUpscale(e.target.checked)}
                        style={{ width: '16px', height: '16px' }}
                      />
                      <span>📈 Upscale Video</span>
                    </label>
                    {postUpscale && (
                      <select
                        value={postUpscaleScale}
                        onChange={(e) => setPostUpscaleScale(parseInt(e.target.value))}
                        style={{
                          padding: '4px 8px',
                          backgroundColor: 'var(--bg-tertiary)',
                          border: '1px solid var(--border-color)',
                          borderRadius: '4px',
                          color: 'var(--text-primary)'
                        }}
                      >
                        <option value={2}>2x</option>
                        <option value={4}>4x</option>
                      </select>
                    )}
                  </div>

                  {/* Frame Interpolation option */}
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '8px 12px',
                    backgroundColor: postInterpolate ? 'rgba(var(--success-rgb), 0.1)' : 'var(--bg-secondary)',
                    borderRadius: '8px',
                    border: postInterpolate ? '1px solid var(--success-color)' : '1px solid var(--border-color)'
                  }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', flex: 1 }}>
                      <input
                        type="checkbox"
                        checked={postInterpolate}
                        onChange={(e) => setPostInterpolate(e.target.checked)}
                        style={{ width: '16px', height: '16px' }}
                      />
                      <span>🔄 Smooth Motion (RIFE)</span>
                    </label>
                    {postInterpolate && (
                      <select
                        value={postInterpolateFps}
                        onChange={(e) => setPostInterpolateFps(parseInt(e.target.value))}
                        style={{
                          padding: '4px 8px',
                          backgroundColor: 'var(--bg-tertiary)',
                          border: '1px solid var(--border-color)',
                          borderRadius: '4px',
                          color: 'var(--text-primary)'
                        }}
                      >
                        <option value={30}>30 fps</option>
                        <option value={60}>60 fps</option>
                      </select>
                    )}
                  </div>

                  {/* Add Audio option */}
                  <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '8px',
                    padding: '8px 12px',
                    backgroundColor: postAudio ? 'rgba(var(--success-rgb), 0.1)' : 'var(--bg-secondary)',
                    borderRadius: '8px',
                    border: postAudio ? '1px solid var(--success-color)' : '1px solid var(--border-color)'
                  }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={postAudio}
                        onChange={(e) => {
                          setPostAudio(e.target.checked)
                          if (!e.target.checked) setPostAudioFile(null)
                        }}
                        style={{ width: '16px', height: '16px' }}
                      />
                      <span>🔊 Add Audio Track</span>
                    </label>
                    {postAudio && (
                      <input
                        type="file"
                        accept="audio/*"
                        onChange={(e) => setPostAudioFile(e.target.files?.[0] || null)}
                        style={{
                          fontSize: '0.8rem',
                          color: 'var(--text-muted)'
                        }}
                      />
                    )}
                  </div>

                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                    💡 Post-processing runs as chained jobs after video generation completes
                  </div>
                </div>
              )}
            </div>

            {/* LoRA Settings */}
            <div style={{
              marginTop: '16px',
              paddingTop: '16px',
              borderTop: '1px solid var(--border-color)'
            }}>
              <div
                onClick={() => setShowLoraPanel(!showLoraPanel)}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  cursor: 'pointer',
                  marginBottom: showLoraPanel ? '12px' : 0
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Layers size={16} />
                  <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>LoRA Models</span>
                  {loraConfigs.length > 0 && (
                    <span style={{
                      fontSize: '0.7rem',
                      backgroundColor: 'var(--accent-color)',
                      color: 'white',
                      padding: '2px 6px',
                      borderRadius: '4px'
                    }}>
                      {loraConfigs.length} active
                    </span>
                  )}
                </div>
                <span style={{ opacity: 0.5, fontSize: '0.8rem' }}>{showLoraPanel ? '▼' : '▶'}</span>
              </div>

              {showLoraPanel && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {/* Existing LoRAs */}
                  {loraConfigs.map((config, idx) => (
                    <div key={idx} style={{
                      backgroundColor: 'var(--bg-input)',
                      borderRadius: '8px',
                      padding: '12px',
                      border: '1px solid var(--border-color)'
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                        <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>LoRA #{idx + 1}</span>
                        <button
                          onClick={() => setLoraConfigs(loraConfigs.filter((_, i) => i !== idx))}
                          style={{
                            background: 'transparent',
                            border: 'none',
                            color: '#ef4444',
                            cursor: 'pointer',
                            padding: '2px 6px',
                            fontSize: '0.8rem'
                          }}
                        >
                          ✕ Remove
                        </button>
                      </div>

                      {/* High Noise LoRA */}
                      <div style={{ marginBottom: '8px' }}>
                        <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
                          High Noise (steps 0-3)
                        </label>
                        <select
                          value={config.high || ''}
                          onChange={(e) => {
                            const newConfigs = [...loraConfigs]
                            newConfigs[idx] = { ...config, high: e.target.value }
                            setLoraConfigs(newConfigs)
                          }}
                          style={{
                            width: '100%',
                            padding: '6px 10px',
                            backgroundColor: 'var(--bg-secondary)',
                            border: '1px solid var(--border-color)',
                            borderRadius: '4px',
                            color: 'var(--text-primary)',
                            fontSize: '0.8rem'
                          }}
                        >
                          <option value="">None</option>
                          {filteredLoras.by_category && Object.keys(filteredLoras.by_category).sort().map((category) => (
                            <optgroup key={category} label={category === 'root' ? '📁 Root' : `📁 ${category}`}>
                              {filteredLoras.by_category[category].map((lora) => (
                                <option key={lora.path} value={lora.path}>
                                  {lora.name} ({lora.size_mb}MB)
                                </option>
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
                          onChange={(e) => {
                            const newConfigs = [...loraConfigs]
                            newConfigs[idx] = { ...config, low: e.target.value }
                            setLoraConfigs(newConfigs)
                          }}
                          style={{
                            width: '100%',
                            padding: '6px 10px',
                            backgroundColor: 'var(--bg-secondary)',
                            border: '1px solid var(--border-color)',
                            borderRadius: '4px',
                            color: 'var(--text-primary)',
                            fontSize: '0.8rem'
                          }}
                        >
                          <option value="">None (uses High Noise)</option>
                          {filteredLoras.by_category && Object.keys(filteredLoras.by_category).sort().map((category) => (
                            <optgroup key={category} label={category === 'root' ? '📁 Root' : `📁 ${category}`}>
                              {filteredLoras.by_category[category].map((lora) => (
                                <option key={lora.path} value={lora.path}>
                                  {lora.name} ({lora.size_mb}MB)
                                </option>
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
                          type="range"
                          min="0"
                          max="2"
                          step="0.05"
                          value={config.strength || 1.0}
                          onChange={(e) => {
                            const newConfigs = [...loraConfigs]
                            newConfigs[idx] = { ...config, strength: parseFloat(e.target.value) }
                            setLoraConfigs(newConfigs)
                          }}
                          style={{ width: '100%', cursor: 'pointer' }}
                        />
                      </div>
                    </div>
                  ))}

                  {/* Add LoRA button */}
                  <button
                    onClick={() => setLoraConfigs([...loraConfigs, { high: '', low: '', strength: 1.0 }])}
                    style={{
                      padding: '8px 12px',
                      backgroundColor: 'transparent',
                      border: '1px dashed var(--border-color)',
                      borderRadius: '6px',
                      color: 'var(--text-secondary)',
                      cursor: 'pointer',
                      fontSize: '0.85rem',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '6px'
                    }}
                  >
                    + Add LoRA
                  </button>

                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                    💡 Stack multiple LoRAs for combined effects. Each LoRA has its own strength.
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        <div className="form-group" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div className="grok-section-label" style={{ marginBottom: '4px' }}>Generate Audio</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Enable audio generation (increases credits)</div>
          </div>
          <label className="grok-switch">
            <input type="checkbox" />
            <span className="grok-slider"></span>
          </label>
        </div>

        <div className="form-group" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div className="grok-section-label" style={{ marginBottom: '4px' }}>Camera Fixed</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Whether to fix the camera position</div>
          </div>
          <label className="grok-switch">
            <input type="checkbox" />
            <span className="grok-slider"></span>
          </label>
        </div>

        {/* Extend Duration - Sequential Clips */}
        <div className="form-group" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div className="grok-section-label" style={{ marginBottom: '4px' }}>🎬 Extend Duration</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Chain multiple clips sequentially</div>
          </div>
          <label className="grok-switch">
            <input
              type="checkbox"
              checked={extendMode}
              onChange={(e) => {
                setExtendMode(e.target.checked)
                if (!e.target.checked) setClipCount(1)
              }}
            />
            <span className="grok-slider"></span>
          </label>
        </div>

        {/* Clip Count Slider - Only visible when extendMode is on */}
        {extendMode && (
          <div className="form-group" style={{
            background: 'linear-gradient(135deg, rgba(233, 69, 96, 0.1) 0%, rgba(233, 69, 96, 0.05) 100%)',
            borderRadius: '8px',
            padding: '12px',
            marginTop: '-8px',
            border: '1px solid rgba(233, 69, 96, 0.2)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <div className="grok-section-label">Number of Clips: {clipCount}</div>
              <div style={{
                fontSize: '0.75rem',
                color: '#e94560',
                background: 'rgba(233, 69, 96, 0.15)',
                padding: '2px 8px',
                borderRadius: '10px',
                fontWeight: '600'
              }}>
                ≈ {(duration * clipCount).toFixed(0)}s total
              </div>
            </div>
            <input
              type="range"
              min="1"
              max="5"
              value={clipCount}
              onChange={(e) => setClipCount(parseInt(e.target.value))}
              style={{
                width: '100%',
                accentColor: '#e94560'
              }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              <span>1</span>
              <span>2</span>
              <span>3</span>
              <span>4</span>
              <span>5</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '8px', fontStyle: 'italic' }}>
              🔗 Each clip continues from the last frame of the previous clip
            </div>
          </div>
        )}

      </div>

      {/* Aspect Ratio */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Aspect Ratio</div>
        </div>
        <div className="aspect-grid">
          {[
            { label: 'Auto', icon: <Sparkles size={16} /> },
            { label: '21:9', icon: <div style={{ width: '24px', height: '10px', border: '1px solid currentColor' }} /> },
            { label: '16:9', icon: <div style={{ width: '24px', height: '14px', border: '1px solid currentColor' }} /> },
            { label: '4:3', icon: <div style={{ width: '20px', height: '15px', border: '1px solid currentColor' }} /> },
            { label: '1:1', icon: <div style={{ width: '18px', height: '18px', border: '1px solid currentColor' }} /> },
            { label: '3:4', icon: <div style={{ width: '15px', height: '20px', border: '1px solid currentColor' }} /> },
            { label: '9:16', icon: <div style={{ width: '14px', height: '24px', border: '1px solid currentColor' }} /> },
          ].map((ratio) => (
            <button
              key={ratio.label}
              className={`aspect-btn ${aspectRatio === ratio.label ? 'active' : ''}`}
              onClick={() => setAspectRatio(ratio.label)}
            >
              <div className="aspect-icon" style={{ background: 'transparent', display: 'flex', alignItems: 'center', justifyContent: 'center', border: 'none' }}>
                {ratio.icon}
              </div>
              <span className="aspect-label">{ratio.label}</span>
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div style={{
          padding: '12px',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.2)',
          borderRadius: '8px',
          color: '#ef4444',
          marginBottom: '16px',
          fontSize: '0.9rem'
        }}>
          {error}
        </div>
      )}

      {/* Time estimate indicator */}
      {!busy && canSubmit && (
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
          color: 'black'
        }}
      >
        {busy ? (
          <>Generating...</>
        ) : (
          <>
            <Sparkles size={18} />
            Generate from Image
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
