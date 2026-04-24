import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react'
import { Upload, Wand2, Loader2, Image as ImageIcon, Settings, ChevronDown, Sliders, X, Zap, Shield, User as UserIcon, Sparkles, Palette } from 'lucide-react'
import InfoTooltip from '../../components/InfoTooltip'
import { BACKEND_BASE, DEBUG, getMediaUrl } from '../../config'
import { apiFetch } from '../../api'
import useGeneration from '../../hooks/useGeneration'
import { extractVideoFirstFrame } from '../../utils/mediaUtils'
import { useAuth } from '../../contexts/AuthContext'
import MediaImportModal from '../../components/MediaImportModal'
import CreationsPickerModal from '../../components/CreationsPickerModal'
import { useToolSettings } from '../../hooks/useToolSettings'
import ResetDefaultsButton from '../../components/ResetDefaultsButton'

const CHECKPOINTS = [
  { value: 'CyberRealistic_Pony_v14.1_FP16.safetensors', label: 'CyberRealistic Pony' },
  { value: 'dreamshaperXL_lightningDPMSDE.safetensors', label: 'Dreamshaper Lightning' },
  { value: 'juggernautXL_ragnarok.safetensors', label: 'Juggernaut XL' },
  { value: 'Realistic_Vision_V5.1.safetensors', label: 'Realistic Vision V5.1' },
  { value: 'waiIllustriousSDXL_v160.safetensors', label: 'Wai Illustrious (Anime)' },
  { value: 'ponyDiffusionV6XL_v6StartWithThisOne.safetensors', label: 'Pony Diffusion V6' },
  { value: 'illustriousRealismBy_v10VAE.safetensors', label: 'Illustrious Realism' },
  { value: 'reapony_v90.safetensors', label: 'ReaPony V9' },
]

const PRESETS = [
  { value: 'fast', label: 'Fast', icon: '⚡', desc: 'Quick transform, no face processing', color: '#22c55e' },
  { value: 'balanced', label: 'Balanced', icon: '⚖️', desc: 'Good quality + face refinement', color: '#3b82f6' },
  { value: 'face_preserve', label: 'Face Preserve', icon: '🛡️', desc: 'Best for keeping faces consistent', color: '#a855f7' },
  { value: 'custom', label: 'Custom', icon: '🔧', desc: 'Full manual control', color: '#f59e0b' },
]

const I2I_DEFAULTS = {
  mode: 'transform',
  engine: 'local',
  // Shared
  negativePrompt: 'ugly, deformed, blurry, low quality, bad anatomy, watermark',
  steps: 25, cfg: 7.0, seed: -1,
  // Transform mode
  prompt: '',
  denoise: 0.6, checkpoint: 'CyberRealistic_Pony_v14.1_FP16.safetensors', preset: 'balanced',
  faceId: false, faceDetailer: true, faceRestore: true, faceIdWeight: 0.85,
  sampler: 'dpmpp_2m', scheduler: 'karras',
  // Edit mode
  instruction: '',
  lightning: false,
  editModel: 'default',
  editAspectRatio: '1:1',
  editResolution: '1024',
  loraConfigs: [],
}

const EDIT_RESOLUTION_PRESETS = {
  '768':  { label: '768',  desc: 'Fast' },
  '1024': { label: '1024', desc: 'Standard' },
  '1280': { label: '1280', desc: 'High' },
  '1536': { label: '1536', desc: 'Ultra' },
}

const EDIT_ASPECT_RATIOS = ['1:1', '16:9', '9:16', '4:3', '3:4', '2:3', '3:2']

const EDIT_MODEL_OPTIONS = [
  {
    value: 'default',
    label: 'I2I Edit 2511 Default',
    desc: 'Official I2I Edit Image Edit fp8mixed model',
  },
]

// Calculate width × height from base resolution + aspect ratio (clamped to multiples of 16)
function getEditDimensions(baseRes, aspectRatio) {
  const base = parseInt(baseRes)
  const [aw, ah] = aspectRatio.split(':').map(Number)
  const ratio = aw / ah
  let w, h
  if (ratio >= 1) {
    w = base
    h = Math.round(base / ratio)
  } else {
    h = base
    w = Math.round(base * ratio)
  }
  // Clamp to multiples of 16
  w = Math.round(w / 16) * 16
  h = Math.round(h / 16) * 16
  // Clamp to backend limits
  w = Math.max(512, Math.min(2048, w))
  h = Math.max(512, Math.min(2048, h))
  return { width: w, height: h }
}

const EXAMPLE_INSTRUCTIONS = [
  'Remove the background',
  'Make it anime style',
  'Change hair color to blonde',
  'Add sunglasses',
  'Turn this into a watercolor painting',
  'Make it look like a pencil sketch',
  'Remove the person on the left',
  'Change the sky to sunset',
  'Make their outfit red',
  'Add a hat',
]

export default function ImageToImageTool({ onOutput, onJobSubmitted, pendingImport, onImportConsumed }) {
  const { user, requestLogin } = useAuth()
  const { initial, save: saveSettings, resetDefaults } = useToolSettings('image_to_image', I2I_DEFAULTS)

  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const fileInputKey = useRef(0)
  const fileRef = useRef(null)  // Authoritative file reference (bypasses React state batching)
  const [importModal, setImportModal] = useState(null)
  const [showCreationsPicker, setShowCreationsPicker] = useState(false)
  // ── Mode ─────────────────────────────────────────────────────────
  const [mode, setMode] = useState(initial.mode || 'transform')
  const [engine, setEngine] = useState(initial.engine || 'local')

  // ── Transform mode state ────────────────────────────────────────
  const [prompt, setPrompt] = useState(initial.prompt)
  const [negativePrompt, setNegativePrompt] = useState(initial.negativePrompt)
  const [denoise, setDenoise] = useState(initial.denoise)
  const [checkpoint, setCheckpoint] = useState(initial.checkpoint)
  const [preset, setPreset] = useState(initial.preset)
  const [faceId, setFaceId] = useState(initial.faceId)
  const [faceDetailer, setFaceDetailer] = useState(initial.faceDetailer)
  const [faceRestore, setFaceRestore] = useState(initial.faceRestore)
  const [faceIdWeight, setFaceIdWeight] = useState(initial.faceIdWeight)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [steps, setSteps] = useState(initial.steps)
  const [cfg, setCfg] = useState(initial.cfg)
  const [seed, setSeed] = useState(initial.seed)
  const [sampler, setSampler] = useState(initial.sampler)
  const [scheduler, setScheduler] = useState(initial.scheduler)

  // ── Edit mode state ─────────────────────────────────────────────
  const [instruction, setInstruction] = useState(initial.instruction || '')
  const [lightning, setLightning] = useState(initial.lightning || false)
  const [editModel, setI2IModel] = useState(initial.editModel || 'default')
  const [editAspectRatio, setEditAspectRatio] = useState(initial.editAspectRatio || '1:1')
  const [editResolution, setEditResolution] = useState(initial.editResolution || '1024')
  const [availableLoras, setAvailableLoras] = useState({ by_category: {} })
  const [loraConfigs, setLoraConfigs] = useState(initial.loraConfigs || [])
  const [showLoraPanel, setShowLoraPanel] = useState(false)

  // ── Auto-save settings ──────────────────────────────────────────
  const settingsSnapshot = useMemo(() => ({
    mode, engine, prompt, negativePrompt, denoise, checkpoint, preset,
    faceId, faceDetailer, faceRestore, faceIdWeight,
    steps, cfg, seed, sampler, scheduler,
    instruction, lightning, editModel, editAspectRatio, editResolution, loraConfigs,
  }), [mode, engine, prompt, negativePrompt, denoise, checkpoint, preset,
    faceId, faceDetailer, faceRestore, faceIdWeight,
    steps, cfg, seed, sampler, scheduler,
    instruction, lightning, editModel, editAspectRatio, editResolution, loraConfigs])
  useEffect(() => { saveSettings(settingsSnapshot) }, [settingsSnapshot, saveSettings])

  const handleResetDefaults = useCallback(() => {
    const d = resetDefaults()
    setMode(d.mode || 'transform')
    setEngine(d.engine || 'local')
    setEngine(d.engine || 'local')
    setPrompt(d.prompt); setNegativePrompt(d.negativePrompt); setDenoise(d.denoise)
    setCheckpoint(d.checkpoint); setPreset(d.preset)
    setFaceId(d.faceId); setFaceDetailer(d.faceDetailer); setFaceRestore(d.faceRestore)
    setFaceIdWeight(d.faceIdWeight)
    setSteps(d.steps); setCfg(d.cfg); setSeed(d.seed); setSampler(d.sampler); setScheduler(d.scheduler)
    setInstruction(d.instruction || ''); setLightning(d.lightning || false)
    setI2IModel(d.editModel || 'default')
    setEditAspectRatio(d.editAspectRatio || '1:1')
    setEditResolution(d.editResolution || '1024')
    setLoraConfigs(d.loraConfigs || [])
  }, [resetDefaults])

  // ── Fetch LoRAs for edit mode ───────────────────────────────────
  useEffect(() => {
    const fetchLoras = async () => {
      try {
        const res = await apiFetch('/loras')
        if (res.ok) {
          const data = await res.json()
          setAvailableLoras(data)
          if (DEBUG) console.debug('🎨 I2I loaded LoRAs:', data.count)
        }
      } catch (e) {
        console.error('Failed to fetch LoRAs:', e)
      }
    }
    fetchLoras()
  }, [])

  const filteredLoras = useMemo(() => {
    const result = {}
    if (availableLoras.by_category) {
      Object.keys(availableLoras.by_category).sort().forEach(cat => {
        const items = availableLoras.by_category[cat] || []
        if (items.length > 0) result[cat] = items
      })
    }
    return result
  }, [availableLoras])

  // ── Lightning mode auto-adjusts steps/cfg ───────────────────────
  useEffect(() => {
    if (mode !== 'edit') return
    if (lightning) {
      setSteps(4); setCfg(1.0)
    } else {
      setSteps(40); setCfg(4.0)
    }
  }, [lightning, mode])

  const [error, setError] = useState(null)
  const [lastQueued, setLastQueued] = useState(null)
  const [result, setResult] = useState(null)

  const { generate, loading: submitting } = useGeneration({
    onSuccess: (data, req) => {
      const promptId = data.prompt_id || data.id
      let qMode = req.adapter_hint === 'cloud-i2i-edit' ? 'edit' : 'transform'
      setLastQueued({
        promptId: promptId,
        mode: qMode,
        checkpoint: qMode === 'transform' ? (CHECKPOINTS.find(c => c.value === req.checkpoint)?.label || req.checkpoint) : undefined,
        editModel: qMode === 'edit' ? (EDIT_MODEL_OPTIONS.find(c => c.value === (req.edit_model || 'default'))?.label || req.edit_model || 'I2I Edit 2511 Default') : undefined,
        lightning: req.lightning,
        runpodJobId: data.runpod_job_id,
        credits: data.credits_used
      })
      if (onJobSubmitted) onJobSubmitted({ prompt_id: promptId })
      if (onOutput) onOutput(data)
    },
    onError: (err) => setError(err)
  })

  // Update settings when preset changes
  useEffect(() => {
    if (preset === 'fast') {
      setSteps(15); setCfg(7.0); setSampler('dpmpp_2m'); setScheduler('karras')
      setFaceId(false); setFaceDetailer(false); setFaceRestore(false)
    } else if (preset === 'balanced') {
      setSteps(25); setCfg(7.0); setSampler('dpmpp_2m'); setScheduler('karras')
      setFaceId(false); setFaceDetailer(true); setFaceRestore(true)
    } else if (preset === 'face_preserve') {
      setSteps(30); setCfg(7.5); setSampler('dpmpp_2m_sde'); setScheduler('karras')
      setFaceId(true); setFaceDetailer(true); setFaceRestore(true)
    }
    // custom = no auto-change
  }, [preset])

  // Auto-open import modal when Dashboard sends a pendingImport
  useEffect(() => {
    if (!pendingImport) return
    setImportModal(pendingImport)
    if (onImportConsumed) onImportConsumed()
  }, [pendingImport])

  // Helper: set file in both state (for UI) and ref (for submission)
  const updateFile = useCallback((f, previewUrl) => {
    fileRef.current = f
    setFile(f)
    setPreview(previewUrl || (f ? URL.createObjectURL(f) : null))
    setResult(null)
    setError(null)
    setLastQueued(null)
    console.warn(`🖼️ I2I file set: ${f?.name || 'null'} (${f ? (f.size / 1024).toFixed(0) + 'KB' : '-'}, type=${f?.type || '-'})`)
  }, [])

  const handleApplyImport = async (selected) => {
    if (selected.image && importModal?.item) {
      const item = importModal.item

      // If the sender passed a direct File blob (e.g. from Image-to-Text), use it directly
      if (item._file instanceof File) {
        const previewUrl = item.url?.startsWith('blob:') ? item.url : URL.createObjectURL(item._file)
        updateFile(item._file, previewUrl)
      } else {
        // Fetch from URL (standard MyMedia flow)
        if (item.type === 'video' && item.filename?.match(/\.(mp4|webm|mov)$/i)) {
          try {
            const fetchUrl = item.signed_url || (item.url?.startsWith('/') ? item.url : `/${item.url}`)
            console.debug('🎬 I2I: video detected, extracting first frame:', item.filename)
            const { file: fileObj, previewUrl } = await extractVideoFirstFrame(apiFetch, fetchUrl, item.filename)
            updateFile(fileObj, previewUrl)
          } catch (e) {
            console.error('Failed to extract frame from video:', e)
            setError('⚠️ Failed to extract first frame from video')
          }
        } else {
          const imageUrl = getMediaUrl(item.url, item.signed_url)
          const imageFilename = item.filename || imageUrl.split('/').pop()

          try {
            const response = await apiFetch(imageUrl)
            if (!response.ok) throw new Error(`Failed to fetch image: ${response.status}`)
            const blob = await response.blob()
            const fileObj = new File([blob], imageFilename, { type: blob.type || 'image/png' })
            updateFile(fileObj, imageUrl)
          } catch (e) {
            console.error('Failed to load image from import:', e)
            setError('⚠️ Failed to load image from import')
          }
        }
      }
    }
    if (selected.positive)  setPrompt(String(selected.positive))
    if (selected.negative)  setNegativePrompt(String(selected.negative))
    if (selected.steps)     setSteps(Number(selected.steps) || steps)
    if (selected.cfg)       setCfg(Number(selected.cfg) || cfg)
    if (selected.seed)      setSeed(Number(selected.seed) || seed)
    setImportModal(null)
  }

  const handleCreationsSelect = useCallback(async (item) => {
    try {
      if (item.type === 'video' && item.filename?.match(/\.(mp4|webm|mov)$/i)) {
        const fetchUrl = item.signed_url || (item.url?.startsWith('/') ? item.url : `/${item.url}`)
        console.debug('🎬 I2I creations: extracting first frame from video:', item.filename)
        const { file: fileObj, previewUrl } = await extractVideoFirstFrame(apiFetch, fetchUrl, item.filename)
        updateFile(fileObj, previewUrl)
      } else {
        const imageUrl = getMediaUrl(item.url, item.signed_url)
        const imageFilename = item.filename || imageUrl.split('/').pop()
        const response = await apiFetch(imageUrl)
        if (!response.ok) throw new Error(`Failed to fetch: ${response.status}`)
        const blob = await response.blob()
        const fileObj = new File([blob], imageFilename, { type: blob.type || 'image/png' })
        updateFile(fileObj, imageUrl)
      }
    } catch (e) {
      console.error('Failed to load from creations:', e)
      setError('⚠️ Failed to load image from My Creations')
    }
  }, [updateFile])

  const handleFileChange = useCallback((e) => {
    const f = e.target.files?.[0]
    if (f) {
      updateFile(f, URL.createObjectURL(f))
    }
    // Reset input value so re-selecting the same file triggers onChange
    e.target.value = ''
  }, [updateFile])

  const handleClearImage = useCallback((e) => {
    e?.stopPropagation()
    updateFile(null, null)
    fileInputKey.current += 1
  }, [updateFile])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0]
    if (f && f.type.startsWith('image/')) {
      updateFile(f, URL.createObjectURL(f))
    }
  }, [updateFile])

  const handleGenerate = async () => {
    if (!user) {
      requestLogin('Log in om te genereren')
      return
    }
    if (!file) return
    if (mode === 'edit' && !instruction.trim()) {
      setError('Please describe what you want to change')
      return
    }

    setError(null)
    setLastQueued(null)

    try {
      const currentFile = fileRef.current
      if (!currentFile) {
        setError('No file selected')
        return
      }

      const reader = new FileReader()
      reader.onload = async () => {
        const base64Data = reader.result.split(',')[1]

        const reqPayload = {
          operation: 'transform',
          target_type: 'image',
          input_images: [base64Data],
          seed: seed,
          steps: steps,
          cfg: cfg,
        }

        if (mode === 'transform') {
          // ── Transform mode ──
          reqPayload.adapter_hint = engine === 'cloud' ? 'cloud-i2i-transform' : 'local-i2i-transform'
          reqPayload.prompt = prompt || 'high quality, detailed'
          reqPayload.negative_prompt = negativePrompt
          reqPayload.denoise = denoise
          reqPayload.checkpoint = checkpoint
          reqPayload.sampler_name = sampler
          reqPayload.scheduler = scheduler
          reqPayload.preset = preset
          reqPayload.face_id = faceId
          reqPayload.face_detailer = faceDetailer
          reqPayload.face_restore = faceRestore
          reqPayload.face_id_weight = faceIdWeight

          if (DEBUG) console.debug('🖼️ I2I transform:', { denoise, checkpoint, steps })
          await generate(reqPayload)

        } else {
          // ── Edit mode → RunPod cloud ──
          reqPayload.adapter_hint = 'cloud-i2i-edit'
          reqPayload.instruction = instruction
          reqPayload.negative_prompt = negativePrompt
          reqPayload.lightning = lightning
          reqPayload.edit_model = editModel
          
          const editDims = getEditDimensions(editResolution, editAspectRatio)
          reqPayload.width = editDims.width
          reqPayload.height = editDims.height
          
          if (loraConfigs.length > 0) {
            reqPayload.lora_configs = loraConfigs.filter(c => c.name)
          }

          if (DEBUG) console.debug('✏️ I2I edit:', { instruction, steps, cfg, lightning, editModel, loras: loraConfigs.length })
          await generate(reqPayload)
        }
      }
      reader.onerror = () => {
        setError('Failed to read file')
      }
      reader.readAsDataURL(currentFile)

    } catch (err) {
      console.error('I2I error:', err)
      setError(err.message || 'Failed to start generation')
    }
  }

  return (
    <div className="tool-container">
      {/* Source Image Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <ImageIcon size={16} />
            Source Image
          </div>
          <ResetDefaultsButton onReset={handleResetDefaults} />
        </div>

        <div
          className={`upload-dropzone ${preview ? 'has-preview' : ''}`}
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => document.getElementById('i2i-file-input').click()}
        >
          {preview ? (
            <>
              <img src={preview} alt="Preview" className="upload-preview" />
              <button
                className="btn-clear-image"
                onClick={handleClearImage}
                title="Clear image"
              >
                <X size={16} />
              </button>
              {file && (
                <div className="upload-filename">
                  📎 {file.name} ({(file.size / 1024).toFixed(0)}KB)
                </div>
              )}
            </>
          ) : (
            <div className="upload-placeholder">
              <Upload size={32} />
              <p>Drop image here or click to upload</p>
            </div>
          )}
          <input
            key={fileInputKey.current}
            id="i2i-file-input"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
        </div>

        <button
          onClick={() => setShowCreationsPicker(true)}
          className="btn-creations-picker"
        >
          📁 From My Creations
        </button>

        <CreationsPickerModal
          show={showCreationsPicker}
          onClose={() => setShowCreationsPicker(false)}
          onSelect={handleCreationsSelect}
          filter="image"
          title="Select Image from My Creations"
        />
      </div>

      {/* Mode Selector */}
      <div className="grok-card" style={{ padding: '4px', overflow: 'hidden' }}>
        <div style={{ display: 'flex', gap: '4px' }}>
          <button
            onClick={() => setMode('transform')}
            className={`mode-btn ${mode === 'transform' ? 'active' : ''}`}
          >
            <Wand2 size={14} />
            Transform
            
          </button>
          <button
            onClick={() => setMode('edit')}
            className={`mode-btn ${mode === 'edit' ? 'active' : ''}`}
          >
            <Sparkles size={14} />
            AI Edit
            <span className="mode-hint">Cloud · I2I</span>
          </button>
        </div>
      </div>

      {/* Engine Selector for Transform Mode */}
      {mode === 'transform' && (
        <div className="grok-card" style={{ padding: '0.75rem', marginBottom: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            <button
              onClick={() => setEngine('local')}
              className={`mode-btn ${engine === 'local' ? 'active' : ''}`}
            >
              <Sliders size={14} />
              Local Engine
              <span className="mode-hint">SDXL with Face Preserve</span>
            </button>
            <button
              onClick={() => setEngine('cloud')}
              className={`mode-btn ${engine === 'cloud' ? 'active' : ''}`}
            >
              <Zap size={14} />
              Cloud Engine
              <span className="mode-hint">RunPod Transform (Fast)</span>
            </button>
          </div>
          {engine === 'cloud' && (
            <div className="text-xs text-blue-400 bg-blue-900/20 p-2 rounded-md border border-blue-800/50">
              ⚡ Cloud engine runs on RunPod I2I worker. It skips face-detailer pipelines for faster generation.
            </div>
          )}
        </div>
      )}

      {/* ════════ TRANSFORM MODE ════════ */}
      {mode === 'transform' && (<>

      {/* Transformation Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Wand2 size={16} />
            Transformation
          </div>
        </div>

        <div className="form-group">
          <label className="grok-section-label">Prompt (describe desired changes) <InfoTooltip text="Describe what you want the transformed image to look like. Be specific — e.g., 'anime style illustration' or 'oil painting with warm tones'. The AI uses this along with the denoise strength to guide the transformation." /></label>
          <textarea
            className="form-textarea"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            placeholder="Describe what you want the image to become... (e.g., 'anime style illustration')"
          />
        </div>

        <div className="form-group">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <label className="grok-section-label" style={{ marginBottom: 0 }}>Denoise Strength <InfoTooltip text="How much freedom the AI has to change the original image. Low values (0.1-0.3) make subtle tweaks, high values (0.7-1.0) can completely reimagine the image from your prompt." /></label>
            <span className="nav-badge">{denoise.toFixed(2)}</span>
          </div>
          <input
            type="range"
            className="form-range"
            min="0.1"
            max="1.0"
            step="0.05"
            value={denoise}
            onChange={(e) => setDenoise(parseFloat(e.target.value))}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            <span>Subtle (0.1)</span>
            <span>Complete (1.0)</span>
          </div>
          <div style={{ marginTop: '6px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            {denoise < 0.3 && '💡 Minor adjustments, preserves most of original'}
            {denoise >= 0.3 && denoise < 0.6 && '💡 Moderate changes, good balance'}
            {denoise >= 0.6 && denoise < 0.8 && '💡 Significant transformation'}
            {denoise >= 0.8 && '💡 Near-complete regeneration from prompt'}
          </div>
        </div>

        <div className="form-group">
          <label className="grok-section-label">Model <InfoTooltip text="The checkpoint model determines the base art style and capabilities. SDXL models offer high quality with more control, while Flux models are faster." /></label>
          <select className="form-select" value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)}>
            {CHECKPOINTS.map((c) => (
              <option key={c.value} value={c.value}>{c.label}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Quality Preset Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Zap size={16} />
            Quality Preset
          </div>
        </div>
        <div className="preset-grid">
          {PRESETS.map((p) => (
            <button
              key={p.value}
              className={`preset-card ${preset === p.value ? 'active' : ''}`}
              onClick={() => setPreset(p.value)}
              style={preset === p.value ? { borderColor: p.color, boxShadow: `0 0 12px ${p.color}33` } : {}}
            >
              <span className="preset-icon">{p.icon}</span>
              <span className="preset-label">{p.label}</span>
              <span className="preset-desc">{p.desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Face Processing Card - Only shown in Local Engine */}
      {engine === 'local' && (
        <div className="grok-card">
          <div className="grok-card-header">
            <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <UserIcon size={16} />
              Face Processing
            </div>
          </div>

        <div className="face-toggles">
          <label className="toggle-row">
            <div className="toggle-info">
              <span className="toggle-label">🛡️ Face Identity (IP-Adapter) <InfoTooltip text="IP-Adapter FaceID Plus V2: Extracts face identity from the source image and preserves it during generation. Great for keeping someone's likeness while changing the style." size={12} /></span>
              <span className="toggle-desc">Preserves face from source image</span>
            </div>
            <input
              type="checkbox"
              checked={faceId}
              onChange={(e) => { setFaceId(e.target.checked); if (preset !== 'custom') setPreset('custom') }}
            />
            <span className="toggle-slider" />
          </label>

          {faceId && (
            <div className="form-group face-weight-slider">
              <label>
                FaceID Strength
                <span className="label-value">{faceIdWeight.toFixed(2)}</span>
              </label>
              <input
                type="range"
                min="0.3"
                max="1.0"
                step="0.05"
                value={faceIdWeight}
                onChange={(e) => setFaceIdWeight(parseFloat(e.target.value))}
              />
              <div className="range-labels">
                <span>Subtle (0.3)</span>
                <span>Strong (1.0)</span>
              </div>
            </div>
          )}

          <label className="toggle-row">
            <div className="toggle-info">
              <span className="toggle-label">✨ Face Detailer <InfoTooltip text="FaceDetailer: Auto-detects faces using YOLO and refines them with a second diffusion pass using SAM segmentation. Greatly improves face quality in complex scenes." size={12} /></span>
              <span className="toggle-desc">Auto-detect &amp; refine faces</span>
            </div>
            <input
              type="checkbox"
              checked={faceDetailer}
              onChange={(e) => { setFaceDetailer(e.target.checked); if (preset !== 'custom') setPreset('custom') }}
            />
            <span className="toggle-slider" />
          </label>

          <label className="toggle-row">
            <div className="toggle-info">
              <span className="toggle-label">💎 Face Restore (GFPGAN) <InfoTooltip text="GFPGAN v1.4: Final face quality enhancement pass. Sharpens eyes, skin texture, and facial features for photorealistic results. Best used as a finishing touch." size={12} /></span>
              <span className="toggle-desc">Final polish on face quality</span>
            </div>
            <input
              type="checkbox"
              checked={faceRestore}
              onChange={(e) => { setFaceRestore(e.target.checked); if (preset !== 'custom') setPreset('custom') }}
            />
            <span className="toggle-slider" />
          </label>
        </div>
      </div>
      )}

      {/* Advanced Settings Card */}
      <div className="grok-card" style={{ padding: 0, overflow: 'hidden' }}>
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          style={{
            width: '100%', padding: '14px 20px', background: 'transparent', border: 'none',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            color: 'var(--text-secondary)', cursor: 'pointer',
          }}
        >
          <span style={{ fontSize: '0.85rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Settings size={16} />
            Advanced Settings
          </span>
          <ChevronDown size={16} style={{ transition: 'transform 0.2s', transform: showAdvanced ? 'rotate(180deg)' : 'none' }} />
        </button>

        {showAdvanced && (
          <div style={{ padding: '0 20px 20px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div className="form-group">
              <label className="grok-section-label">Negative Prompt <InfoTooltip text="Describe what you DON'T want in the output. Common entries: blurry, low quality, distorted, extra limbs, watermark, text." /></label>
              <textarea
                className="form-textarea"
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
                rows={2}
              />
            </div>

            <div style={{ display: 'flex', gap: '12px' }}>
              <div style={{ flex: 1 }}>
                <label className="grok-section-label">Steps <InfoTooltip text="Number of diffusion sampling steps. More steps = higher quality but slower. 20-30 steps is optimal for most use cases." /></label>
                <input
                  className="form-input"
                  type="number"
                  value={steps}
                  onChange={(e) => setSteps(parseInt(e.target.value) || 25)}
                  min="1"
                  max="50"
                />
              </div>
              <div style={{ flex: 1 }}>
                <label className="grok-section-label">CFG Scale <InfoTooltip text="Classifier-Free Guidance — how strictly the AI follows your prompt vs being creative. Higher = more prompt adherence. 5-8 is a good balance." /></label>
                <input
                  className="form-input"
                  type="number"
                  value={cfg}
                  onChange={(e) => setCfg(parseFloat(e.target.value) || 7.0)}
                  min="1"
                  max="20"
                  step="0.5"
                />
              </div>
            </div>

            <div style={{ display: 'flex', gap: '12px' }}>
              <div style={{ flex: 1 }}>
                <label className="grok-section-label">Sampler <InfoTooltip text="The algorithm used for the diffusion process. Euler is fast and reliable, DPM++ 2M produces high-quality results, Ancestral variants add more randomness." /></label>
                <select className="form-select" value={sampler} onChange={(e) => setSampler(e.target.value)}>
                  <option value="euler">Euler</option>
                  <option value="euler_ancestral">Euler Ancestral</option>
                  <option value="dpmpp_2m">DPM++ 2M</option>
                  <option value="dpmpp_2m_sde">DPM++ 2M SDE</option>
                  <option value="dpmpp_3m_sde">DPM++ 3M SDE</option>
                </select>
              </div>
              <div style={{ flex: 1 }}>
                <label className="grok-section-label">Scheduler <InfoTooltip text="Controls how noise is distributed across sampling steps. Karras produces cleaner results at fewer steps, Normal is the default, Exponential can work well with specific samplers." /></label>
                <select className="form-select" value={scheduler} onChange={(e) => setScheduler(e.target.value)}>
                  <option value="normal">Normal</option>
                  <option value="karras">Karras</option>
                  <option value="exponential">Exponential</option>
                  <option value="sgm_uniform">SGM Uniform</option>
                </select>
              </div>
            </div>

            <div>
              <label className="grok-section-label">Seed (-1 = random) <InfoTooltip text="Random seed for reproducibility. Use -1 for a different result each time, or set a specific number to reproduce the exact same output." /></label>
              <input
                className="form-input"
                type="number"
                value={seed}
                onChange={(e) => setSeed(parseInt(e.target.value) || -1)}
              />
            </div>
          </div>
        )}
      </div>

      </>)}

      {/* ════════ EDIT MODE ════════ */}
      {mode === 'edit' && (<>

      {/* Edit Instruction Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Wand2 size={16} />
            Edit Instruction
            <InfoTooltip text="Describe what you want to change in natural language. Unlike Transform which 'repaints' the image, AI Edit actually understands your instruction and applies it precisely." />
          </div>
        </div>

        <div className="form-group">
          <label className="grok-section-label">What should change?</label>
          <textarea
            className="form-textarea"
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            rows={3}
            placeholder="Describe the edit... (e.g. 'Remove the background and replace with a beach sunset')"
          />
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '8px' }}>
          {EXAMPLE_INSTRUCTIONS.slice(0, 5).map((ex) => (
            <button
              key={ex}
              onClick={() => setInstruction(ex)}
              style={{
                padding: '4px 10px', fontSize: '0.75rem',
                background: 'rgba(139, 92, 246, 0.08)',
                border: '1px solid rgba(139, 92, 246, 0.2)',
                borderRadius: '12px', color: 'var(--accent-color, #a78bfa)',
                cursor: 'pointer', transition: 'all 0.15s',
              }}
            >
              {ex}
            </button>
          ))}
        </div>

        <div className="form-group" style={{ marginTop: '14px' }}>
          <label className="grok-section-label">
            Negative Prompt (optional)
            <InfoTooltip text="Describe what to avoid in the output. Usually not needed for I2I Edit Edit, but can help prevent unwanted artifacts." />
          </label>
          <textarea
            className="form-textarea"
            value={negativePrompt}
            onChange={(e) => setNegativePrompt(e.target.value)}
            rows={2}
            placeholder="Optional: what to avoid..."
          />
        </div>
      </div>

      {/* Resolution & Aspect Ratio Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            Resolution & Ratio
            <InfoTooltip text="Pick a base resolution and aspect ratio. Higher resolution = more detail but slower. The actual pixel dimensions are shown below." />
          </div>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 600 }}>
            {getEditDimensions(editResolution, editAspectRatio).width}×{getEditDimensions(editResolution, editAspectRatio).height}
          </span>
        </div>

        {/* Resolution toggle */}
        <div className="form-group" style={{ marginBottom: '12px' }}>
          <label className="grok-section-label" style={{ fontSize: '0.75rem' }}>Resolution</label>
          <div className="grok-toggle-group">
            {Object.entries(EDIT_RESOLUTION_PRESETS).map(([key, preset]) => (
              <button
                key={key}
                className={`grok-toggle-btn ${editResolution === key ? 'active' : ''}`}
                onClick={() => setEditResolution(key)}
                type="button"
              >
                {preset.label}
                <span style={{ fontSize: '0.65rem', opacity: 0.7, display: 'block' }}>
                  {preset.desc}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Aspect ratio buttons */}
        <div className="form-group">
          <label className="grok-section-label" style={{ fontSize: '0.75rem' }}>Aspect Ratio</label>
          <div className="aspect-grid" style={{ gridTemplateColumns: 'repeat(7, 1fr)' }}>
            {EDIT_ASPECT_RATIOS.map((label) => {
              const [aw, ah] = label.split(':').map(Number)
              const ratio = aw / ah
              const maxDim = 22
              const w = ratio >= 1 ? maxDim : Math.round(maxDim * ratio)
              const h = ratio >= 1 ? Math.round(maxDim / ratio) : maxDim
              const dims = getEditDimensions(editResolution, label)
              return (
                <button
                  key={label}
                  className={`aspect-btn ${editAspectRatio === label ? 'active' : ''}`}
                  onClick={() => setEditAspectRatio(label)}
                  style={{ height: '60px' }}
                  title={`${dims.width}×${dims.height}`}
                >
                  <div className="aspect-icon" style={{ background: 'transparent', display: 'flex', alignItems: 'center', justifyContent: 'center', border: 'none', marginBottom: '4px' }}>
                    <div style={{ width: `${w}px`, height: `${h}px`, border: '1px solid currentColor' }} />
                  </div>
                  <span className="aspect-label" style={{ fontSize: '0.6rem' }}>{label}</span>
                </button>
              )
            })}
          </div>
        </div>
      </div>

      {/* Model Variant Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Palette size={16} />
            I2I Edit Model
            <InfoTooltip text="Choose which I2I Edit edit model variant runs on RunPod. The default model is the current official I2I Edit Image Edit setup. JIB Mix is an alternative merge with a different realism/style balance." />
          </div>
        </div>

        <div className="form-group">
          <label className="grok-section-label">Model Variant</label>
          <select className="form-select" value={editModel} onChange={(e) => setI2IModel(e.target.value)}>
            {EDIT_MODEL_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
          <div style={{ marginTop: '8px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            {EDIT_MODEL_OPTIONS.find((option) => option.value === editModel)?.desc}
          </div>
        </div>
      </div>

      {/* Speed & Quality Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Zap size={16} />
            Speed & Quality
          </div>
        </div>

        <label className="toggle-row">
          <div className="toggle-info">
            <span className="toggle-label">
              ⚡ Lightning Mode
              <InfoTooltip text="Uses a Lightning LoRA for 4-step generation instead of 40. Much faster but may sacrifice some quality. Great for quick iterations." size={12} />
            </span>
            <span className="toggle-desc">4-step fast generation (vs 40 normal)</span>
          </div>
          <input
            type="checkbox"
            checked={lightning}
            onChange={(e) => setLightning(e.target.checked)}
          />
          <span className="toggle-slider" />
        </label>

        <div style={{ display: 'flex', gap: '12px', marginTop: '14px' }}>
          <div style={{ flex: 1 }}>
            <label className="grok-section-label">
              Steps <InfoTooltip text="Number of sampling steps. 40 for normal quality, 4 for lightning mode." />
            </label>
            <input className="form-input" type="number" value={steps}
              onChange={(e) => setSteps(parseInt(e.target.value) || 40)} min="1" max="50" />
          </div>
          <div style={{ flex: 1 }}>
            <label className="grok-section-label">
              CFG Scale <InfoTooltip text="How strictly the model follows the instruction. 4.0 for normal, 1.0 for lightning." />
            </label>
            <input className="form-input" type="number" value={cfg}
              onChange={(e) => setCfg(parseFloat(e.target.value) || 4.0)} min="1" max="10" step="0.5" />
          </div>
          <div style={{ flex: 1 }}>
            <label className="grok-section-label">
              Seed <InfoTooltip text="Random seed for reproducible results. Use -1 for random." />
            </label>
            <input className="form-input" type="number" value={seed}
              onChange={(e) => setSeed(parseInt(e.target.value) || -1)} />
          </div>
        </div>
      </div>

      {/* LoRA Stack Card */}
      <div className="grok-card">
        <div className="grok-card-header" style={{ cursor: 'pointer' }} onClick={() => setShowLoraPanel(!showLoraPanel)}>
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Palette size={16} />
            LoRA Stack
            <InfoTooltip text="Add custom LoRAs to influence the editing style. Each LoRA adds 2 credits." />
            {loraConfigs.length > 0 && (
              <span style={{
                fontSize: '0.7rem', padding: '2px 8px',
                backgroundColor: 'rgba(139, 92, 246, 0.15)',
                borderRadius: '10px', color: 'var(--accent-color, #a78bfa)', fontWeight: 600,
              }}>
                {loraConfigs.length} active
              </span>
            )}
          </div>
          <span style={{ opacity: 0.5, fontSize: '0.8rem' }}>{showLoraPanel ? '▼' : '▶'}</span>
        </div>

        {showLoraPanel && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', padding: '0 0 4px' }}>
            {loraConfigs.map((config, idx) => (
              <div key={idx} style={{
                padding: '10px', backgroundColor: 'var(--bg-tertiary, rgba(0,0,0,0.1))',
                borderRadius: '6px', border: '1px solid var(--border-color)',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                  <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>LoRA #{idx + 1}</span>
                  <button
                    onClick={() => setLoraConfigs(loraConfigs.filter((_, i) => i !== idx))}
                    style={{ background: 'transparent', border: 'none', color: '#ef4444', cursor: 'pointer', fontSize: '0.75rem' }}
                  >✕ Remove</button>
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
                    style={{
                      width: '100%', padding: '6px 10px',
                      backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)',
                      borderRadius: '4px', color: 'var(--text-primary)', fontSize: '0.8rem',
                    }}
                  >
                    <option value="">None</option>
                    {Object.keys(filteredLoras).map((category) => (
                      <optgroup key={category} label={category === 'root' ? 'General' : category}>
                        {filteredLoras[category].map((lora) => (
                          <option key={lora.path} value={lora.path}>{lora.name} ({lora.size_mb}MB)</option>
                        ))}
                      </optgroup>
                    ))}
                  </select>
                </div>

                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Strength</label>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{(config.strength || 1.0).toFixed(2)}</span>
                  </div>
                  <input type="range" min="0" max="2" step="0.05"
                    value={config.strength || 1.0}
                    onChange={(e) => {
                      const nc = [...loraConfigs]
                      nc[idx] = { ...config, strength: parseFloat(e.target.value) }
                      setLoraConfigs(nc)
                    }}
                    style={{ width: '100%', cursor: 'pointer' }}
                  />
                </div>
              </div>
            ))}

            <button
              onClick={() => setLoraConfigs([...loraConfigs, { name: '', strength: 1.0 }])}
              style={{
                padding: '8px 12px', backgroundColor: 'transparent',
                border: '1px dashed var(--border-color)', borderRadius: '6px',
                color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '0.85rem',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
              }}
            >
              + Add LoRA
            </button>

            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
              💡 Stack multiple LoRAs for combined style effects. Each LoRA adds 2 credits.
            </div>
          </div>
        )}
      </div>

      {/* Cloud info banner */}
      <div style={{
        padding: '12px 16px',
        background: 'rgba(139, 92, 246, 0.06)',
        border: '1px solid rgba(139, 92, 246, 0.15)',
        borderRadius: '10px', fontSize: '0.8rem',
        color: 'var(--text-muted)', lineHeight: 1.5,
      }}>
        <strong>☁️ Cloud-powered</strong> — Runs on RunPod cloud GPUs (48GB+).
        {lightning ? ' ⚡ Lightning: ~30s.' : ' 🎨 Normal: ~2-3 min.'}
        {editModel === 'jib_mix_v6' ? ' 🧪 JIB Mix selected: first cold start can be slower.' : ''}
      </div>

      </>)}

      {/* Queued notification */}
      {lastQueued && (
        <div className="queued-notice">
          ✅ Job queued! Check the Queue panel for progress.
          {lastQueued.mode === 'edit'
            ? <span className="queued-mode">{lastQueued.editModel} · {lastQueued.lightning ? '⚡ Lightning' : '🎨 Full quality'} (Cloud)</span>
            : <span className="queued-mode">{lastQueued.checkpoint}</span>
          }
        </div>
      )}

      {error && <div className="error-message">⚠️ {error}</div>}

      <button
        className="btn-primary btn-large"
        onClick={handleGenerate}
        disabled={!file || submitting || (mode === 'edit' && !instruction.trim())}
      >
        {submitting ? (
          <>
            <Loader2 size={18} className="spin" />
            {mode === 'edit' ? 'Submitting to Cloud...' : 'Queueing...'}
          </>
        ) : mode === 'edit' ? (
          <>
            <Sparkles size={18} />
            Edit Image
          </>
        ) : (
          <>
            <Wand2 size={18} />
            Transform Image
            {(faceId || faceDetailer || faceRestore) && (
              <span className="btn-badge">
                {[faceId && '🛡️', faceDetailer && '✨', faceRestore && '💎'].filter(Boolean).join('')}
              </span>
            )}
          </>
        )}
      </button>

      {/* Result */}
      {result && (
        <div className="result-section">
          <h3>Result</h3>
          <div className="comparison">
            <div className="comparison-item">
              <span className="comparison-label">Original</span>
              <img src={preview} alt="Original" />
            </div>
            <div className="comparison-item">
              <span className="comparison-label">Transformed</span>
              <img src={result} alt="Result" />
            </div>
          </div>
        </div>
      )}

      {importModal && (
        <MediaImportModal
          item={importModal.item}
          parsedData={importModal.workflow || {}}
          availableFields={['image', 'positive', 'negative', 'steps', 'cfg', 'seed']}
          onApply={handleApplyImport}
          onClose={() => setImportModal(null)}
        />
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
          flex-direction: column;
          position: relative;
        }
        .upload-preview {
          max-width: 100%;
          max-height: 300px;
          border-radius: 8px;
          object-fit: contain;
        }
        .btn-clear-image {
          position: absolute;
          top: 12px;
          right: 12px;
          background: rgba(0, 0, 0, 0.6);
          color: white;
          border: none;
          border-radius: 50%;
          width: 28px;
          height: 28px;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          padding: 0;
          z-index: 2;
        }
        .btn-clear-image:hover {
          background: rgba(220, 50, 50, 0.8);
        }
        .upload-filename {
          margin-top: 6px;
          font-size: 11px;
          color: var(--text-muted, #888);
          text-align: center;
          word-break: break-all;
          max-width: 100%;
        }
        .upload-placeholder {
          color: var(--text-muted, #888);
        }
        .upload-placeholder p {
          margin-top: 12px;
        }

        /* Preset Grid */
        .preset-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 8px;
        }
        .preset-card {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 4px;
          padding: 12px 8px;
          border: 1px solid var(--border-color, #333);
          border-radius: 10px;
          background: var(--bg-secondary, #1a1a1a);
          cursor: pointer;
          transition: all 0.2s;
          text-align: center;
        }
        .preset-card:hover {
          border-color: var(--text-muted, #666);
          background: var(--bg-tertiary, #222);
        }
        .preset-card.active {
          background: var(--bg-tertiary, #252525);
        }
        .preset-icon {
          font-size: 20px;
        }
        .preset-label {
          font-size: 12px;
          font-weight: 600;
          color: var(--text-color, #fff);
        }
        .preset-desc {
          font-size: 10px;
          color: var(--text-muted, #888);
          line-height: 1.3;
        }

        /* Face Processing Toggles */
        .face-toggles {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .toggle-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 10px 12px;
          border: 1px solid var(--border-color, #333);
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.15s;
          background: var(--bg-secondary, #1a1a1a);
        }
        .toggle-row:hover {
          border-color: var(--text-muted, #555);
        }
        .toggle-info {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }
        .toggle-label {
          font-size: 13px;
          font-weight: 500;
          color: var(--text-color, #fff);
        }
        .toggle-desc {
          font-size: 11px;
          color: var(--text-muted, #888);
        }
        .toggle-row input[type="checkbox"] {
          appearance: none;
          -webkit-appearance: none;
          width: 40px;
          height: 22px;
          background: var(--border-color, #444);
          border-radius: 12px;
          position: relative;
          cursor: pointer;
          transition: background 0.2s;
          flex-shrink: 0;
        }
        .toggle-row input[type="checkbox"]:checked {
          background: var(--accent-color, #7c3aed);
        }
        .toggle-row input[type="checkbox"]::after {
          content: '';
          position: absolute;
          top: 3px;
          left: 3px;
          width: 16px;
          height: 16px;
          background: white;
          border-radius: 50%;
          transition: transform 0.2s;
        }
        .toggle-row input[type="checkbox"]:checked::after {
          transform: translateX(18px);
        }
        .toggle-row .toggle-slider {
          display: none;
        }
        .face-weight-slider {
          margin: 0 0 4px 0;
          padding: 8px 12px;
          background: var(--bg-tertiary, #1e1e1e);
          border-radius: 8px;
          border: 1px solid var(--border-color, #333);
        }
        .face-weight-slider label {
          margin-bottom: 6px !important;
        }

        /* Button Badge */
        .btn-badge {
          margin-left: 8px;
          font-size: 14px;
        }

        .form-group {
          margin-bottom: 16px;
        }
        .form-group label {
          display: flex;
          align-items: center;
          gap: 6px;
          margin-bottom: 8px;
          font-size: 13px;
          color: var(--text-secondary, #aaa);
        }
        .label-value {
          margin-left: auto;
          color: var(--accent-color, #7c3aed);
          font-weight: 500;
        }
        .form-group textarea,
        .form-group select,
        .form-group input[type="number"] {
          width: 100%;
          padding: 10px 12px;
          border-radius: 8px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          font-size: 13px;
        }
        .range-labels {
          display: flex;
          justify-content: space-between;
          font-size: 11px;
          color: var(--text-muted, #888);
          margin-top: 4px;
        }
        .denoise-hint {
          margin-top: 8px;
          font-size: 12px;
          color: var(--text-muted, #888);
        }
        .form-row {
          display: flex;
          gap: 16px;
        }
        .form-group.half {
          flex: 1;
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
        .result-section {
          margin-top: 24px;
          padding-top: 24px;
          border-top: 1px solid var(--border-color, #333);
        }
        .comparison {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 16px;
          margin-top: 16px;
        }
        .comparison-item {
          position: relative;
        }
        .comparison-label {
          position: absolute;
          top: 8px;
          left: 8px;
          background: rgba(0,0,0,0.7);
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 11px;
        }
        .comparison-item img {
          width: 100%;
          border-radius: 8px;
        }
        .queued-notice .queued-features {
          display: flex;
          gap: 6px;
          margin-top: 4px;
          font-size: 11px;
          color: var(--text-muted, #888);
        }
        .spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        /* Mode Selector */
        .mode-btn {
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 2px;
          padding: 10px 12px;
          border: 2px solid transparent;
          border-radius: 10px;
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-muted, #888);
          cursor: pointer;
          transition: all 0.2s;
          font-size: 0.85rem;
          font-weight: 600;
        }
        .mode-btn:hover {
          border-color: var(--border-color, #444);
          color: var(--text-secondary, #aaa);
        }
        .mode-btn.active {
          border-color: var(--accent-color, #7c3aed);
          background: rgba(124, 58, 237, 0.08);
          color: var(--text-color, #fff);
          box-shadow: 0 0 12px rgba(124, 58, 237, 0.15);
        }
        .mode-hint {
          font-size: 0.65rem;
          font-weight: 400;
          opacity: 0.6;
        }
      `}</style>
    </div>
  )
}
