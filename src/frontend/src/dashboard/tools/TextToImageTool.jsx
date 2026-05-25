import React, { useState, useEffect, useMemo, useCallback } from 'react'
import { Settings2, Image as ImageIcon, Info, ChevronDown, Wand2, Loader2, Sparkles } from 'lucide-react'
import InfoTooltip from '../../components/InfoTooltip'
import { DEBUG, getMediaUrl } from '../../config'
import { apiFetch } from '../../api'
import { useNSFW } from '../../contexts/NSFWContext'
import { useAuth } from '../../contexts/AuthContext'
import CameraPositionSelector, { getCameraPositionPrefix } from '../../components/CameraPositionSelector'
import MediaImportModal from '../../components/MediaImportModal'
import useLLMEnhance from '../../hooks/useLLMEnhance'
import useGeneration from '../../hooks/useGeneration'
import LLMQueueIndicator from '../../components/LLMQueueIndicator'
import { PROMPT_LLM_MODELS, DEFAULT_PROMPT_LLM } from '../../constants/llmModels'
import { useToolSettings } from '../../hooks/useToolSettings'
import ResetDefaultsButton from '../../components/ResetDefaultsButton'

// Models grouped by category
const MODEL_GROUPS = {
  flux: {
    label: '⚡ Flux',
    desc: 'Best quality',
    models: [
      { value: 'flux1-dev-fp8', label: 'Flux.1 Dev (FP8)', desc: 'Highest quality, slower' },
    ]
  },
  sdxl: {
    label: '🎨 SDXL',
    desc: 'Great balance',
    models: [
      { value: 'CyberRealistic_Pony_v14.1_FP16.safetensors', label: 'CyberRealistic Pony', desc: 'Photorealistic + Pony tags' },
      { value: 'dreamshaperXL_lightningDPMSDE.safetensors', label: 'Dreamshaper Lightning', desc: 'Fast, artistic' },
      { value: 'illustriousRealismBy_v10VAE.safetensors', label: 'Illustrious Realism', desc: 'Detailed realistic' },
      { value: 'juggernautXL_ragnarok.safetensors', label: 'Juggernaut XL', desc: 'All-rounder' },
      { value: 'novaAnimeXL_ilV150.safetensors', label: 'Nova Anime XL', desc: 'Anime style' },
      { value: 'ponyDiffusionV6XL_v6StartWithThisOne.safetensors', label: 'Pony Diffusion V6', desc: 'Booru tags, NSFW' },
      { value: 'reapony_v90.safetensors', label: 'Reapony V9', desc: 'Realistic + Pony' },
      { value: 'ultraRealisticByStable_v20FP16.safetensors', label: 'Ultra Realistic', desc: 'Hyperrealistic' },
      { value: 'waiIllustriousSDXL_v160.safetensors', label: 'Wai Illustrious', desc: 'Anime + 2.5D' },
    ]
  },
  sd15: {
    label: '🚀 SD 1.5',
    desc: 'Fast, low VRAM',
    models: [
      { value: 'Realistic_Vision_V5.1.safetensors', label: 'Realistic Vision V5.1', desc: 'Fast realistic' },
    ]
  },
  wan22: {
    label: '🎬 Wan2.2',
    desc: 'Video model T2I',
    models: [
      { value: 'wan2.2-t2i', label: 'Wan2.2 T2I', desc: 'Multi-GPU video model' },
    ]
  },
  ernie: {
    label: '🖼️ ERNIE-Image',
    desc: 'Baidu ERNIE',
    models: [
      { value: 'ernie-image', label: 'ERNIE-Image', desc: 'Flux2 latent, Ministral-3B' },
    ]
  },
  diffusers: {
    label: '🐍 Diffusers',
    desc: 'Python pipeline',
    models: [
      { value: 'sd3.5-large-int8', label: 'SD3.5 Large (INT8)', desc: 'Latest SD3.5' },
    ]
  },
}

// Determine model type
const getModelType = (modelValue) => {
  for (const [type, group] of Object.entries(MODEL_GROUPS)) {
    if (group.models.find(m => m.value === modelValue)) return type
  }
  return 'sdxl'
}

const T2I_DEFAULTS = {
  prompt: '', negativePrompt: 'ugly, deformed, blurry, low quality, bad anatomy, watermark, signature, text',
  aspectRatio: '1:1', model: 'CyberRealistic_Pony_v14.1_FP16.safetensors', batchCount: 1,
  enhanceModel: DEFAULT_PROMPT_LLM, cameraPosition: '',
  selectedLoras: [{ name: 'None', strength: 1.0 }, { name: 'None', strength: 1.0 }, { name: 'None', strength: 1.0 }],
  steps: 30, cfg: 7.5, guidance: 3.5, seed: -1, sampler: 'dpmpp_2m', scheduler: 'karras',
}

export default function TextToImageTool({ onOutput, onJobSubmitted, pendingImport = null, onImportConsumed = null }) {
  const { nsfwEnabled } = useNSFW()
  const { user, requestLogin } = useAuth()
  const { initial, save: saveSettings, resetDefaults } = useToolSettings('text_to_image', T2I_DEFAULTS)

  const [prompt, setPrompt] = useState(initial.prompt)
  const [negativePrompt, setNegativePrompt] = useState(initial.negativePrompt)
  const [aspectRatio, setAspectRatio] = useState(initial.aspectRatio)
  const [model, setModel] = useState(initial.model)
  const [batchCount, setBatchCount] = useState(initial.batchCount)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isEnhancing, setIsEnhancing] = useState(false)
  const [enhanceModel, setEnhanceModel] = useState(initial.enhanceModel)
  const [error, setError] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [lastQueued, setLastQueued] = useState(null)
  const [cameraPosition, setCameraPosition] = useState(initial.cameraPosition)

  // LoRA settings
  const [availableLoras, setAvailableLoras] = useState([])
  const [selectedLoras, setSelectedLoras] = useState(initial.selectedLoras)

  // Advanced settings
  const [steps, setSteps] = useState(initial.steps)
  const [cfg, setCfg] = useState(initial.cfg)
  const [guidance, setGuidance] = useState(initial.guidance)  // For Flux
  const [seed, setSeed] = useState(initial.seed)
  const [sampler, setSampler] = useState(initial.sampler)
  const [scheduler, setScheduler] = useState(initial.scheduler)

  // ── Auto-save settings ──────────────────────────────────────────────
  const settingsSnapshot = useMemo(() => ({
    prompt, negativePrompt, aspectRatio, model, batchCount, enhanceModel,
    cameraPosition, selectedLoras, steps, cfg, guidance, seed, sampler, scheduler,
  }), [prompt, negativePrompt, aspectRatio, model, batchCount, enhanceModel,
    cameraPosition, selectedLoras, steps, cfg, guidance, seed, sampler, scheduler])
  useEffect(() => { saveSettings(settingsSnapshot) }, [settingsSnapshot, saveSettings])

  const handleResetDefaults = useCallback(() => {
    const d = resetDefaults()
    setPrompt(d.prompt); setNegativePrompt(d.negativePrompt); setAspectRatio(d.aspectRatio)
    setModel(d.model); setBatchCount(d.batchCount); setEnhanceModel(d.enhanceModel)
    setCameraPosition(d.cameraPosition); setSelectedLoras(d.selectedLoras)
    setSteps(d.steps); setCfg(d.cfg); setGuidance(d.guidance); setSeed(d.seed)
    setSampler(d.sampler); setScheduler(d.scheduler)
  }, [resetDefaults])

  // Pending import modal state
  const [importModal, setImportModal] = useState(null)  // { item, workflow }

  // When Dashboard sends a new pendingImport, show the modal
  useEffect(() => {
    if (!pendingImport) return
    setImportModal(pendingImport)
    if (onImportConsumed) onImportConsumed()
  }, [pendingImport])

  const handleApplyImport = (selected) => {
    if (selected.positive)  setPrompt(selected.positive)
    if (selected.negative)  setNegativePrompt(selected.negative)
    if (selected.steps)     setSteps(selected.steps)
    if (selected.cfg)       setCfg(selected.cfg)
    if (selected.sampler)   setSampler(selected.sampler)
    if (selected.scheduler) setScheduler(selected.scheduler)
    if (selected.seed)      setSeed(selected.seed)
    setImportModal(null)
  }

  // Random subjects for empty prompt generation
  const randomSubjects = [
    'beautiful woman in elegant dress',
    'handsome man in suit',
    'cute cat lounging',
    'majestic wolf in forest',
    'futuristic city skyline',
    'fantasy castle on mountain',
    'cozy coffee shop interior',
    'tropical beach sunset',
    'mystical forest with glowing mushrooms',
    'cyberpunk street at night',
    'portrait of elegant lady',
    'vintage car in countryside',
    'underwater coral reef',
    'astronaut on alien planet',
    'steampunk airship',
  ]

  // LLM prompt enhancement queue
  const llm = useLLMEnhance()

  // Enhance prompt with LLM (via async queue)
  const handleEnhancePrompt = async () => {
    if (isEnhancing) return
    setIsEnhancing(true)
    setError('')

    // If empty, pick a random subject
    const inputPrompt = prompt.trim() || randomSubjects[Math.floor(Math.random() * randomSubjects.length)]

    const result = await llm.enhance({
      input: inputPrompt,
      mode: 'expand',
      include_negative: true,
      include_motion: false,
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

  // Fetch available LoRAs on mount
  useEffect(() => {
    const fetchLoras = async () => {
      try {
        const res = await apiFetch('/loras')
        if (res.ok) {
          const data = await res.json()
          setAvailableLoras(data.loras || [])
        }
      } catch (e) {
        console.warn('Failed to fetch LoRAs:', e)
      }
    }
    fetchLoras()
  }, [])

  // Filter LoRAs based on NSFW setting
  const filteredLoras = useMemo(() => {
    if (nsfwEnabled) return availableLoras
    return availableLoras.filter(l => !l.nsfw)
  }, [availableLoras, nsfwEnabled])

  // Update LoRA selection
  const updateLora = (index, field, value) => {
    setSelectedLoras(prev => {
      const updated = [...prev]
      updated[index] = { ...updated[index], [field]: value }
      return updated
    })
  }

  const { generate } = useGeneration()

  const resolveImmediateImageOutput = useCallback((result) => {
    const resultUrl = result?.meta?.result_url
    const resultPath = result?.meta?.result_path

    let backendPath = resultUrl || null

    if (!backendPath && typeof resultPath === 'string') {
      if (resultPath.includes('/media/generated/')) {
        backendPath = `/media/generated/${resultPath.split('/media/generated/')[1]}`
      } else if (resultPath.includes('/ComfyUI/output/')) {
        backendPath = `/comfyui/output/${resultPath.split('/ComfyUI/output/')[1]}`
      }
    }

    if (!backendPath) return null

    const filename = result?.meta?.filename || backendPath.split('/').pop() || 'generated-image.png'
    const mediaUrl = getMediaUrl(backendPath)

    return {
      kind: 'image',
      url: mediaUrl,
      backendUrl: mediaUrl,
      filename,
    }
  }, [])

  const handleGenerate = async () => {
    // Check if user is logged in
    if (!user) {
      requestLogin('Log in om te genereren')
      return
    }

    if (!prompt.trim()) return
    setIsGenerating(true)
    setError('')
    setLastQueued(null)

    try {
      const queuedJobs = []

      // Build prompt with camera position prefix
      const positionPrefix = getCameraPositionPrefix(cameraPosition)
      const finalPrompt = positionPrefix + prompt

      for (let i = 0; i < batchCount; i++) {
        // Determine adapter based on model type
        const modelType = getModelType(model)

        let adapterHint = 'sdxl-local-t2i'
        if (modelType === 'wan22') adapterHint = 'wan22-local-t2i'
        else if (modelType === 'ernie') adapterHint = 'ernie-local-t2i'
        else if (modelType === 'flux') adapterHint = 'flux-local-t2i'
        else if (modelType === 'sd15') adapterHint = 'sd15-local-t2i'

        if (DEBUG) console.debug('🎨 T2I V2 request:', { adapterHint, modelType, model })

        const requestPayload = {
          operation: 'generate',
          target_type: 'image',
          adapter_hint: adapterHint,
          prompt: finalPrompt,
          negative_prompt: negativePrompt || undefined,
          seed: parseInt(seed, 10),
          steps: parseInt(steps, 10),
          cfg: parseFloat(cfg),
          aspect_ratio: aspectRatio,
          checkpoint: model, // Some adapters use this, some ignore it securely
          sampler: sampler,
          scheduler: scheduler,
          loras: selectedLoras.filter(l => l.name && l.name !== 'None').map(l => ({ name: l.name, strength: l.strength }))
        }

        const result = await generate(requestPayload)

        if (!result) {
          // generate() returned null, meaning error occurred (handled inside hook or aborted).
          // We can break early to not keep failing the batch
          break
        }

        if (DEBUG) console.log(`📋 Batch ${i+1}/${batchCount} queued:`, result)

        const immediateOutput = resolveImmediateImageOutput(result)
        if (immediateOutput) {
          if (onOutput) onOutput(immediateOutput)
          continue
        }

        // Track queued job
        if (result.prompt_id) {
          queuedJobs.push(result.prompt_id)
        }

        // Notify queue indicator
        if (onJobSubmitted) onJobSubmitted({ prompt_id: result.prompt_id })
      }

      if (queuedJobs.length > 0) {
        // Show queued confirmation
        setLastQueued({
          count: queuedJobs.length,
          model: getModelLabel(),
          promptIds: queuedJobs
        })
      }

    } catch (e) {
      console.error('Generation error:', e)
      setError(e.message || 'Failed to generate image')
    } finally {
      setIsGenerating(false)
    }
  }

  // Get model display label
  const getModelLabel = () => {
    for (const group of Object.values(MODEL_GROUPS)) {
      const found = group.models.find(m => m.value === model)
      if (found) return found.label
    }
    return model
  }

  return (
    <div className="tool-container">
      {/* Import from previous generation modal */}
      {importModal && (
        <MediaImportModal
          item={importModal.item}
          parsedData={importModal.workflow}
          availableFields={['positive', 'negative', 'steps', 'cfg', 'sampler', 'scheduler', 'seed']}
          onApply={handleApplyImport}
          onClose={() => setImportModal(null)}
        />
      )}

      {/* Model Selection - Grouped with hover info */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            Model
            <ResetDefaultsButton onReset={handleResetDefaults} />
          </div>
          <span className="nav-badge" style={{ fontSize: '0.7rem' }}>
            {getModelType(model).toUpperCase()}
          </span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {Object.entries(MODEL_GROUPS).map(([groupKey, group]) => (
            <div key={groupKey}>
              <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '6px' }}>
                {group.label}
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                {group.models.map((m) => (
                  <button
                    key={m.value}
                    onClick={() => setModel(m.value)}
                    title={m.desc}
                    style={{
                      padding: '6px 12px',
                      background: model === m.value ? 'var(--primary-color)' : 'var(--bg-input)',
                      border: model === m.value ? '1px solid var(--primary-color)' : '1px solid var(--border-color)',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      color: model === m.value ? '#fff' : 'var(--text-secondary)',
                      fontSize: '0.8rem',
                      fontWeight: 500,
                      transition: 'all 0.15s ease',
                    }}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Prompt Input */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Positive Prompt</div>
          <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
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
              {PROMPT_LLM_MODELS.map((m) => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
            </select>
            <button
              className="icon-btn"
              style={{ width: '24px', height: '24px', padding: '4px' }}
              onClick={handleEnhancePrompt}
              disabled={isEnhancing}
              title={prompt.trim() ? 'Enhance prompt with AI' : 'Generate random prompt with AI'}
            >
              {isEnhancing ? <Loader2 size={12} className="spin" /> : <Wand2 size={12} />}
            </button>
            <LLMQueueIndicator queuePosition={llm.queuePosition} isLoading={llm.isLoading} />
          </div>
        </div>
        <textarea
          className="form-textarea"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={4}
          placeholder="A attractive blonde woman, tattoos, looking at me defiantly..."
          style={{
            backgroundColor: '#0f0f0f',
            border: 'none',
            resize: 'none',
          }}
        />
        {/* Camera Position Selector */}
        <CameraPositionSelector value={cameraPosition} onChange={setCameraPosition} style={{ marginTop: '12px' }} />
      </div>

      {/* Negative Prompt (for SDXL and SD1.5, not Flux) */}
      {(getModelType(model) === 'sdxl' || getModelType(model) === 'sd15') && (
        <div className="grok-card">
          <div className="grok-card-header">
            <div className="grok-card-title">Negative Prompt</div>
          </div>
          <textarea
            className="form-textarea"
            value={negativePrompt}
            onChange={(e) => setNegativePrompt(e.target.value)}
            rows={2}
            placeholder="ugly, deformed, blurry..."
            style={{
              backgroundColor: '#0f0f0f',
              border: 'none',
              resize: 'none',
              fontSize: '0.85rem'
            }}
          />
        </div>
      )}

      {/* Aspect Ratio */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Aspect Ratio</div>
        </div>
        <div className="aspect-grid" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>
          {[
            { label: '1:1', icon: <div style={{ width: '18px', height: '18px', border: '1px solid currentColor' }} /> },
            { label: '16:9', icon: <div style={{ width: '24px', height: '14px', border: '1px solid currentColor' }} /> },
            { label: '9:16', icon: <div style={{ width: '14px', height: '24px', border: '1px solid currentColor' }} /> },
            { label: '4:3', icon: <div style={{ width: '20px', height: '15px', border: '1px solid currentColor' }} /> },
            { label: '3:4', icon: <div style={{ width: '15px', height: '20px', border: '1px solid currentColor' }} /> },
            { label: '2:3', icon: <div style={{ width: '16px', height: '24px', border: '1px solid currentColor' }} /> },
            { label: '3:2', icon: <div style={{ width: '24px', height: '16px', border: '1px solid currentColor' }} /> },
            { label: '4:5', icon: <div style={{ width: '16px', height: '20px', border: '1px solid currentColor' }} /> },
            { label: '5:4', icon: <div style={{ width: '20px', height: '16px', border: '1px solid currentColor' }} /> },
            { label: '9:21', icon: <div style={{ width: '10px', height: '24px', border: '1px solid currentColor' }} /> },
            { label: '21:9', icon: <div style={{ width: '24px', height: '10px', border: '1px solid currentColor' }} /> },
          ].map((ratio) => (
            <button
              key={ratio.label}
              className={`aspect-btn ${aspectRatio === ratio.label ? 'active' : ''}`}
              onClick={() => setAspectRatio(ratio.label)}
              style={{ height: '60px' }}
            >
              <div className="aspect-icon" style={{ background: 'transparent', display: 'flex', alignItems: 'center', justifyContent: 'center', border: 'none', marginBottom: '4px' }}>
                {ratio.icon}
              </div>
              <span className="aspect-label" style={{ fontSize: '0.65rem' }}>{ratio.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Advanced Settings */}
      <div className="grok-card">
        <div
          className="grok-card-header"
          style={{ cursor: 'pointer' }}
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          <div className="grok-card-title">Advanced Settings</div>
          <ChevronDown
            size={16}
            className="text-muted"
            style={{
              transform: showAdvanced ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.2s'
            }}
          />
        </div>

        {showAdvanced && (
          <>
            {/* Batch Count */}
            <div className="form-group">
              <label className="grok-section-label">Batch Count <InfoTooltip text="Number of images to generate at once. Higher = more variations to choose from, but takes longer. Start with 1 for quick testing, use 2-4 when exploring prompt ideas." /></label>
              <div className="grok-toggle-group">
                {[1, 2, 3, 4].map((num) => (
                  <button
                    key={num}
                    className={`grok-toggle-btn ${batchCount === num ? 'active' : ''}`}
                    onClick={() => setBatchCount(num)}
                  >
                    {num}
                  </button>
                ))}
              </div>
            </div>

            {/* Flux-specific settings */}
            {getModelType(model) === 'flux' && (
              <>
                <div className="form-group" style={{ marginTop: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <label className="grok-section-label">Steps <InfoTooltip text="Number of denoising iterations. Flux needs 10-30 steps. More = better quality but slower. 20 is a good default." /></label>
                    <span className="nav-badge">{steps}</span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="30"
                    value={steps}
                    onChange={(e) => setSteps(parseInt(e.target.value))}
                    className="form-range"
                  />
                </div>

                <div className="form-group">
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <label className="grok-section-label">Guidance <InfoTooltip text="Controls how much the model follows your prompt vs being creative. Low (1-3) = more creative freedom. High (7-10) = strict prompt adherence. Default: 3.5 for Flux." /></label>
                    <span className="nav-badge">{guidance}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    step="0.5"
                    value={guidance}
                    onChange={(e) => setGuidance(parseFloat(e.target.value))}
                    className="form-range"
                  />
                </div>

                <div className="form-group">
                  <label className="grok-section-label">Seed (-1 = random) <InfoTooltip text="Use -1 for a random seed. Set a specific number to reproduce the exact same image with identical settings." /></label>
                  <input
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(parseInt(e.target.value) || -1)}
                    className="form-input"
                    style={{
                      backgroundColor: '#0f0f0f',
                      border: '1px solid #333',
                      borderRadius: '6px',
                      padding: '8px',
                      width: '100%'
                    }}
                  />
                </div>
              </>
            )}

            {/* Wan2.2 T2I settings */}
            {getModelType(model) === 'wan22' && (
              <>
                <div className="form-group" style={{ marginTop: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <label className="grok-section-label">Steps <InfoTooltip text="Denoising steps for Wan2.2 text-to-image. Uses multi-GPU DisTorch2 pipeline. 20-30 recommended for quality results." /></label>
                    <span className="nav-badge">{steps}</span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="50"
                    value={steps}
                    onChange={(e) => setSteps(parseInt(e.target.value))}
                    className="form-range"
                  />
                  <div style={{ fontSize: '0.7rem', opacity: 0.6, marginTop: '4px' }}>
                    Multi-GPU workflow (DisTorch2) - 2-stage denoising
                  </div>
                </div>

                <div className="form-group">
                  <label className="grok-section-label">Seed (-1 = random) <InfoTooltip text="Use -1 for random. Set a specific seed to reproduce the exact same output." /></label>
                  <input
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(parseInt(e.target.value) || -1)}
                    className="form-input"
                    style={{
                      backgroundColor: '#0f0f0f',
                      border: '1px solid #333',
                      borderRadius: '6px',
                      padding: '8px',
                      width: '100%'
                    }}
                  />
                </div>
              </>
            )}

            {/* SDXL and SD1.5 settings */}
            {(getModelType(model) === 'sdxl' || getModelType(model) === 'sd15') && (
              <>
                <div className="form-group" style={{ marginTop: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <label className="grok-section-label">Steps <InfoTooltip text="Denoising steps. More = better quality. 20-30 is good for SDXL. 50 for absolute best quality (slow)." /></label>
                    <span className="nav-badge">{steps}</span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="50"
                    value={steps}
                    onChange={(e) => setSteps(parseInt(e.target.value))}
                    className="form-range"
                  />
                </div>

                <div className="form-group">
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <label className="grok-section-label">CFG Scale <InfoTooltip text="Classifier-Free Guidance. Low (1-5) = creative, dreamy. Medium (7-10) = balanced. High (12-15) = strict prompt adherence but may over-saturate. Default: 7 for SDXL." /></label>
                    <span className="nav-badge">{cfg}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="15"
                    step="0.5"
                    value={cfg}
                    onChange={(e) => setCfg(parseFloat(e.target.value))}
                    className="form-range"
                  />
                </div>

                <div className="form-group">
                  <label className="grok-section-label">Sampler <InfoTooltip text="The sampling algorithm used for denoising. euler = simple/fast. dpmpp_2m = high quality. euler_ancestral = more variation. dpmpp_sde = slower but very detailed." /></label>
                  <div className="grok-toggle-group" style={{ flexWrap: 'wrap', gap: '4px' }}>
                    {['euler', 'euler_ancestral', 'dpmpp_2m', 'dpmpp_sde'].map((s) => (
                      <button
                        key={s}
                        className={`grok-toggle-btn ${sampler === s ? 'active' : ''}`}
                        onClick={() => setSampler(s)}
                        style={{ fontSize: '0.7rem', padding: '4px 8px' }}
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="form-group">
                  <label className="grok-section-label">Scheduler <InfoTooltip text="Controls the noise schedule curve. normal = standard. karras = smooth transitions (recommended for most). exponential = more aggressive. sgm_uniform = even spacing." /></label>
                  <div className="grok-toggle-group" style={{ flexWrap: 'wrap', gap: '4px' }}>
                    {['normal', 'karras', 'exponential', 'sgm_uniform'].map((s) => (
                      <button
                        key={s}
                        className={`grok-toggle-btn ${scheduler === s ? 'active' : ''}`}
                        onClick={() => setScheduler(s)}
                        style={{ fontSize: '0.7rem', padding: '4px 8px' }}
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="form-group">
                  <label className="grok-section-label">Seed (-1 = random) <InfoTooltip text="Use -1 for random. Set a specific seed to reproduce the exact same image." /></label>
                  <input
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(parseInt(e.target.value) || -1)}
                    className="form-input"
                    style={{
                      backgroundColor: '#0f0f0f',
                      border: '1px solid #333',
                      borderRadius: '6px',
                      padding: '8px',
                      width: '100%'
                    }}
                  />
                </div>

                {/* LoRA Settings (SDXL only) */}
                {getModelType(model) === 'sdxl' && filteredLoras.length > 0 && (
                  <div className="form-group">
                    <label className="grok-section-label" style={{ marginBottom: '8px' }}>
                      LoRAs (up to 3) <InfoTooltip text="LoRA (Low-Rank Adaptation) models add specific styles, characters, or concepts to your images. Stack up to 3 LoRAs. Adjust strength per LoRA — 0.5-0.8 is usually best." /> {!nsfwEnabled && availableLoras.length > filteredLoras.length && (
                        <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginLeft: '8px' }}>
                          ({availableLoras.length - filteredLoras.length} hidden)
                        </span>
                      )}
                    </label>
                    {selectedLoras.map((lora, idx) => (
                      <div key={idx} style={{
                        display: 'flex',
                        gap: '8px',
                        marginBottom: '8px',
                        alignItems: 'center'
                      }}>
                        <select
                          value={lora.name}
                          onChange={(e) => updateLora(idx, 'name', e.target.value)}
                          style={{
                            flex: 1,
                            backgroundColor: '#0f0f0f',
                            border: '1px solid #333',
                            borderRadius: '6px',
                            padding: '6px 8px',
                            color: '#fff',
                            fontSize: '0.75rem'
                          }}
                        >
                          <option value="None">None</option>
                          {filteredLoras.map((l) => (
                            <option key={l.path} value={l.name}>{l.name}</option>
                          ))}
                        </select>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', minWidth: '80px' }}>
                          <input
                            type="range"
                            min="0"
                            max="2"
                            step="0.1"
                            value={lora.strength}
                            onChange={(e) => updateLora(idx, 'strength', parseFloat(e.target.value))}
                            disabled={lora.name === 'None'}
                            style={{ width: '50px' }}
                          />
                          <span style={{ fontSize: '0.7rem', opacity: lora.name === 'None' ? 0.3 : 1 }}>
                            {lora.strength.toFixed(1)}
                          </span>
                        </div>
                      </div>
                    ))}
                    <div style={{ fontSize: '0.65rem', opacity: 0.5, marginTop: '4px' }}>
                      Strength: 0.5-1.0 recommended
                    </div>
                  </div>
                )}
              </>
            )}
          </>
        )}
      </div>

      {error && (
        <div style={{ color: '#ef4444', marginBottom: '12px', fontSize: '0.9rem' }}>
          {error}
        </div>
      )}

      <button
        className="primary-btn"
        onClick={handleGenerate}
        disabled={isGenerating || !prompt.trim()}
        style={{
          height: '48px',
          fontSize: '1rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '8px',
          backgroundColor: 'white',
          color: 'black'
        }}
      >
        {isGenerating ? (
          <>Queueing...</>
        ) : (
          <>
            <Sparkles size={18} />
            Generate {batchCount > 1 ? `${batchCount} Images` : 'Image'} ({batchCount})
          </>
        )}
      </button>

      {/* Queued confirmation */}
      {lastQueued && (
        <div style={{
          padding: '12px 16px',
          backgroundColor: 'rgba(34, 197, 94, 0.2)',
          border: '1px solid rgba(34, 197, 94, 0.5)',
          borderRadius: '8px',
          color: '#86efac',
          fontSize: '0.875rem',
          marginTop: '12px'
        }}>
          ✅ {lastQueued.count > 1 ? `${lastQueued.count} jobs` : 'Job'} queued! ({lastQueued.model}) - Check queue panel for progress
        </div>
      )}

      {error && (
        <div style={{
          padding: '12px 16px',
          backgroundColor: 'rgba(239, 68, 68, 0.2)',
          border: '1px solid rgba(239, 68, 68, 0.5)',
          borderRadius: '8px',
          color: '#fca5a5',
          fontSize: '0.875rem',
          marginTop: '12px'
        }}>
          {error}
        </div>
      )}
    </div>
  )
}
