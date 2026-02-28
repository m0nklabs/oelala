import React, { useState, useEffect, useMemo } from 'react'
import { Settings2, Image as ImageIcon, Info, ChevronDown, Wand2, Loader2, Sparkles } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm } from '../../api'
import { useNSFW } from '../../contexts/NSFWContext'
import { useAuth } from '../../contexts/AuthContext'
import CameraPositionSelector, { getCameraPositionPrefix } from '../../components/CameraPositionSelector'
import MediaImportModal from '../../components/MediaImportModal'

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

export default function TextToImageTool({ onOutput, onJobSubmitted, pendingImport = null, onImportConsumed = null }) {
  const { nsfwEnabled } = useNSFW()
  const { user, requestLogin } = useAuth()

  const [prompt, setPrompt] = useState('')
  const [negativePrompt, setNegativePrompt] = useState('ugly, deformed, blurry, low quality, bad anatomy, watermark, signature, text')
  const [aspectRatio, setAspectRatio] = useState('1:1')
  const [model, setModel] = useState('CyberRealistic_Pony_v14.1_FP16.safetensors')
  const [batchCount, setBatchCount] = useState(1)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isEnhancing, setIsEnhancing] = useState(false)
  const [enhanceModel, setEnhanceModel] = useState('GLM-4.7-Flash-Claude-Opus-Reasoning')
  const [error, setError] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [lastQueued, setLastQueued] = useState(null)
  const [cameraPosition, setCameraPosition] = useState('')

  // LoRA settings
  const [availableLoras, setAvailableLoras] = useState([])
  const [selectedLoras, setSelectedLoras] = useState([
    { name: 'None', strength: 1.0 },
    { name: 'None', strength: 1.0 },
    { name: 'None', strength: 1.0 },
  ])

  // Advanced settings
  const [steps, setSteps] = useState(30)
  const [cfg, setCfg] = useState(7.5)
  const [guidance, setGuidance] = useState(3.5)  // For Flux
  const [seed, setSeed] = useState(-1)
  const [sampler, setSampler] = useState('dpmpp_2m')
  const [scheduler, setScheduler] = useState('karras')

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

  // Enhance prompt with LLM
  const handleEnhancePrompt = async () => {
    if (isEnhancing) return
    setIsEnhancing(true)
    setError('')

    // If empty, pick a random subject
    const inputPrompt = prompt.trim() || randomSubjects[Math.floor(Math.random() * randomSubjects.length)]

    try {
      const res = await fetch(`${BACKEND_BASE}/generate-prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: inputPrompt,
          style: null,
          mode: 'expand',
          include_negative: true,
          include_motion: false,
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
      setPrompt(data.prompt)
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

  // Fetch available LoRAs on mount
  useEffect(() => {
    const fetchLoras = async () => {
      try {
        const res = await fetch(`${BACKEND_BASE}/loras`)
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
        const jobId = `t2i-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
        const formData = new FormData()
        formData.append('prompt', finalPrompt)
        formData.append('aspect_ratio', aspectRatio)

        // Determine endpoint based on model type
        const modelType = getModelType(model)
        let endpoint = '/generate-image'

        if (modelType === 'wan22') {
          endpoint = '/generate-wan22-t2i'
          formData.append('steps', steps)
          formData.append('seed', seed)
        } else if (modelType === 'flux') {
          endpoint = '/generate-flux'
          formData.append('steps', steps)
          formData.append('guidance', guidance)
          formData.append('seed', seed)
        } else if (modelType === 'sdxl') {
          endpoint = '/generate-sdxl'
          formData.append('checkpoint', model)
          formData.append('negative_prompt', negativePrompt)
          formData.append('steps', steps)
          formData.append('cfg', cfg)
          formData.append('seed', seed)
          formData.append('sampler_name', sampler)
          formData.append('scheduler', scheduler)
          // Add LoRA configs
          const activeLoras = selectedLoras.filter(l => l.name && l.name !== 'None')
          if (activeLoras.length > 0) {
            formData.append('lora_configs', JSON.stringify(activeLoras))
          }
        } else if (modelType === 'sd15') {
          endpoint = '/generate-sd15'
          formData.append('negative_prompt', negativePrompt)
          formData.append('steps', steps)
          formData.append('cfg', cfg)
          formData.append('seed', seed)
          formData.append('sampler_name', sampler)
          formData.append('scheduler', scheduler)
        } else {
          // Diffusers (legacy)
          formData.append('mode', mode)
          formData.append('model', model)
          formData.append('job_id', jobId)
        }

        if (DEBUG) console.debug('🎨 T2I request:', { endpoint, model, modelType })

        const result = await postForm(`${BACKEND_BASE}${endpoint}`, formData)
        if (!result.ok) {
          throw new Error(result.data?.detail || `Generation failed (status ${result.status})`)
        }

        if (DEBUG) console.log(`📋 Batch ${i+1}/${batchCount} queued:`, result.data)

        // Track queued job
        if (result.data?.prompt_id) {
          queuedJobs.push(result.data.prompt_id)
        }

        // Notify queue indicator
        if (onJobSubmitted) onJobSubmitted({ prompt_id: result.data?.prompt_id })
      }

      // Show queued confirmation
      setLastQueued({
        count: batchCount,
        model: getModelLabel(),
        promptIds: queuedJobs
      })

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
          <div className="grok-card-title">Model</div>
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
              disabled={isEnhancing}
              title={prompt.trim() ? 'Enhance prompt with AI' : 'Generate random prompt with AI'}
            >
              {isEnhancing ? <Loader2 size={12} className="spin" /> : <Wand2 size={12} />}
            </button>
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
              <label className="grok-section-label">Batch Count</label>
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
                    <label className="grok-section-label">Steps</label>
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
                    <label className="grok-section-label">Guidance</label>
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
                  <label className="grok-section-label">Seed (-1 = random)</label>
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
                    <label className="grok-section-label">Steps</label>
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
                  <label className="grok-section-label">Seed (-1 = random)</label>
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
                    <label className="grok-section-label">Steps</label>
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
                    <label className="grok-section-label">CFG Scale</label>
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
                  <label className="grok-section-label">Sampler</label>
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
                  <label className="grok-section-label">Scheduler</label>
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
                  <label className="grok-section-label">Seed (-1 = random)</label>
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
                      LoRAs (up to 3) {!nsfwEnabled && availableLoras.length > filteredLoras.length && (
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
