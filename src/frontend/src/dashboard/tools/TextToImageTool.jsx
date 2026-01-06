import React, { useState, useEffect, useMemo } from 'react'
import { Sparkles, Settings2, Image as ImageIcon, Info, ChevronDown } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm } from '../../api'
import { useNSFW } from '../../contexts/NSFWContext'
import { useAuth } from '../../contexts/AuthContext'

// Model categories
const MODEL_CATEGORIES = {
  wan22: [
    { value: 'wan2.2-t2i', label: 'Wan2.2 T2I (Multi-GPU)', category: 'Video Model' },
  ],
  flux: [
    { value: 'flux1-dev-fp8', label: 'Flux.1 Dev (FP8)', category: 'Flux' },
  ],
  sdxl: [
    { value: 'CyberRealistic_Pony_v14.1_FP16.safetensors', label: 'CyberRealistic Pony', category: 'Realistic/Pony' },
    { value: 'dreamshaperXL_lightningDPMSDE.safetensors', label: 'Dreamshaper Lightning', category: 'General' },
    { value: 'illustriousRealismBy_v10VAE.safetensors', label: 'Illustrious Realism', category: 'Realistic' },
    { value: 'juggernautXL_ragnarok.safetensors', label: 'Juggernaut XL', category: 'General' },
    { value: 'novaAnimeXL_ilV150.safetensors', label: 'Nova Anime XL', category: 'Anime' },
    { value: 'ponyDiffusionV6XL_v6StartWithThisOne.safetensors', label: 'Pony Diffusion V6', category: 'Pony' },
    { value: 'reapony_v90.safetensors', label: 'Reapony V9', category: 'Realistic/Pony' },
    { value: 'ultraRealisticByStable_v20FP16.safetensors', label: 'Ultra Realistic', category: 'Realistic' },
    { value: 'waiIllustriousSDXL_v160.safetensors', label: 'Wai Illustrious', category: 'Anime' },
  ],
  sd15: [
    { value: 'Realistic_Vision_V5.1.safetensors', label: 'Realistic Vision V5.1', category: 'Realistic' },
  ],
  diffusers: [
    { value: 'sd3.5-large-int8', label: 'SD3.5 Large (INT8)' },
  ],
}

// Determine model type
const getModelType = (model) => {
  if (model === 'wan2.2-t2i') return 'wan22'
  if (model.startsWith('flux')) return 'flux'
  if (model === 'Realistic_Vision_V5.1.safetensors') return 'sd15'
  if (model.endsWith('.safetensors')) return 'sdxl'
  return 'diffusers'
}

export default function TextToImageTool({ onOutput, onJobSubmitted }) {
  const { nsfwEnabled } = useNSFW()
  const { user, requestLogin } = useAuth()

  const [prompt, setPrompt] = useState('')
  const [negativePrompt, setNegativePrompt] = useState('ugly, deformed, blurry, low quality, bad anatomy, watermark, signature, text')
  const [aspectRatio, setAspectRatio] = useState('1:1')
  const [mode, setMode] = useState('normal')
  const [model, setModel] = useState('CyberRealistic_Pony_v14.1_FP16.safetensors')
  const [batchCount, setBatchCount] = useState(1)
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [lastQueued, setLastQueued] = useState(null)

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

      for (let i = 0; i < batchCount; i++) {
        const jobId = `t2i-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
        const formData = new FormData()
        formData.append('prompt', prompt)
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
    const allModels = [
      ...MODEL_CATEGORIES.wan22,
      ...MODEL_CATEGORIES.flux,
      ...MODEL_CATEGORIES.sdxl,
      ...MODEL_CATEGORIES.sd15,
      ...MODEL_CATEGORIES.diffusers
    ]
    const found = allModels.find(m => m.value === model)
    return found?.label || model
  }

  return (
    <div className="tool-container">
      {/* Mode Selection */}
      <div className="grok-card">
        <div className="form-group">
          <label className="grok-section-label">Mode</label>
          <div className="form-select" style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <Sparkles size={16} className="text-primary" />
            <span>Normal</span>
          </div>
          <div className="info-badge">
            <span style={{ color: '#93c5fd' }}>Standard Quality</span>
            <div style={{ marginTop: '4px', opacity: 0.8 }}>Fast and efficient image generation (1 credit per image)</div>
          </div>
        </div>
      </div>

      {/* Prompt Input */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Enter Image Prompt</div>
          <div style={{ display: 'flex', gap: '4px' }}>
             {/* Mock icons for prompt tools */}
             <button className="icon-btn" style={{ width: '24px', height: '24px', fontSize: '10px' }}>T</button>
             <button className="icon-btn" style={{ width: '24px', height: '24px', fontSize: '10px' }}>✨</button>
          </div>
        </div>

        <div style={{ position: 'relative' }}>
          <textarea
            className="form-textarea"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={4}
            placeholder="A attractive blonde woman with cup f, tattoos, looking at me defiantly."
            style={{
              backgroundColor: '#0f0f0f',
              border: 'none',
              resize: 'none',
              paddingBottom: '24px'
            }}
          />
        </div>
      </div>

      {/* Model Selection */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Model</div>
          <span className="nav-badge" style={{ fontSize: '0.7rem' }}>
            {getModelType(model).toUpperCase()}
          </span>
        </div>

        {/* Flux Models */}
        <div style={{ marginBottom: '12px' }}>
          <label className="grok-section-label" style={{ fontSize: '0.75rem', opacity: 0.7, marginBottom: '8px' }}>
            ⚡ Flux (Best Quality)
          </label>
          <div className="grok-toggle-group" style={{ flexWrap: 'wrap', gap: '6px' }}>
            {MODEL_CATEGORIES.flux.map((option) => (
              <button
                key={option.value}
                className={`grok-toggle-btn ${model === option.value ? 'active' : ''}`}
                onClick={() => setModel(option.value)}
                style={{
                  fontSize: '0.75rem',
                  padding: '6px 10px',
                  minWidth: 'auto'
                }}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        {/* SDXL Models (ComfyUI) */}
        <div style={{ marginBottom: '12px' }}>
          <label className="grok-section-label" style={{ fontSize: '0.75rem', opacity: 0.7, marginBottom: '8px' }}>
            🎨 SDXL Checkpoints
          </label>
          <div className="grok-toggle-group" style={{ flexWrap: 'wrap', gap: '6px' }}>
            {MODEL_CATEGORIES.sdxl.map((option) => (
              <button
                key={option.value}
                className={`grok-toggle-btn ${model === option.value ? 'active' : ''}`}
                onClick={() => setModel(option.value)}
                style={{
                  fontSize: '0.75rem',
                  padding: '6px 10px',
                  minWidth: 'auto'
                }}
                title={option.category}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        {/* SD 1.5 Models */}
        <div style={{ marginBottom: '12px' }}>
          <label className="grok-section-label" style={{ fontSize: '0.75rem', opacity: 0.7, marginBottom: '8px' }}>
            🖼️ SD 1.5 (Fast, Low VRAM)
          </label>
          <div className="grok-toggle-group" style={{ flexWrap: 'wrap', gap: '6px' }}>
            {MODEL_CATEGORIES.sd15.map((option) => (
              <button
                key={option.value}
                className={`grok-toggle-btn ${model === option.value ? 'active' : ''}`}
                onClick={() => setModel(option.value)}
                style={{
                  fontSize: '0.75rem',
                  padding: '6px 10px',
                  minWidth: 'auto'
                }}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        {/* Wan2.2 Models (Video Model T2I) */}
        <div style={{ marginBottom: '12px' }}>
          <label className="grok-section-label" style={{ fontSize: '0.75rem', opacity: 0.7, marginBottom: '8px' }}>
            🎬 Wan2.2 (Video Model T2I)
          </label>
          <div className="grok-toggle-group" style={{ flexWrap: 'wrap', gap: '6px' }}>
            {MODEL_CATEGORIES.wan22.map((option) => (
              <button
                key={option.value}
                className={`grok-toggle-btn ${model === option.value ? 'active' : ''}`}
                onClick={() => setModel(option.value)}
                style={{
                  fontSize: '0.75rem',
                  padding: '6px 10px',
                  minWidth: 'auto'
                }}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        {/* Diffusers Models */}
        <div>
          <label className="grok-section-label" style={{ fontSize: '0.75rem', opacity: 0.7, marginBottom: '8px' }}>
            🐍 Diffusers (Python)
          </label>
          <div className="grok-toggle-group" style={{ flexWrap: 'wrap', gap: '6px' }}>
            {MODEL_CATEGORIES.diffusers.map((option) => (
              <button
                key={option.value}
                className={`grok-toggle-btn ${model === option.value ? 'active' : ''}`}
                onClick={() => setModel(option.value)}
                style={{
                  fontSize: '0.75rem',
                  padding: '6px 10px',
                  minWidth: 'auto'
                }}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
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
