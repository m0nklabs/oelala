import React, { useRef, useState, useEffect } from 'react'
import { Sparkles, Settings2, Image as ImageIcon, Info, ChevronDown } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm } from '../../api'

// Model categories
const MODEL_CATEGORIES = {
  diffusers: [
    { value: 'sd3.5-large-int8', label: 'SD3.5 Large (INT8)' },
    { value: 'realvisxl-v5.0', label: 'RealVisXL V5.0' },
  ],
  sdxl: [
    { value: 'CyberRealistic_Pony_v14.1_FP16.safetensors', label: 'CyberRealistic Pony', category: 'Realistic/Pony' },
    { value: 'dreamshaperXL_lightningDPMSDE.safetensors', label: 'Dreamshaper XL Lightning', category: 'General' },
    { value: 'illustriousRealismBy_v10VAE.safetensors', label: 'Illustrious Realism', category: 'Realistic' },
    { value: 'juggernautXL_ragnarok.safetensors', label: 'Juggernaut XL Ragnarok', category: 'General' },
    { value: 'novaAnimeXL_ilV150.safetensors', label: 'Nova Anime XL', category: 'Anime' },
    { value: 'ponyDiffusionV6XL_v6StartWithThisOne.safetensors', label: 'Pony Diffusion V6', category: 'Pony' },
    { value: 'reapony_v90.safetensors', label: 'Reapony V9', category: 'Realistic/Pony' },
    { value: 'ultraRealisticByStable_v20FP16.safetensors', label: 'Ultra Realistic', category: 'Realistic' },
    { value: 'waiIllustriousSDXL_v160.safetensors', label: 'Wai Illustrious SDXL', category: 'Anime' },
  ]
}

// Check if model is SDXL (ComfyUI)
const isSDXLModel = (model) => model.endsWith('.safetensors')

export default function TextToImageTool({ onOutput }) {
  const [prompt, setPrompt] = useState('')
  const [negativePrompt, setNegativePrompt] = useState('ugly, deformed, blurry, low quality, bad anatomy, watermark, signature, text')
  const [aspectRatio, setAspectRatio] = useState('1:1')
  const [mode, setMode] = useState('normal')
  const [model, setModel] = useState('CyberRealistic_Pony_v14.1_FP16.safetensors')
  const [batchCount, setBatchCount] = useState(1)
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState('')
  const [progress, setProgress] = useState(0)
  const [showAdvanced, setShowAdvanced] = useState(false)
  
  // Advanced settings for SDXL
  const [steps, setSteps] = useState(30)
  const [cfg, setCfg] = useState(7.5)
  const [seed, setSeed] = useState(-1)
  const [sampler, setSampler] = useState('dpmpp_2m')
  const [scheduler, setScheduler] = useState('karras')
  
  const pollerRef = useRef(null)

  const handleGenerate = async () => {
    if (!prompt.trim()) return
    setIsGenerating(true)
    setProgress(0)
    setError('')

    try {
      for (let i = 0; i < batchCount; i++) {
        const jobId = `t2i-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
        const formData = new FormData()
        formData.append('prompt', prompt)
        formData.append('aspect_ratio', aspectRatio)
        
        // Different endpoint for SDXL vs diffusers models
        const useSDXL = isSDXLModel(model)
        const endpoint = useSDXL ? '/generate-sdxl' : '/generate-image'
        
        if (useSDXL) {
          formData.append('checkpoint', model)
          formData.append('negative_prompt', negativePrompt)
          formData.append('steps', steps)
          formData.append('cfg', cfg)
          formData.append('seed', seed)
          formData.append('sampler_name', sampler)
          formData.append('scheduler', scheduler)
        } else {
          formData.append('mode', mode)
          formData.append('model', model)
          formData.append('job_id', jobId)
          
          // Start polling backend progress for diffusers models
          if (pollerRef.current) {
            clearInterval(pollerRef.current)
            pollerRef.current = null
          }
          pollerRef.current = setInterval(async () => {
            try {
              const res = await fetch(`${BACKEND_BASE}/progress/${jobId}`)
              if (!res.ok) return
              const data = await res.json()
              if (typeof data.progress === 'number') setProgress(data.progress)
              if (data.status === 'done' || data.status === 'failed') {
                clearInterval(pollerRef.current)
                pollerRef.current = null
              }
            } catch (e) {
              // Swallow polling errors
            }
          }, 1000)
        }
        
        if (DEBUG) console.debug('🎨 T2I request:', { endpoint, model, useSDXL })
        
        const result = await postForm(`${BACKEND_BASE}${endpoint}`, formData)
        if (!result.ok) {
          throw new Error(result.data?.detail || `Generation failed (status ${result.status})`)
        }

        console.log(`Batch ${i+1}/${batchCount} success:`, result.data)

        const imageUrl = result.data?.url
        const url = imageUrl ? `${BACKEND_BASE}${imageUrl}` : ''

        setProgress(100)

        onOutput({
          kind: 'image',
          url,
          backendUrl: url,
          filename: result.data?.filename,
          meta: result.data?.meta,
        })
      }
    } catch (e) {
      console.error('Generation error:', e)
      setError(e.message || 'Failed to generate image')
    } finally {
      if (pollerRef.current) {
        clearInterval(pollerRef.current)
        pollerRef.current = null
      }
      setIsGenerating(false)
      setTimeout(() => setProgress(0), 500)
    }
  }
  
  // Get model display label
  const getModelLabel = () => {
    const allModels = [...MODEL_CATEGORIES.diffusers, ...MODEL_CATEGORIES.sdxl]
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
            {isSDXLModel(model) ? 'ComfyUI' : 'Diffusers'}
          </span>
        </div>
        
        {/* SDXL Models (ComfyUI) */}
        <div style={{ marginBottom: '12px' }}>
          <label className="grok-section-label" style={{ fontSize: '0.75rem', opacity: 0.7, marginBottom: '8px' }}>
            SDXL Checkpoints (ComfyUI)
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
        
        {/* Diffusers Models */}
        <div>
          <label className="grok-section-label" style={{ fontSize: '0.75rem', opacity: 0.7, marginBottom: '8px' }}>
            Diffusers (Python)
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

      {/* Negative Prompt (only for SDXL) */}
      {isSDXLModel(model) && (
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
            
            {/* SDXL-specific settings */}
            {isSDXLModel(model) && (
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
          <>Generating...</>
        ) : (
          <>
            <Sparkles size={18} />
            Generate {batchCount > 1 ? `${batchCount} Images` : 'Image'} ({batchCount})
          </>
        )}
      </button>

      {isGenerating && (
        <div className="progress-container">
          <div className="progress-fill" style={{ width: `${Math.min(progress, 100)}%` }}></div>
        </div>
      )}
    </div>
  )
}
