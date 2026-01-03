import React, { useState, useCallback, useRef } from 'react'
import { Frame, Upload, Loader2, Download, Copy, Move, ChevronDown } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm, getJson } from '../../api'

const ASPECT_RATIOS = [
  { id: '1:1', label: '1:1 (Square)', width: 1024, height: 1024 },
  { id: '16:9', label: '16:9 (Widescreen)', width: 1280, height: 720 },
  { id: '9:16', label: '9:16 (Portrait)', width: 720, height: 1280 },
  { id: '4:3', label: '4:3 (Standard)', width: 1024, height: 768 },
  { id: '3:4', label: '3:4 (Portrait)', width: 768, height: 1024 },
  { id: '21:9', label: '21:9 (Ultrawide)', width: 1344, height: 576 },
  { id: '3:2', label: '3:2 (Photo)', width: 1152, height: 768 },
  { id: '2:3', label: '2:3 (Photo Portrait)', width: 768, height: 1152 },
]

const POSITIONS = [
  { id: 'center', label: 'Center', icon: '⊕' },
  { id: 'top', label: 'Top', icon: '⬆️' },
  { id: 'bottom', label: 'Bottom', icon: '⬇️' },
  { id: 'left', label: 'Left', icon: '⬅️' },
  { id: 'right', label: 'Right', icon: '➡️' },
  { id: 'top-left', label: 'Top Left', icon: '↖️' },
  { id: 'top-right', label: 'Top Right', icon: '↗️' },
  { id: 'bottom-left', label: 'Bottom Left', icon: '↙️' },
  { id: 'bottom-right', label: 'Bottom Right', icon: '↘️' },
]

const MODELS = [
  { id: 'sdxl', label: 'SDXL (Quality)', file: 'CyberRealisticPony_v8.safetensors' },
  { id: 'flux', label: 'Flux (Fast)', file: 'flux1-dev-bnb-nf4.safetensors' },
]

export default function ReframeTool() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [originalSize, setOriginalSize] = useState({ width: 0, height: 0 })
  const [aspectRatio, setAspectRatio] = useState(ASPECT_RATIOS[0])
  const [position, setPosition] = useState('center')
  const [model, setModel] = useState(MODELS[0])
  const [prompt, setPrompt] = useState('')
  const [steps, setSteps] = useState(25)
  const [cfg, setCfg] = useState(7)
  const [denoise, setDenoise] = useState(0.85)
  const [feathering, setFeathering] = useState(32)
  const [isLoading, setIsLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const fileInputRef = useRef(null)

  const handleFileDrop = useCallback((e) => {
    e.preventDefault()
    const dropped = e.dataTransfer?.files?.[0] || e.target?.files?.[0]
    if (dropped && dropped.type.startsWith('image/')) {
      setFile(dropped)
      setResult(null)
      setError(null)
      
      // Get original dimensions
      const url = URL.createObjectURL(dropped)
      const img = new Image()
      img.onload = () => {
        setOriginalSize({ width: img.naturalWidth, height: img.naturalHeight })
        setPreview(url)
      }
      img.src = url
    }
  }, [])

  const handleDragOver = (e) => e.preventDefault()

  const pollForCompletion = async (promptId) => {
    const maxAttempts = 300
    let attempts = 0
    
    while (attempts < maxAttempts) {
      await new Promise(r => setTimeout(r, 1000))
      attempts++
      setProgress(Math.min(95, attempts * 0.5))
      
      const res = await getJson(`${BACKEND_BASE}/comfyui/job/${promptId}`)
      if (DEBUG) console.log('🔍 Reframe poll:', res.data)
      
      if (res.data?.status === 'completed') {
        return res.data
      } else if (res.data?.status === 'error') {
        throw new Error(res.data?.error || 'Generation failed')
      }
    }
    throw new Error('Generation timed out')
  }

  const handleGenerate = async () => {
    if (!file) {
      setError('Please upload an image first')
      return
    }
    
    setIsLoading(true)
    setProgress(0)
    setError(null)
    setResult(null)
    
    try {
      const formData = new FormData()
      formData.append('image', file)
      formData.append('target_width', aspectRatio.width)
      formData.append('target_height', aspectRatio.height)
      formData.append('position', position)
      formData.append('prompt', prompt || 'seamless natural extension, high quality')
      formData.append('model', model.file)
      formData.append('steps', steps)
      formData.append('cfg', cfg)
      formData.append('denoise', denoise)
      formData.append('feathering', feathering)
      
      if (DEBUG) console.log('🖼️ Reframe request:', {
        target: `${aspectRatio.width}x${aspectRatio.height}`,
        position, model: model.id
      })
      
      const res = await postForm(`${BACKEND_BASE}/reframe`, formData)
      
      if (!res.ok) {
        throw new Error(res.data?.detail || 'Reframe request failed')
      }
      
      if (res.data?.prompt_id) {
        setProgress(5)
        const completed = await pollForCompletion(res.data.prompt_id)
        
        if (completed.images?.length > 0) {
          setResult({
            url: completed.images[0],
            prompt_id: res.data.prompt_id
          })
        } else if (completed.url) {
          setResult({
            url: completed.url,
            prompt_id: res.data.prompt_id
          })
        }
      } else if (res.data?.url) {
        setResult({ url: res.data.url })
      }
      
      setProgress(100)
    } catch (err) {
      console.error('❌ Reframe error:', err)
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleDownload = () => {
    if (!result?.url) return
    const a = document.createElement('a')
    a.href = result.url
    a.download = `reframed_${aspectRatio.id.replace(':', 'x')}_${Date.now()}.png`
    a.click()
  }

  const calculatePreview = () => {
    if (!originalSize.width || !originalSize.height) return null
    
    const targetW = aspectRatio.width
    const targetH = aspectRatio.height
    const origW = originalSize.width
    const origH = originalSize.height
    
    // Scale to fit within target while maintaining aspect
    const scaleW = targetW / origW
    const scaleH = targetH / origH
    const scale = Math.min(scaleW, scaleH)
    
    const scaledW = Math.round(origW * scale)
    const scaledH = Math.round(origH * scale)
    
    // Calculate position offsets
    let offsetX = 0, offsetY = 0
    
    if (position.includes('left')) offsetX = 0
    else if (position.includes('right')) offsetX = targetW - scaledW
    else offsetX = (targetW - scaledW) / 2
    
    if (position.includes('top')) offsetY = 0
    else if (position.includes('bottom')) offsetY = targetH - scaledH
    else offsetY = (targetH - scaledH) / 2
    
    return { scaledW, scaledH, offsetX, offsetY, targetW, targetH }
  }

  const previewLayout = calculatePreview()

  return (
    <div className="space-y-4">
      {/* Image Upload */}
      <div
        onClick={() => fileInputRef.current?.click()}
        onDrop={handleFileDrop}
        onDragOver={handleDragOver}
        className="border-2 border-dashed border-gray-600 rounded-lg p-6 text-center cursor-pointer hover:border-purple-500 transition-colors"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileDrop}
          className="hidden"
        />
        {preview ? (
          <div className="flex flex-col items-center gap-2">
            <img src={preview} alt="Preview" className="max-h-32 rounded" />
            <span className="text-sm text-gray-400">
              Original: {originalSize.width}×{originalSize.height}
            </span>
            <span className="text-xs text-gray-500">Click to change</span>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2 text-gray-400">
            <Upload className="w-8 h-8" />
            <span>Drop image here or click to upload</span>
          </div>
        )}
      </div>

      {/* Target Aspect Ratio */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Target Aspect Ratio
        </label>
        <div className="grid grid-cols-4 gap-2">
          {ASPECT_RATIOS.map(ar => (
            <button
              key={ar.id}
              onClick={() => setAspectRatio(ar)}
              className={`px-3 py-2 text-sm rounded transition-colors ${
                aspectRatio.id === ar.id
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {ar.label}
            </button>
          ))}
        </div>
        <span className="text-xs text-gray-500 mt-1 block">
          Output: {aspectRatio.width}×{aspectRatio.height}
        </span>
      </div>

      {/* Position Control */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Image Position
        </label>
        <div className="grid grid-cols-3 gap-2 w-40 mx-auto">
          {['top-left', 'top', 'top-right', 'left', 'center', 'right', 'bottom-left', 'bottom', 'bottom-right'].map(pos => (
            <button
              key={pos}
              onClick={() => setPosition(pos)}
              className={`p-2 text-lg rounded transition-colors ${
                position === pos
                  ? 'bg-purple-600'
                  : 'bg-gray-700 hover:bg-gray-600'
              }`}
              title={pos}
            >
              {POSITIONS.find(p => p.id === pos)?.icon || '○'}
            </button>
          ))}
        </div>
      </div>

      {/* Preview Layout */}
      {previewLayout && (
        <div className="bg-gray-800 rounded-lg p-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Layout Preview
          </label>
          <div 
            className="relative mx-auto border border-gray-600 bg-gray-900"
            style={{
              width: Math.min(300, previewLayout.targetW / 3),
              height: Math.min(300, previewLayout.targetH / 3),
              aspectRatio: `${previewLayout.targetW} / ${previewLayout.targetH}`
            }}
          >
            {/* Outpaint area (striped) */}
            <div className="absolute inset-0 bg-stripes opacity-30" />
            
            {/* Original image position */}
            <div
              className="absolute bg-purple-600/50 border-2 border-purple-400 flex items-center justify-center text-xs"
              style={{
                width: `${(previewLayout.scaledW / previewLayout.targetW) * 100}%`,
                height: `${(previewLayout.scaledH / previewLayout.targetH) * 100}%`,
                left: `${(previewLayout.offsetX / previewLayout.targetW) * 100}%`,
                top: `${(previewLayout.offsetY / previewLayout.targetH) * 100}%`,
              }}
            >
              Original
            </div>
          </div>
          <p className="text-xs text-gray-500 text-center mt-2">
            Purple = original image, striped = AI-generated fill
          </p>
        </div>
      )}

      {/* Prompt */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Fill Prompt (optional)
        </label>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe what should appear in the extended areas..."
          className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 resize-none"
          rows={2}
        />
      </div>

      {/* Model Selector */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Model
        </label>
        <div className="flex gap-2">
          {MODELS.map(m => (
            <button
              key={m.id}
              onClick={() => setModel(m)}
              className={`flex-1 px-3 py-2 text-sm rounded transition-colors ${
                model.id === m.id
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {/* Advanced Settings */}
      <div className="border border-gray-700 rounded-lg overflow-hidden">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full px-4 py-2 bg-gray-800 flex items-center justify-between text-gray-300 hover:bg-gray-750"
        >
          <span className="text-sm font-medium">Advanced Settings</span>
          <ChevronDown className={`w-4 h-4 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
        </button>
        
        {showAdvanced && (
          <div className="p-4 space-y-4 bg-gray-850">
            {/* Steps */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Steps: {steps}
              </label>
              <input
                type="range"
                min={10}
                max={50}
                value={steps}
                onChange={(e) => setSteps(Number(e.target.value))}
                className="w-full accent-purple-500"
              />
            </div>

            {/* CFG */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                CFG Scale: {cfg}
              </label>
              <input
                type="range"
                min={1}
                max={15}
                step={0.5}
                value={cfg}
                onChange={(e) => setCfg(Number(e.target.value))}
                className="w-full accent-purple-500"
              />
            </div>

            {/* Denoise */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Denoise: {denoise.toFixed(2)}
              </label>
              <input
                type="range"
                min={0.5}
                max={1}
                step={0.05}
                value={denoise}
                onChange={(e) => setDenoise(Number(e.target.value))}
                className="w-full accent-purple-500"
              />
              <span className="text-xs text-gray-500">Higher = more creative fill</span>
            </div>

            {/* Feathering */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Edge Feathering: {feathering}px
              </label>
              <input
                type="range"
                min={0}
                max={64}
                step={8}
                value={feathering}
                onChange={(e) => setFeathering(Number(e.target.value))}
                className="w-full accent-purple-500"
              />
              <span className="text-xs text-gray-500">Blend between original and fill</span>
            </div>
          </div>
        )}
      </div>

      {/* Generate Button */}
      <button
        onClick={handleGenerate}
        disabled={isLoading || !file}
        className="w-full py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg font-semibold flex items-center justify-center gap-2 transition-colors"
      >
        {isLoading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Reframing... {progress > 0 && `${Math.round(progress)}%`}
          </>
        ) : (
          <>
            <Frame className="w-5 h-5" />
            Reframe Image
          </>
        )}
      </button>

      {/* Error */}
      {error && (
        <div className="p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-200 text-sm">
          {error}
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="space-y-3">
          <div className="rounded-lg overflow-hidden border border-gray-700">
            <img src={result.url} alt="Reframed" className="w-full" />
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleDownload}
              className="flex-1 py-2 bg-green-600 hover:bg-green-700 rounded-lg flex items-center justify-center gap-2"
            >
              <Download className="w-4 h-4" />
              Download
            </button>
            <button
              onClick={() => {
                setFile(null)
                setPreview(null)
                setResult(null)
                const blob = fetch(result.url).then(r => r.blob()).then(b => {
                  const f = new File([b], 'reframed.png', { type: 'image/png' })
                  setFile(f)
                  setPreview(result.url)
                  const img = new Image()
                  img.onload = () => setOriginalSize({ width: img.naturalWidth, height: img.naturalHeight })
                  img.src = result.url
                })
              }}
              className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center justify-center gap-2"
            >
              <Move className="w-4 h-4" />
              Use as Input
            </button>
          </div>
        </div>
      )}

      {/* Info */}
      <div className="text-xs text-gray-500 space-y-1">
        <p>💡 <strong>Reframe</strong> extends your image to a new aspect ratio using AI outpainting.</p>
        <p>📐 The original image will be placed according to the position you select.</p>
        <p>🎨 Use the prompt to guide what should appear in the extended areas.</p>
      </div>
    </div>
  )
}
