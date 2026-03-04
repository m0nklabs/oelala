import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { Paintbrush, Eraser, Undo2, Redo2, Loader2, Upload, Wand2, RotateCcw, ZoomIn, ZoomOut } from 'lucide-react'
import { BACKEND_BASE, DEBUG, getMediaUrl } from '../../config'
import { postForm, apiFetch } from '../../api'
import { useAuth } from '../../contexts/AuthContext'
import MediaImportModal from '../../components/MediaImportModal'
import CreationsPickerModal from '../../components/CreationsPickerModal'
import { useToolSettings } from '../../hooks/useToolSettings'
import ResetDefaultsButton from '../../components/ResetDefaultsButton'

const MODELS = [
  { value: 'dreamshaperXL_lightningDPMSDE.safetensors', label: 'Dreamshaper Lightning', desc: 'Fast, artistic' },
  { value: 'CyberRealistic_Pony_v14.1_FP16.safetensors', label: 'CyberRealistic Pony', desc: 'Photorealistic' },
  { value: 'illustriousRealismBy_v10VAE.safetensors', label: 'Illustrious Realism', desc: 'Detailed realistic' },
  { value: 'juggernautXL_ragnarok.safetensors', label: 'Juggernaut XL', desc: 'All-rounder' },
  { value: 'ultraRealisticByStable_v20FP16.safetensors', label: 'Ultra Realistic', desc: 'Hyperrealistic' },
]

const INPAINT_DEFAULTS = {
  tool: 'brush', brushSize: 30, brushOpacity: 1.0,
  prompt: '', negativePrompt: 'ugly, blurry, watermark, text, artifacts',
  model: MODELS[0].value, steps: 20, cfg: 7.0, denoise: 0.85, feathering: 16,
  showAdvanced: false, zoom: 1,
}

export default function InpaintTool({ onOutput, onJobSubmitted, pendingImport, onImportConsumed }) {
  const { user, requestLogin } = useAuth()
  const { initial, save: saveSettings, resetDefaults } = useToolSettings('inpaint', INPAINT_DEFAULTS)

  // Import modal state
  const [importModal, setImportModal] = useState(null)
  const [showCreationsPicker, setShowCreationsPicker] = useState(false)

  // Image state
  const [sourceImage, setSourceImage] = useState(null)
  const [tool, setTool] = useState(initial.tool)
  const [brushSize, setBrushSize] = useState(initial.brushSize)
  const [brushOpacity, setBrushOpacity] = useState(initial.brushOpacity)

  // Canvas refs
  const canvasRef = useRef(null)
  const maskCanvasRef = useRef(null)
  const containerRef = useRef(null)

  // History for undo/redo
  const [history, setHistory] = useState([])
  const [historyIndex, setHistoryIndex] = useState(-1)
  const isDrawing = useRef(false)
  const lastPos = useRef(null)

  // Generation params
  const [prompt, setPrompt] = useState(initial.prompt)
  const [negativePrompt, setNegativePrompt] = useState(initial.negativePrompt)
  const [model, setModel] = useState(initial.model)
  const [steps, setSteps] = useState(initial.steps)
  const [cfg, setCfg] = useState(initial.cfg)
  const [denoise, setDenoise] = useState(initial.denoise)
  const [feathering, setFeathering] = useState(initial.feathering)
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(initial.showAdvanced)

  // Canvas zoom/pan
  const [zoom, setZoom] = useState(initial.zoom)

  // Auto-save settings
  const settingsSnapshot = useMemo(() => ({
    tool, brushSize, brushOpacity, prompt, negativePrompt, model, steps, cfg, denoise, feathering, showAdvanced, zoom
  }), [tool, brushSize, brushOpacity, prompt, negativePrompt, model, steps, cfg, denoise, feathering, showAdvanced, zoom])
  useEffect(() => { saveSettings(settingsSnapshot) }, [settingsSnapshot, saveSettings])

  const handleResetDefaults = useCallback(() => {
    const d = resetDefaults()
    setTool(d.tool); setBrushSize(d.brushSize); setBrushOpacity(d.brushOpacity)
    setPrompt(d.prompt); setNegativePrompt(d.negativePrompt); setModel(d.model)
    setSteps(d.steps); setCfg(d.cfg); setDenoise(d.denoise); setFeathering(d.feathering)
    setShowAdvanced(d.showAdvanced); setZoom(d.zoom)
  }, [resetDefaults])

  // Auto-open import modal when Dashboard sends a pendingImport
  useEffect(() => {
    if (!pendingImport) return
    setImportModal(pendingImport)
    if (onImportConsumed) onImportConsumed()
  }, [pendingImport])

  const handleApplyImport = async (selected) => {
    if (selected.image && importModal?.item) {
      const item = importModal.item

      // If item is a video, use the companion .png (first frame) instead
      let imageUrl
      if (item.type === 'video' && item.filename?.match(/\.(mp4|webm|mov)$/i)) {
        imageUrl = item.url?.replace(/\.(mp4|webm|mov)$/i, '.png')
        console.debug('🎬 Inpaint: video detected, using companion image')
      } else {
        imageUrl = getMediaUrl(item.url, item.signed_url)
      }

      try {
        const response = await apiFetch(imageUrl)
        if (!response.ok) throw new Error(`Failed to fetch image: ${response.status}`)
        const blob = await response.blob()
        const filename = imageUrl.split('/').pop() || 'image.png'
        const fileObj = new File([blob], filename, { type: blob.type || 'image/png' })
        const url = URL.createObjectURL(fileObj)
        const img = new Image()
        img.onload = () => {
          setSourceImage({ url, width: img.width, height: img.height, file: fileObj })
          setHistory([])
          setHistoryIndex(-1)
          setZoom(1)
          if (DEBUG) console.log('🎨 Inpaint imported image:', filename)
        }
        img.src = url
      } catch (e) {
        console.error('Failed to load image from import:', e)
        setError('⚠️ Failed to load image from import')
      }
    }
    if (selected.positive)  setPrompt(String(selected.positive))
    if (selected.negative)  setNegativePrompt(String(selected.negative))
    setImportModal(null)
  }

  // Initialize canvas when image is loaded
  useEffect(() => {
    if (!sourceImage || !canvasRef.current || !maskCanvasRef.current) return

    const canvas = canvasRef.current
    const maskCanvas = maskCanvasRef.current
    const ctx = canvas.getContext('2d')
    const maskCtx = maskCanvas.getContext('2d')

    canvas.width = sourceImage.width
    canvas.height = sourceImage.height
    maskCanvas.width = sourceImage.width
    maskCanvas.height = sourceImage.height

    // Draw source image
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      ctx.drawImage(img, 0, 0)
      // Save initial state
      saveHistory()
    }
    img.src = sourceImage.url

    // Clear mask (all black = keep everything)
    maskCtx.fillStyle = '#000000'
    maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height)
  }, [sourceImage]) // eslint-disable-line react-hooks/exhaustive-deps

  const saveHistory = useCallback(() => {
    if (!maskCanvasRef.current) return
    const maskCtx = maskCanvasRef.current.getContext('2d')
    const data = maskCtx.getImageData(0, 0, maskCanvasRef.current.width, maskCanvasRef.current.height)
    const newHistory = history.slice(0, historyIndex + 1)
    newHistory.push(data)
    // Limit history to 30 states
    if (newHistory.length > 30) newHistory.shift()
    setHistory(newHistory)
    setHistoryIndex(newHistory.length - 1)
  }, [history, historyIndex])

  const undo = useCallback(() => {
    if (historyIndex <= 0) return
    const newIndex = historyIndex - 1
    const maskCtx = maskCanvasRef.current.getContext('2d')
    maskCtx.putImageData(history[newIndex], 0, 0)
    setHistoryIndex(newIndex)
    redrawOverlay()
  }, [historyIndex, history]) // eslint-disable-line react-hooks/exhaustive-deps

  const redo = useCallback(() => {
    if (historyIndex >= history.length - 1) return
    const newIndex = historyIndex + 1
    const maskCtx = maskCanvasRef.current.getContext('2d')
    maskCtx.putImageData(history[newIndex], 0, 0)
    setHistoryIndex(newIndex)
    redrawOverlay()
  }, [historyIndex, history]) // eslint-disable-line react-hooks/exhaustive-deps

  // Redraw the visible canvas: source image + mask overlay
  const redrawOverlay = useCallback(() => {
    if (!canvasRef.current || !maskCanvasRef.current || !sourceImage) return
    const ctx = canvasRef.current.getContext('2d')
    const maskCanvas = maskCanvasRef.current

    // Redraw source image
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      ctx.drawImage(img, 0, 0)

      // Draw mask overlay (semi-transparent red where white in mask)
      ctx.save()
      ctx.globalAlpha = 0.4
      ctx.globalCompositeOperation = 'source-atop'

      // Create overlay from mask
      const maskCtx = maskCanvas.getContext('2d')
      const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height)
      const overlayCanvas = document.createElement('canvas')
      overlayCanvas.width = maskCanvas.width
      overlayCanvas.height = maskCanvas.height
      const overlayCtx = overlayCanvas.getContext('2d')
      const overlayData = overlayCtx.createImageData(maskCanvas.width, maskCanvas.height)

      for (let i = 0; i < maskData.data.length; i += 4) {
        const val = maskData.data[i] // R channel
        if (val > 128) {
          // White = inpaint area → show as red overlay
          overlayData.data[i] = 255     // R
          overlayData.data[i + 1] = 50  // G
          overlayData.data[i + 2] = 50  // B
          overlayData.data[i + 3] = 160 // A
        } else {
          overlayData.data[i + 3] = 0 // Transparent
        }
      }
      overlayCtx.putImageData(overlayData, 0, 0)

      ctx.globalAlpha = 1
      ctx.globalCompositeOperation = 'source-over'
      ctx.drawImage(overlayCanvas, 0, 0)
      ctx.restore()
    }
    img.src = sourceImage.url
  }, [sourceImage])

  // Get canvas-relative position from mouse/touch event
  const getCanvasPos = useCallback((e) => {
    if (!canvasRef.current) return null
    const rect = canvasRef.current.getBoundingClientRect()
    const scaleX = canvasRef.current.width / rect.width
    const scaleY = canvasRef.current.height / rect.height
    const clientX = e.touches ? e.touches[0].clientX : e.clientX
    const clientY = e.touches ? e.touches[0].clientY : e.clientY
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    }
  }, [])

  // Draw on mask canvas
  const drawStroke = useCallback((from, to) => {
    if (!maskCanvasRef.current) return
    const maskCtx = maskCanvasRef.current.getContext('2d')

    maskCtx.lineWidth = brushSize
    maskCtx.lineCap = 'round'
    maskCtx.lineJoin = 'round'
    maskCtx.globalAlpha = brushOpacity

    if (tool === 'brush') {
      maskCtx.strokeStyle = '#ffffff'
      maskCtx.globalCompositeOperation = 'source-over'
    } else {
      maskCtx.strokeStyle = '#000000'
      maskCtx.globalCompositeOperation = 'source-over'
    }

    maskCtx.beginPath()
    maskCtx.moveTo(from.x, from.y)
    maskCtx.lineTo(to.x, to.y)
    maskCtx.stroke()

    maskCtx.globalAlpha = 1
    redrawOverlay()
  }, [brushSize, brushOpacity, tool, redrawOverlay])

  // Mouse/touch event handlers
  const handlePointerDown = useCallback((e) => {
    e.preventDefault()
    const pos = getCanvasPos(e)
    if (!pos) return
    isDrawing.current = true
    lastPos.current = pos
    // Draw a dot at the start
    drawStroke(pos, pos)
  }, [getCanvasPos, drawStroke])

  const handlePointerMove = useCallback((e) => {
    e.preventDefault()
    if (!isDrawing.current || !lastPos.current) return
    const pos = getCanvasPos(e)
    if (!pos) return
    drawStroke(lastPos.current, pos)
    lastPos.current = pos
  }, [getCanvasPos, drawStroke])

  const handlePointerUp = useCallback(() => {
    if (isDrawing.current) {
      isDrawing.current = false
      lastPos.current = null
      saveHistory()
    }
  }, [saveHistory])

  const handleCreationsSelect = useCallback(async (item) => {
    try {
      let imageUrl
      if (item.type === 'video' && item.filename?.match(/\.(mp4|webm|mov)$/i)) {
        imageUrl = item.url?.replace(/\.(mp4|webm|mov)$/i, '.png')
      } else {
        imageUrl = getMediaUrl(item.url, item.signed_url)
      }
      const response = await apiFetch(imageUrl)
      if (!response.ok) throw new Error(`Failed to fetch: ${response.status}`)
      const blob = await response.blob()
      const filename = imageUrl.split('/').pop() || 'image.png'
      const fileObj = new File([blob], filename, { type: blob.type || 'image/png' })
      const url = URL.createObjectURL(fileObj)
      const img = new Image()
      img.onload = () => {
        setSourceImage({ url, width: img.width, height: img.height, file: fileObj })
        setHistory([])
        setHistoryIndex(-1)
        setZoom(1)
        if (DEBUG) console.log('\ud83d\udcc1 Inpaint: loaded from creations:', filename)
      }
      img.src = url
    } catch (e) {
      console.error('Failed to load from creations:', e)
      setError('\u26a0\ufe0f Failed to load image from My Creations')
    }
  }, [])

  // Handle image upload
  const handleImageUpload = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => {
      setSourceImage({ url, width: img.width, height: img.height, file })
      setHistory([])
      setHistoryIndex(-1)
      setZoom(1)
    }
    img.src = url
  }

  // Clear mask
  const clearMask = () => {
    if (!maskCanvasRef.current) return
    const maskCtx = maskCanvasRef.current.getContext('2d')
    maskCtx.fillStyle = '#000000'
    maskCtx.fillRect(0, 0, maskCanvasRef.current.width, maskCanvasRef.current.height)
    saveHistory()
    redrawOverlay()
  }

  // Generate (submit inpaint request)
  const handleGenerate = async () => {
    if (!user) { requestLogin(); return }
    if (!sourceImage || !prompt.trim()) {
      setError('Please upload an image and enter a prompt')
      return
    }

    setGenerating(true)
    setError('')

    try {
      // Export mask as PNG blob
      const maskBlob = await new Promise(resolve => {
        maskCanvasRef.current.toBlob(resolve, 'image/png')
      })

      const formData = new FormData()
      formData.append('image', sourceImage.file)
      formData.append('mask', maskBlob, 'mask.png')
      formData.append('prompt', prompt)
      formData.append('negative_prompt', negativePrompt)
      formData.append('model', model)
      formData.append('steps', String(steps))
      formData.append('cfg', String(cfg))
      formData.append('denoise', String(denoise))
      formData.append('feathering', String(feathering))

      const { ok, data } = await postForm(BACKEND_BASE + '/inpaint', formData)

      if (ok && data?.prompt_id) {
        if (DEBUG) console.log('🎨 Inpaint queued:', data.prompt_id)
        if (onJobSubmitted) onJobSubmitted()
        setError('')
      } else {
        setError(data?.detail || 'Failed to queue inpaint job')
      }
    } catch (err) {
      console.error('❌ Inpaint error:', err)
      setError(err.message || 'Unknown error')
    } finally {
      setGenerating(false)
    }
  }

  // Keyboard shortcuts
  useEffect(() => {
    const handleKey = (e) => {
      if (e.ctrlKey && e.key === 'z') { e.preventDefault(); undo() }
      if (e.ctrlKey && e.key === 'y') { e.preventDefault(); redo() }
      if (e.key === 'b') setTool('brush')
      if (e.key === 'e') setTool('eraser')
      if (e.key === '[') setBrushSize(s => Math.max(2, s - 5))
      if (e.key === ']') setBrushSize(s => Math.min(200, s + 5))
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [undo, redo])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', height: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: '4px' }}>
        <ResetDefaultsButton onReset={handleResetDefaults} />
      </div>
      <style>{`
        .inpaint-tool .toolbar {
          display: flex; align-items: center; gap: 8px; padding: 8px 12px;
          background: #1e1e1e; border-radius: 10px; border: 1px solid #333;
          flex-wrap: wrap;
        }
        .inpaint-tool .toolbar button {
          padding: 8px; background: #2a2a2a; border: 1px solid #444;
          border-radius: 6px; color: #ccc; cursor: pointer; display: flex;
          align-items: center; gap: 4px; font-size: 13px;
        }
        .inpaint-tool .toolbar button:hover { background: #333; }
        .inpaint-tool .toolbar button.active {
          background: #6366f1; border-color: #818cf8; color: #fff;
        }
        .inpaint-tool .toolbar button:disabled { opacity: 0.4; cursor: not-allowed; }
        .inpaint-tool .slider-group {
          display: flex; align-items: center; gap: 6px; font-size: 12px; color: #999;
        }
        .inpaint-tool .slider-group input[type="range"] {
          width: 80px; accent-color: #6366f1;
        }
        .inpaint-tool .canvas-container {
          position: relative; flex: 1; min-height: 300px;
          background: #111; border-radius: 10px; overflow: hidden;
          display: flex; align-items: center; justify-content: center;
          border: 2px dashed #333;
        }
        .inpaint-tool .canvas-container.has-image { border: 1px solid #333; }
        .inpaint-tool canvas {
          cursor: crosshair; max-width: 100%; max-height: 100%;
          image-rendering: auto;
        }
        .inpaint-tool .upload-prompt {
          text-align: center; color: #666; padding: 40px;
        }
        .inpaint-tool .upload-prompt label {
          display: inline-flex; align-items: center; gap: 8px;
          padding: 12px 24px; background: linear-gradient(135deg, #6366f1, #8b5cf6);
          border-radius: 10px; color: #fff; font-weight: 600;
          cursor: pointer; font-size: 15px;
        }
        .inpaint-tool .upload-prompt label:hover { opacity: 0.9; }
        .inpaint-tool .controls-panel {
          display: flex; flex-direction: column; gap: 10px;
          background: #1a1a1a; border-radius: 10px; padding: 14px;
          border: 1px solid #333;
        }
        .inpaint-tool .controls-panel label {
          font-size: 12px; color: #999; font-weight: 500;
        }
        .inpaint-tool .controls-panel input[type="text"],
        .inpaint-tool .controls-panel textarea,
        .inpaint-tool .controls-panel select {
          width: 100%; padding: 8px 10px; background: #2a2a2a;
          border: 1px solid #444; border-radius: 6px; color: #eee;
          font-size: 13px; resize: vertical;
        }
        .inpaint-tool .controls-panel textarea { min-height: 60px; }
        .inpaint-tool .generate-btn {
          padding: 12px 24px; font-size: 15px; font-weight: 600;
          border: none; border-radius: 10px; cursor: pointer;
          display: flex; align-items: center; justify-content: center; gap: 8px;
          background: linear-gradient(135deg, #6366f1, #8b5cf6);
          color: #fff; width: 100%;
        }
        .inpaint-tool .generate-btn:hover { opacity: 0.9; }
        .inpaint-tool .generate-btn:disabled { opacity: 0.5; cursor: wait; }
        .inpaint-tool .error { color: #ef4444; font-size: 13px; padding: 6px 0; }
        .inpaint-tool .advanced-toggle {
          font-size: 12px; color: #888; cursor: pointer; text-align: center;
          padding: 4px; user-select: none;
        }
        .inpaint-tool .advanced-toggle:hover { color: #aaa; }
        .inpaint-tool .advanced-grid {
          display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
        }
        .inpaint-tool .advanced-grid .field {
          display: flex; flex-direction: column; gap: 4px;
        }
        .inpaint-tool .advanced-grid input[type="number"],
        .inpaint-tool .advanced-grid input[type="range"] {
          width: 100%; padding: 6px; background: #2a2a2a;
          border: 1px solid #444; border-radius: 4px; color: #eee; font-size: 12px;
        }
        .inpaint-tool .shortcut-hint {
          font-size: 11px; color: #555; text-align: center; padding: 2px 0;
        }
      `}</style>

      {importModal && (
        <MediaImportModal
          item={importModal.item}
          parsedData={importModal.workflow || {}}
          availableFields={['image', 'positive', 'negative']}
          onApply={handleApplyImport}
          onClose={() => setImportModal(null)}
        />
      )}

      <div className="inpaint-tool" style={{ display: 'flex', gap: '16px', height: '100%' }}>
        {/* Left: Canvas area */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '8px', minWidth: 0 }}>
          {/* Toolbar */}
          {sourceImage && (
            <div className="toolbar">
              <button
                className={tool === 'brush' ? 'active' : ''}
                onClick={() => setTool('brush')}
                title="Brush (B) — paint area to regenerate"
              >
                <Paintbrush size={16} /> Brush
              </button>
              <button
                className={tool === 'eraser' ? 'active' : ''}
                onClick={() => setTool('eraser')}
                title="Eraser (E) — remove from selection"
              >
                <Eraser size={16} /> Eraser
              </button>

              <div style={{ width: 1, height: 24, background: '#444' }} />

              <div className="slider-group">
                Size
                <input
                  type="range" min="2" max="200" value={brushSize}
                  onChange={e => setBrushSize(Number(e.target.value))}
                />
                <span style={{ minWidth: 28 }}>{brushSize}</span>
              </div>

              <div className="slider-group">
                Opacity
                <input
                  type="range" min="0.1" max="1" step="0.1" value={brushOpacity}
                  onChange={e => setBrushOpacity(Number(e.target.value))}
                />
              </div>

              <div style={{ width: 1, height: 24, background: '#444' }} />

              <button onClick={undo} disabled={historyIndex <= 0} title="Undo (Ctrl+Z)">
                <Undo2 size={16} />
              </button>
              <button onClick={redo} disabled={historyIndex >= history.length - 1} title="Redo (Ctrl+Y)">
                <Redo2 size={16} />
              </button>
              <button onClick={clearMask} title="Clear mask">
                <RotateCcw size={16} /> Clear
              </button>

              <div style={{ width: 1, height: 24, background: '#444' }} />

              <button onClick={() => setZoom(z => Math.min(3, z + 0.25))} title="Zoom in">
                <ZoomIn size={16} />
              </button>
              <button onClick={() => setZoom(z => Math.max(0.25, z - 0.25))} title="Zoom out">
                <ZoomOut size={16} />
              </button>
              <span style={{ fontSize: 12, color: '#888' }}>{Math.round(zoom * 100)}%</span>

              <div style={{ flex: 1 }} />

              <label style={{
                padding: '6px 12px', background: '#2a2a2a', border: '1px solid #444',
                borderRadius: '6px', color: '#ccc', cursor: 'pointer', fontSize: 13,
                display: 'flex', alignItems: 'center', gap: 4,
              }}>
                <Upload size={14} /> New Image
                <input type="file" accept="image/*" onChange={handleImageUpload}
                  style={{ display: 'none' }} />
              </label>
            </div>
          )}

          {/* Canvas */}
          <div
            ref={containerRef}
            className={`canvas-container ${sourceImage ? 'has-image' : ''}`}
          >
            {!sourceImage ? (
              <div className="upload-prompt">
                <p style={{ marginBottom: 12, fontSize: 15 }}>Upload an image to start inpainting</p>
                <label>
                  <Upload size={18} /> Choose Image
                  <input type="file" accept="image/*" onChange={handleImageUpload}
                    style={{ display: 'none' }} />
                </label>
                <button
                  onClick={() => setShowCreationsPicker(true)}
                  style={{
                    marginTop: 12, padding: '8px 16px',
                    backgroundColor: 'var(--bg-tertiary, #2a2a2a)',
                    border: '1px solid var(--border-color, #444)',
                    borderRadius: 8, cursor: 'pointer',
                    color: 'var(--text-primary, #eee)', fontSize: 13,
                  }}
                >
                  {'\ud83d\udcc1'} From My Creations
                </button>

                <CreationsPickerModal
                  show={showCreationsPicker}
                  onClose={() => setShowCreationsPicker(false)}
                  onSelect={handleCreationsSelect}
                  filter="image"
                  title="Select Image for Inpainting"
                />

                <p style={{ marginTop: 16, fontSize: 12 }}>
                  Paint over the areas you want to regenerate, then describe what should appear.
                </p>
              </div>
            ) : (
              <canvas
                ref={canvasRef}
                style={{ transform: `scale(${zoom})`, transformOrigin: 'center center' }}
                onMouseDown={handlePointerDown}
                onMouseMove={handlePointerMove}
                onMouseUp={handlePointerUp}
                onMouseLeave={handlePointerUp}
                onTouchStart={handlePointerDown}
                onTouchMove={handlePointerMove}
                onTouchEnd={handlePointerUp}
              />
            )}
            {/* Hidden mask canvas */}
            <canvas ref={maskCanvasRef} style={{ display: 'none' }} />
          </div>

          {sourceImage && (
            <div className="shortcut-hint">
              <strong>B</strong> Brush · <strong>E</strong> Eraser · <strong>[ ]</strong> Size · <strong>Ctrl+Z</strong> Undo · <strong>Ctrl+Y</strong> Redo
            </div>
          )}
        </div>

        {/* Right: Controls panel */}
        <div style={{ width: 300, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div className="controls-panel">
            <label>Prompt *</label>
            <textarea
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              placeholder="Describe what should appear in the painted area..."
            />

            <label>Model</label>
            <select value={model} onChange={e => setModel(e.target.value)}>
              {MODELS.map(m => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>

            <label>Denoise Strength ({denoise})</label>
            <input
              type="range" min="0.1" max="1" step="0.05" value={denoise}
              onChange={e => setDenoise(Number(e.target.value))}
              style={{ accentColor: '#6366f1' }}
            />
            <span style={{ fontSize: 11, color: '#666' }}>
              Lower = subtle changes, Higher = complete regeneration
            </span>

            <div className="advanced-toggle" onClick={() => setShowAdvanced(!showAdvanced)}>
              {showAdvanced ? '▲ Hide advanced' : '▼ Show advanced'}
            </div>

            {showAdvanced && (
              <>
                <label>Negative Prompt</label>
                <textarea
                  value={negativePrompt}
                  onChange={e => setNegativePrompt(e.target.value)}
                  placeholder="What to avoid..."
                  style={{ minHeight: 40 }}
                />

                <div className="advanced-grid">
                  <div className="field">
                    <label>Steps</label>
                    <input type="number" value={steps} min={1} max={50}
                      onChange={e => setSteps(Number(e.target.value))} />
                  </div>
                  <div className="field">
                    <label>CFG</label>
                    <input type="number" value={cfg} min={1} max={20} step={0.5}
                      onChange={e => setCfg(Number(e.target.value))} />
                  </div>
                  <div className="field">
                    <label>Feathering</label>
                    <input type="number" value={feathering} min={0} max={64}
                      onChange={e => setFeathering(Number(e.target.value))} />
                  </div>
                </div>
              </>
            )}
          </div>

          {error && <div className="error">⚠️ {error}</div>}

          <button
            className="generate-btn"
            onClick={handleGenerate}
            disabled={generating || !sourceImage || !prompt.trim()}
          >
            {generating ? (
              <><Loader2 size={18} className="spin" /> Generating...</>
            ) : (
              <><Wand2 size={18} /> Inpaint (8 credits)</>
            )}
          </button>

          <div style={{ fontSize: 12, color: '#666', lineHeight: 1.5 }}>
            <strong>How to use:</strong>
            <ol style={{ margin: '4px 0 0 16px', padding: 0 }}>
              <li>Upload an image</li>
              <li>Paint over the area to change (shown in red)</li>
              <li>Describe what should appear there</li>
              <li>Click Inpaint to generate</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  )
}
