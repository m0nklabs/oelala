import React, { useState, useCallback, useRef } from 'react'
import { Upload, Video, Loader2, Settings, ChevronDown, Wand2 } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm } from '../../api'

// Style presets for V2V
const STYLE_PRESETS = [
  { value: 'none', label: 'Custom', desc: 'Use your own prompt' },
  { value: 'anime', label: 'Anime', desc: 'Japanese animation style' },
  { value: 'cartoon', label: 'Cartoon', desc: 'Cartoon/comic style' },
  { value: 'sketch', label: 'Sketch', desc: 'Pencil sketch effect' },
  { value: 'oil-painting', label: 'Oil Painting', desc: 'Classic oil painting style' },
  { value: 'watercolor', label: 'Watercolor', desc: 'Watercolor painting effect' },
  { value: 'pixel-art', label: 'Pixel Art', desc: 'Retro pixel art style' },
  { value: 'cyberpunk', label: 'Cyberpunk', desc: 'Neon futuristic style' },
  { value: '3d-render', label: '3D Render', desc: 'Modern 3D rendered look' },
]

const STYLE_PROMPTS = {
  'anime': 'anime style, japanese animation, cel shading, vibrant colors, detailed linework',
  'cartoon': 'cartoon style, comic art, bold outlines, bright colors, disney style',
  'sketch': 'pencil sketch, hand-drawn, graphite, detailed linework, black and white',
  'oil-painting': 'oil painting style, classical art, brush strokes, rich colors, masterpiece',
  'watercolor': 'watercolor painting, soft edges, translucent colors, artistic, flowing',
  'pixel-art': 'pixel art style, 8-bit, retro gaming, blocky, nostalgic',
  'cyberpunk': 'cyberpunk style, neon lights, futuristic, rain, dark atmosphere, high tech',
  '3d-render': '3d render, modern cgi, photorealistic, octane render, unreal engine',
}

export default function VideoToVideoTool({ onOutput, onJobSubmitted }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [videoInfo, setVideoInfo] = useState(null)
  
  const [style, setStyle] = useState('none')
  const [prompt, setPrompt] = useState('')
  const [negativePrompt, setNegativePrompt] = useState('blurry, low quality, distorted, watermark')
  const [denoise, setDenoise] = useState(0.5)
  const [fps, setFps] = useState(8)
  const [maxFrames, setMaxFrames] = useState(32)
  
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [steps, setSteps] = useState(20)
  const [cfg, setCfg] = useState(7.5)
  const [seed, setSeed] = useState(-1)
  
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)
  const [lastQueued, setLastQueued] = useState(null)
  const [result, setResult] = useState(null)
  
  const videoRef = useRef(null)

  const handleFileChange = useCallback((e) => {
    const f = e.target.files?.[0]
    if (f) {
      setFile(f)
      const url = URL.createObjectURL(f)
      setPreview(url)
      setResult(null)
      setError(null)
      setLastQueued(null)
      
      // Get video info
      const video = document.createElement('video')
      video.onloadedmetadata = () => {
        setVideoInfo({
          duration: video.duration.toFixed(1),
          width: video.videoWidth,
          height: video.videoHeight,
        })
      }
      video.src = url
    }
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0]
    if (f && f.type.startsWith('video/')) {
      setFile(f)
      const url = URL.createObjectURL(f)
      setPreview(url)
      setResult(null)
      setError(null)
      setLastQueued(null)
      
      const video = document.createElement('video')
      video.onloadedmetadata = () => {
        setVideoInfo({
          duration: video.duration.toFixed(1),
          width: video.videoWidth,
          height: video.videoHeight,
        })
      }
      video.src = url
    }
  }, [])

  const handleTransform = async () => {
    if (!file) return
    
    // Determine final prompt
    const finalPrompt = style !== 'none' 
      ? STYLE_PROMPTS[style] + (prompt ? ', ' + prompt : '')
      : prompt
    
    if (!finalPrompt.trim()) {
      setError('Please select a style or enter a prompt')
      return
    }
    
    setSubmitting(true)
    setError(null)
    setLastQueued(null)
    
    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('prompt', finalPrompt)
      formData.append('negative_prompt', negativePrompt)
      formData.append('denoise', String(denoise))
      formData.append('fps', String(fps))
      formData.append('max_frames', String(maxFrames))
      formData.append('steps', String(steps))
      formData.append('cfg', String(cfg))
      formData.append('seed', String(seed))
      
      if (DEBUG) console.debug('🎬 V2V request:', { style, denoise, fps, maxFrames })
      
      const res = await postForm(`${BACKEND_BASE}/generate-v2v`, formData)
      
      if (!res.ok) {
        throw new Error(res.data?.detail || 'V2V transform failed')
      }
      
      const promptId = res.data?.prompt_id
      if (!promptId) {
        throw new Error('No prompt_id returned')
      }
      
      // Show queued confirmation
      setLastQueued({
        promptId,
        style: style !== 'none' ? style : 'custom'
      })
      
      // Notify queue indicator
      if (onJobSubmitted) onJobSubmitted({ prompt_id: promptId })
      
      if (DEBUG) console.debug('📋 V2V queued:', promptId)
      
      // Don't wait for completion - job will appear in queue/history when done
      
    } catch (err) {
      console.error('V2V error:', err)
      setError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="tool-container">
      <div className="tool-section">
        <h3>
          <Video size={18} />
          Source Video
        </h3>
        
        <div
          className={`upload-dropzone ${preview ? 'has-preview' : ''}`}
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => document.getElementById('v2v-file-input').click()}
        >
          {preview ? (
            <video 
              ref={videoRef}
              src={preview} 
              className="upload-preview" 
              controls 
              muted 
              loop
              style={{ maxHeight: '250px' }}
            />
          ) : (
            <div className="upload-placeholder">
              <Upload size={32} />
              <p>Drop video here or click to upload</p>
              <span style={{ fontSize: '12px', opacity: 0.6 }}>MP4, WebM, MOV</span>
            </div>
          )}
          <input
            id="v2v-file-input"
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
        </div>
        
        {videoInfo && (
          <div className="video-info">
            <span>📐 {videoInfo.width} × {videoInfo.height}px</span>
            <span>⏱️ {videoInfo.duration}s</span>
          </div>
        )}
      </div>

      {/* Style Selection */}
      <div className="tool-section">
        <h3>
          <Wand2 size={18} />
          Style Transform
        </h3>
        
        <div className="style-grid">
          {STYLE_PRESETS.map((preset) => (
            <button
              key={preset.value}
              className={`style-btn ${style === preset.value ? 'active' : ''}`}
              onClick={() => setStyle(preset.value)}
            >
              <span className="style-name">{preset.label}</span>
              <span className="style-desc">{preset.desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Prompt */}
      <div className="tool-section">
        <h3>Prompt {style !== 'none' && <span className="hint">(optional - adds to style)</span>}</h3>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder={style !== 'none' 
            ? 'Add extra details to the style...'
            : 'Describe the desired look...'}
          rows={3}
          className="prompt-textarea"
        />
      </div>

      {/* Strength */}
      <div className="tool-section">
        <h3>Transform Strength</h3>
        <div className="slider-row">
          <input
            type="range"
            min="0.1"
            max="1"
            step="0.05"
            value={denoise}
            onChange={(e) => setDenoise(parseFloat(e.target.value))}
          />
          <span className="slider-value">{(denoise * 100).toFixed(0)}%</span>
        </div>
        <div className="slider-labels">
          <span>Subtle</span>
          <span>Complete</span>
        </div>
      </div>

      {/* Advanced */}
      <div className="tool-section collapsible">
        <h3 
          onClick={() => setShowAdvanced(!showAdvanced)}
          style={{ cursor: 'pointer' }}
        >
          <Settings size={16} />
          Advanced Settings
          <ChevronDown 
            size={16} 
            style={{ 
              marginLeft: 'auto',
              transform: showAdvanced ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.2s'
            }}
          />
        </h3>
        
        {showAdvanced && (
          <div className="advanced-content">
            <div className="form-row">
              <label>Output FPS</label>
              <select value={fps} onChange={(e) => setFps(parseInt(e.target.value))}>
                <option value={8}>8 fps</option>
                <option value={12}>12 fps</option>
                <option value={16}>16 fps</option>
                <option value={24}>24 fps</option>
              </select>
            </div>
            
            <div className="form-row">
              <label>Max Frames</label>
              <select value={maxFrames} onChange={(e) => setMaxFrames(parseInt(e.target.value))}>
                <option value={16}>16 frames (~2s @8fps)</option>
                <option value={32}>32 frames (~4s @8fps)</option>
                <option value={48}>48 frames (~6s @8fps)</option>
                <option value={64}>64 frames (~8s @8fps)</option>
              </select>
            </div>
            
            <div className="form-row">
              <label>Steps</label>
              <input
                type="number"
                min={10}
                max={50}
                value={steps}
                onChange={(e) => setSteps(parseInt(e.target.value))}
              />
            </div>
            
            <div className="form-row">
              <label>CFG Scale</label>
              <input
                type="number"
                min={1}
                max={15}
                step={0.5}
                value={cfg}
                onChange={(e) => setCfg(parseFloat(e.target.value))}
              />
            </div>
            
            <div className="form-row">
              <label>Seed (-1 = random)</label>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(parseInt(e.target.value) || -1)}
              />
            </div>
            
            <div className="form-row">
              <label>Negative Prompt</label>
              <textarea
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
                rows={2}
                style={{ fontSize: '12px' }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Queued notification */}
      {lastQueued && (
        <div className="queued-notice">
          ✅ Job queued! Check the Queue panel for progress.
          <span className="queued-mode">{lastQueued.style.toUpperCase()}</span>
        </div>
      )}

      {error && <div className="error-message">⚠️ {error}</div>}

      <button
        className="btn-primary btn-large"
        onClick={handleTransform}
        disabled={!file || submitting}
      >
        {submitting ? (
          <>
            <Loader2 size={18} className="spin" />
            Queueing...
          </>
        ) : (
          <>
            <Wand2 size={18} />
            Transform Video
          </>
        )}
      </button>

      {/* Result */}
      {result && (
        <div className="result-section">
          <h3>Result</h3>
          <video src={result} controls className="result-video" />
          <a 
            href={result} 
            download 
            className="btn-secondary"
            style={{ marginTop: 12 }}
          >
            Download Video
          </a>
        </div>
      )}

      <style>{`
        .tool-section {
          margin-bottom: 20px;
        }
        .tool-section h3 {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
          font-weight: 500;
          margin-bottom: 12px;
          color: var(--text-color, #fff);
        }
        .tool-section h3 .hint {
          font-weight: 400;
          font-size: 12px;
          color: var(--text-muted, #666);
        }
        .upload-dropzone {
          border: 2px dashed var(--border-color, #444);
          border-radius: 12px;
          padding: 40px;
          text-align: center;
          cursor: pointer;
          transition: all 0.2s;
          min-height: 150px;
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
        }
        .upload-preview {
          max-width: 100%;
          border-radius: 8px;
        }
        .upload-placeholder {
          color: var(--text-muted, #888);
        }
        .upload-placeholder p {
          margin-top: 12px;
          margin-bottom: 4px;
        }
        .video-info {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 16px;
          margin-top: 12px;
          font-size: 13px;
          color: var(--text-muted, #888);
        }
        .style-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 8px;
        }
        .style-btn {
          padding: 12px;
          border: 1px solid var(--border-color, #444);
          border-radius: 8px;
          background: var(--bg-secondary, #1a1a1a);
          cursor: pointer;
          transition: all 0.2s;
          text-align: left;
        }
        .style-btn:hover {
          border-color: var(--accent-color, #7c3aed);
        }
        .style-btn.active {
          background: rgba(124, 58, 237, 0.2);
          border-color: var(--accent-color, #7c3aed);
        }
        .style-name {
          display: block;
          font-size: 13px;
          font-weight: 500;
          color: var(--text-color, #fff);
        }
        .style-desc {
          display: block;
          font-size: 11px;
          color: var(--text-muted, #888);
          margin-top: 4px;
        }
        .prompt-textarea {
          width: 100%;
          padding: 12px;
          border-radius: 8px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          font-size: 13px;
          resize: none;
        }
        .slider-row {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .slider-row input[type="range"] {
          flex: 1;
        }
        .slider-value {
          min-width: 45px;
          text-align: right;
          font-weight: 500;
          color: var(--accent-color, #7c3aed);
        }
        .slider-labels {
          display: flex;
          justify-content: space-between;
          font-size: 11px;
          color: var(--text-muted, #666);
          margin-top: 4px;
        }
        .collapsible h3 {
          padding: 12px;
          margin: -12px -12px 0;
          border-radius: 8px;
        }
        .collapsible h3:hover {
          background: var(--bg-secondary, #1a1a1a);
        }
        .advanced-content {
          margin-top: 12px;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        .form-row {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .form-row label {
          min-width: 120px;
          font-size: 13px;
          color: var(--text-secondary, #aaa);
        }
        .form-row select, .form-row input {
          flex: 1;
          padding: 8px 12px;
          border-radius: 6px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          font-size: 13px;
        }
        .form-row textarea {
          flex: 1;
          padding: 8px 12px;
          border-radius: 6px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          resize: none;
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
        .result-video {
          width: 100%;
          border-radius: 8px;
          margin-top: 12px;
        }
        .spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        @media (max-width: 600px) {
          .style-grid {
            grid-template-columns: repeat(2, 1fr);
          }
        }
      `}</style>
    </div>
  )
}
