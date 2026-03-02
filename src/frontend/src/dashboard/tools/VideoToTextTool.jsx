import React, { useState, useCallback, useRef } from 'react'
import { Upload, Video, FileText, Loader2, Copy, Check, Settings, ChevronDown, Link, Youtube, Download } from 'lucide-react'
import { BACKEND_BASE, DEBUG, getMediaUrl } from '../../config'
import { postForm, postJson } from '../../api'
import CreationsPickerModal from '../../components/CreationsPickerModal'

const CAPTION_MODES = [
  { value: 'brief', label: 'Brief', desc: 'Short 1-2 sentence description' },
  { value: 'detailed', label: 'Detailed', desc: 'Comprehensive scene analysis' },
  { value: 'prompt', label: 'Prompt Style', desc: 'Optimized for AI generation' },
  { value: 'timeline', label: 'Timeline', desc: 'Frame-by-frame breakdown' },
]

const MODELS = [
  { value: 'smolvlm', label: 'SmolVLM', desc: 'Fast, lightweight vision model' },
  { value: 'cogvlm', label: 'CogVLM', desc: 'High quality, slower' },
  { value: 'llava', label: 'LLaVA', desc: 'Balanced quality/speed' },
]

const SOURCE_TABS = [
  { value: 'upload', label: 'Upload', icon: Upload },
  { value: 'youtube', label: 'YouTube', icon: Youtube },
]

export default function VideoToTextTool() {
  const [sourceTab, setSourceTab] = useState('upload')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [videoInfo, setVideoInfo] = useState(null)
  const [showCreationsPicker, setShowCreationsPicker] = useState(false)

  // YouTube state
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [youtubeInfo, setYoutubeInfo] = useState(null)
  const [youtubeLoading, setYoutubeLoading] = useState(false)
  const [downloadedVideoPath, setDownloadedVideoPath] = useState(null)

  const [model, setModel] = useState('smolvlm')
  const [mode, setMode] = useState('detailed')
  const [frameInterval, setFrameInterval] = useState(1) // seconds between sampled frames
  const [maxFrames, setMaxFrames] = useState(8)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [status, setStatus] = useState('')
  const [result, setResult] = useState(null)
  const [copied, setCopied] = useState(false)

  const videoRef = useRef(null)

  const handleFileChange = useCallback((e) => {
    const f = e.target.files?.[0]
    if (f) {
      setFile(f)
      const url = URL.createObjectURL(f)
      setPreview(url)
      setResult(null)
      setError(null)

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
      setDownloadedVideoPath(null)

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

  const handleCreationsSelect = useCallback(async (item) => {
    try {
      const mediaUrl = getMediaUrl(item.url, item.signed_url)
      const response = await fetch(mediaUrl)
      if (!response.ok) throw new Error(`Failed to fetch: ${response.status}`)
      const blob = await response.blob()
      const filename = item.filename || mediaUrl.split('/').pop()
      const fileObj = new File([blob], filename, { type: blob.type || 'video/mp4' })
      const url = URL.createObjectURL(fileObj)
      setFile(fileObj)
      setPreview(url)
      setResult(null)
      setError(null)
      setDownloadedVideoPath(null)
      setSourceTab('upload')
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
      if (DEBUG) console.log('\ud83d\udcc1 V2T: loaded from creations:', filename)
    } catch (e) {
      console.error('Failed to load from creations:', e)
      setError('\u26a0\ufe0f Failed to load video from My Creations')
    }
  }, [])

  // YouTube URL validation and info fetch
  const isValidYoutubeUrl = (url) => {
    return /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+/.test(url)
  }

  const handleYoutubeUrlChange = (e) => {
    const url = e.target.value
    setYoutubeUrl(url)
    setYoutubeInfo(null)
    setError(null)
  }

  const handleFetchYoutubeInfo = async () => {
    if (!youtubeUrl || !isValidYoutubeUrl(youtubeUrl)) {
      setError('Please enter a valid YouTube URL')
      return
    }

    setYoutubeLoading(true)
    setError(null)

    try {
      const res = await postJson(`${BACKEND_BASE}/youtube/info`, { url: youtubeUrl })
      if (!res.ok) {
        throw new Error(res.data?.detail || 'Failed to fetch video info')
      }
      setYoutubeInfo(res.data)
      if (DEBUG) console.debug('🎬 YouTube info:', res.data)
    } catch (err) {
      setError(err.message)
    } finally {
      setYoutubeLoading(false)
    }
  }

  const handleDownloadYoutube = async () => {
    if (!youtubeUrl) return

    setYoutubeLoading(true)
    setError(null)
    setStatus('Downloading video from YouTube...')

    try {
      const res = await postJson(`${BACKEND_BASE}/youtube/download`, {
        url: youtubeUrl,
        format: 'video',  // video | audio
        quality: '720p'
      })

      if (!res.ok) {
        throw new Error(res.data?.detail || 'Failed to download video')
      }

      setDownloadedVideoPath(res.data.path)
      setPreview(`${BACKEND_BASE}/file/${encodeURIComponent(res.data.path)}`)
      setVideoInfo({
        duration: res.data.duration?.toFixed(1) || youtubeInfo?.duration,
        width: res.data.width || youtubeInfo?.width || 1280,
        height: res.data.height || youtubeInfo?.height || 720,
        title: youtubeInfo?.title
      })

      if (DEBUG) console.debug('🎬 YouTube downloaded:', res.data)

    } catch (err) {
      setError(err.message)
    } finally {
      setYoutubeLoading(false)
      setStatus('')
    }
  }

  const handleAnalyze = async () => {
    if (!file && !downloadedVideoPath) return

    setLoading(true)
    setError(null)
    setStatus('Uploading video...')

    try {
      const formData = new FormData()

      // If we have a YouTube downloaded video, pass the path instead of file
      if (downloadedVideoPath) {
        formData.append('video_path', downloadedVideoPath)
      } else {
        formData.append('file', file)
      }

      formData.append('model', model)
      formData.append('mode', mode)
      formData.append('frame_interval', String(frameInterval))
      formData.append('max_frames', String(maxFrames))

      if (DEBUG) console.debug('🎬 V2T request:', { model, mode, frameInterval, maxFrames, downloadedVideoPath })

      setStatus('Analyzing video...')

      const res = await postForm(`${BACKEND_BASE}/caption-video`, formData)

      if (!res.ok) {
        throw new Error(res.data?.detail || 'Video analysis failed')
      }

      setResult(res.data)

    } catch (err) {
      console.error('V2T error:', err)
      setError(err.message)
    } finally {
      setLoading(false)
      setStatus('')
    }
  }

  const handleCopy = async (text) => {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="tool-container">
      <CreationsPickerModal
        show={showCreationsPicker}
        onClose={() => setShowCreationsPicker(false)}
        onSelect={handleCreationsSelect}
        filter="video"
        title="Select Video for Captioning"
      />

      <div className="tool-section">
        <h3>
          <Video size={18} />
          Source Video
        </h3>

        {/* Source Tabs */}
        <div className="source-tabs">
          {SOURCE_TABS.map((tab) => (
            <button
              key={tab.value}
              className={`source-tab ${sourceTab === tab.value ? 'active' : ''}`}
              onClick={() => {
                setSourceTab(tab.value)
                setError(null)
              }}
            >
              <tab.icon size={16} />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Upload Tab */}
        {sourceTab === 'upload' && (
          <div
            className={`upload-dropzone ${preview ? 'has-preview' : ''}`}
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => document.getElementById('v2t-file-input').click()}
          >
            {preview && !downloadedVideoPath ? (
              <video
                ref={videoRef}
                src={preview}
                className="upload-preview"
                controls
                muted
                style={{ maxHeight: '200px' }}
              />
            ) : (
              <div className="upload-placeholder">
                <Upload size={32} />
                <p>Drop video here or click to upload</p>
                <span style={{ fontSize: '12px', opacity: 0.6 }}>MP4, WebM, MOV</span>
              </div>
            )}
            <input
              id="v2t-file-input"
              type="file"
              accept="video/*"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
          </div>
        )}

        <button
          onClick={() => setShowCreationsPicker(true)}
          className="btn-creations-picker"
        >
          {'📁'} From My Creations
        </button>

        {/* YouTube Tab */}
        {sourceTab === 'youtube' && (
          <div className="youtube-section">
            <div className="youtube-input-row">
              <div className="youtube-input-wrapper">
                <Link size={16} className="youtube-input-icon" />
                <input
                  type="text"
                  className="youtube-input"
                  placeholder="Paste YouTube URL here..."
                  value={youtubeUrl}
                  onChange={handleYoutubeUrlChange}
                  onKeyDown={(e) => e.key === 'Enter' && handleFetchYoutubeInfo()}
                />
              </div>
              <button
                className="btn-secondary"
                onClick={handleFetchYoutubeInfo}
                disabled={youtubeLoading || !youtubeUrl}
              >
                {youtubeLoading ? <Loader2 size={16} className="spin" /> : 'Fetch'}
              </button>
            </div>

            {youtubeInfo && (
              <div className="youtube-preview">
                {youtubeInfo.thumbnail && (
                  <img
                    src={youtubeInfo.thumbnail}
                    alt="thumbnail"
                    className="youtube-thumbnail"
                  />
                )}
                <div className="youtube-info">
                  <span className="youtube-title">{youtubeInfo.title}</span>
                  <span className="youtube-meta">
                    {youtubeInfo.channel} • {youtubeInfo.duration}s • {youtubeInfo.view_count?.toLocaleString()} views
                  </span>
                </div>
                <button
                  className="btn-primary"
                  onClick={handleDownloadYoutube}
                  disabled={youtubeLoading}
                >
                  {youtubeLoading ? (
                    <Loader2 size={16} className="spin" />
                  ) : (
                    <>
                      <Download size={16} />
                      Download
                    </>
                  )}
                </button>
              </div>
            )}

            {downloadedVideoPath && (
              <div className="youtube-downloaded">
                <Check size={16} style={{ color: '#22c55e' }} />
                <span>Video ready for analysis</span>
                {preview && (
                  <video
                    src={preview}
                    className="upload-preview"
                    controls
                    muted
                    style={{ maxHeight: '200px', marginTop: '12px', width: '100%' }}
                  />
                )}
              </div>
            )}
          </div>
        )}

        {videoInfo && (
          <div className="video-info">
            <span>📐 {videoInfo.width} × {videoInfo.height}</span>
            <span>⏱️ {videoInfo.duration}s</span>
          </div>
        )}
      </div>

      {/* Model Selection */}
      <div className="tool-section">
        <h3>
          <FileText size={18} />
          Analysis Model
        </h3>
        <div className="model-grid">
          {MODELS.map((m) => (
            <button
              key={m.value}
              className={`model-btn ${model === m.value ? 'active' : ''}`}
              onClick={() => setModel(m.value)}
            >
              <span className="model-name">{m.label}</span>
              <span className="model-desc">{m.desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Caption Mode */}
      <div className="tool-section">
        <h3>Output Style</h3>
        <div className="mode-grid">
          {CAPTION_MODES.map((m) => (
            <button
              key={m.value}
              className={`mode-btn ${mode === m.value ? 'active' : ''}`}
              onClick={() => setMode(m.value)}
            >
              <span className="mode-name">{m.label}</span>
              <span className="mode-desc">{m.desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Advanced Settings */}
      <div className="tool-section collapsible">
        <h3
          onClick={() => setShowAdvanced(!showAdvanced)}
          style={{ cursor: 'pointer' }}
        >
          <Settings size={16} />
          Advanced
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
              <label>Frame Interval</label>
              <select value={frameInterval} onChange={(e) => setFrameInterval(parseFloat(e.target.value))}>
                <option value={0.5}>Every 0.5s</option>
                <option value={1}>Every 1s</option>
                <option value={2}>Every 2s</option>
                <option value={5}>Every 5s</option>
              </select>
            </div>
            <div className="form-row">
              <label>Max Frames</label>
              <select value={maxFrames} onChange={(e) => setMaxFrames(parseInt(e.target.value))}>
                <option value={4}>4 frames</option>
                <option value={8}>8 frames</option>
                <option value={16}>16 frames</option>
                <option value={32}>32 frames</option>
              </select>
            </div>
          </div>
        )}
      </div>

      {error && <div className="error-message">⚠️ {error}</div>}

      <button
        className="btn-primary btn-large"
        onClick={handleAnalyze}
        disabled={(!file && !downloadedVideoPath) || loading}
      >
        {loading ? (
          <>
            <Loader2 size={18} className="spin" />
            {status}
          </>
        ) : (
          <>
            <FileText size={18} />
            Analyze Video
          </>
        )}
      </button>

      {/* Result */}
      {result && (
        <div className="result-section">
          <div className="result-header">
            <h3>Description</h3>
            <button
              className="copy-btn"
              onClick={() => handleCopy(result.caption || result.description)}
            >
              {copied ? <Check size={16} /> : <Copy size={16} />}
              {copied ? 'Copied!' : 'Copy'}
            </button>
          </div>

          <div className="result-text">
            {result.caption || result.description}
          </div>

          {result.timeline && result.timeline.length > 0 && (
            <div className="timeline-section">
              <h4>Timeline</h4>
              {result.timeline.map((item, idx) => (
                <div key={idx} className="timeline-item">
                  <span className="timeline-time">{item.time}s</span>
                  <span className="timeline-desc">{item.description}</span>
                </div>
              ))}
            </div>
          )}

          {result.prompt && (
            <div className="prompt-section">
              <div className="prompt-header">
                <h4>AI Generation Prompt</h4>
                <button
                  className="copy-btn small"
                  onClick={() => handleCopy(result.prompt)}
                >
                  <Copy size={14} />
                </button>
              </div>
              <div className="prompt-text">{result.prompt}</div>
            </div>
          )}
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
        .source-tabs {
          display: flex;
          gap: 8px;
          margin-bottom: 12px;
        }
        .source-tab {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 8px 16px;
          border: 1px solid var(--border-color, #444);
          border-radius: 8px;
          background: transparent;
          color: var(--text-muted, #888);
          cursor: pointer;
          transition: all 0.2s;
          font-size: 13px;
        }
        .source-tab:hover {
          border-color: var(--accent-color, #7c3aed);
          color: var(--text-color, #fff);
        }
        .source-tab.active {
          background: rgba(124, 58, 237, 0.2);
          border-color: var(--accent-color, #7c3aed);
          color: var(--text-color, #fff);
        }
        .youtube-section {
          padding: 16px;
          border: 1px solid var(--border-color, #444);
          border-radius: 12px;
          background: var(--bg-secondary, #1a1a1a);
        }
        .youtube-input-row {
          display: flex;
          gap: 8px;
        }
        .youtube-input-wrapper {
          flex: 1;
          position: relative;
        }
        .youtube-input-icon {
          position: absolute;
          left: 12px;
          top: 50%;
          transform: translateY(-50%);
          color: var(--text-muted, #888);
        }
        .youtube-input {
          width: 100%;
          padding: 10px 12px 10px 36px;
          border: 1px solid var(--border-color, #444);
          border-radius: 8px;
          background: var(--bg-primary, #0a0a0a);
          color: var(--text-color, #fff);
          font-size: 13px;
        }
        .youtube-input:focus {
          outline: none;
          border-color: var(--accent-color, #7c3aed);
        }
        .youtube-preview {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-top: 12px;
          padding: 12px;
          background: var(--bg-primary, #0a0a0a);
          border-radius: 8px;
        }
        .youtube-thumbnail {
          width: 120px;
          height: 68px;
          object-fit: cover;
          border-radius: 6px;
        }
        .youtube-info {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 4px;
        }
        .youtube-title {
          font-size: 13px;
          font-weight: 500;
          color: var(--text-color, #fff);
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
        .youtube-meta {
          font-size: 11px;
          color: var(--text-muted, #888);
        }
        .youtube-downloaded {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-top: 12px;
          padding: 12px;
          background: rgba(34, 197, 94, 0.1);
          border: 1px solid rgba(34, 197, 94, 0.3);
          border-radius: 8px;
          color: #22c55e;
          font-size: 13px;
          flex-wrap: wrap;
        }
        .upload-dropzone {
          border: 2px dashed var(--border-color, #444);
          border-radius: 12px;
          padding: 40px;
          text-align: center;
          cursor: pointer;
          transition: all 0.2s;
          min-height: 120px;
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
        .video-info {
          display: flex;
          gap: 16px;
          justify-content: center;
          margin-top: 8px;
          font-size: 12px;
          color: var(--text-muted, #888);
        }
        .model-grid, .mode-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 8px;
        }
        .mode-grid {
          grid-template-columns: repeat(2, 1fr);
        }
        .model-btn, .mode-btn {
          padding: 12px;
          border: 1px solid var(--border-color, #444);
          border-radius: 8px;
          background: var(--bg-secondary, #1a1a1a);
          cursor: pointer;
          transition: all 0.2s;
          text-align: left;
        }
        .model-btn:hover, .mode-btn:hover {
          border-color: var(--accent-color, #7c3aed);
        }
        .model-btn.active, .mode-btn.active {
          background: rgba(124, 58, 237, 0.2);
          border-color: var(--accent-color, #7c3aed);
        }
        .model-name, .mode-name {
          display: block;
          font-size: 13px;
          font-weight: 500;
          color: var(--text-color, #fff);
        }
        .model-desc, .mode-desc {
          display: block;
          font-size: 11px;
          color: var(--text-muted, #888);
          margin-top: 2px;
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
          min-width: 100px;
          font-size: 13px;
          color: var(--text-secondary, #aaa);
        }
        .form-row select {
          flex: 1;
          padding: 8px;
          border-radius: 6px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
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
        .result-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
        }
        .result-header h3 {
          margin: 0;
        }
        .copy-btn {
          display: flex;
          align-items: center;
          gap: 4px;
          padding: 6px 12px;
          border: 1px solid var(--border-color, #444);
          border-radius: 6px;
          background: transparent;
          color: var(--text-color, #fff);
          cursor: pointer;
          font-size: 12px;
        }
        .copy-btn:hover {
          background: var(--bg-secondary, #1a1a1a);
        }
        .copy-btn.small {
          padding: 4px 8px;
        }
        .result-text {
          padding: 16px;
          background: var(--bg-secondary, #1a1a1a);
          border-radius: 8px;
          font-size: 14px;
          line-height: 1.6;
          white-space: pre-wrap;
        }
        .timeline-section {
          margin-top: 16px;
        }
        .timeline-section h4 {
          font-size: 13px;
          margin-bottom: 8px;
          color: var(--text-secondary, #aaa);
        }
        .timeline-item {
          display: flex;
          gap: 12px;
          padding: 8px 0;
          border-bottom: 1px solid var(--border-color, #333);
        }
        .timeline-time {
          min-width: 50px;
          font-weight: 500;
          color: var(--accent-color, #7c3aed);
        }
        .timeline-desc {
          color: var(--text-color, #fff);
          font-size: 13px;
        }
        .prompt-section {
          margin-top: 16px;
          padding: 12px;
          background: rgba(124, 58, 237, 0.1);
          border: 1px solid rgba(124, 58, 237, 0.3);
          border-radius: 8px;
        }
        .prompt-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
        }
        .prompt-header h4 {
          margin: 0;
          font-size: 12px;
          color: var(--accent-color, #7c3aed);
        }
        .prompt-text {
          font-size: 13px;
          color: var(--text-color, #fff);
        }
        .spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}
