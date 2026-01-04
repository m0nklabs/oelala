import React, { useState, useRef, useCallback } from 'react'
import { 
  Video, Upload, Play, Pause, Download, Loader2, X, 
  FileAudio, FileVideo, Volume2, Trash2, Sliders
} from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm, postJson } from '../../api'

const SUPPORTED_VIDEO_FORMATS = ['video/mp4', 'video/webm', 'video/quicktime']
const SUPPORTED_AUDIO_FORMATS = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/flac', 'audio/ogg', 'audio/webm']

export default function LipSyncTool({ onOutput, onJobSubmitted }) {
  // Video state
  const [videoFile, setVideoFile] = useState(null)
  const [videoUrl, setVideoUrl] = useState(null)
  const [uploadedVideoPath, setUploadedVideoPath] = useState(null)
  
  // Audio state
  const [audioFile, setAudioFile] = useState(null)
  const [audioUrl, setAudioUrl] = useState(null)
  const [uploadedAudioPath, setUploadedAudioPath] = useState(null)
  
  // Settings
  const [lipsExpression, setLipsExpression] = useState(1.5)
  const [inferenceSteps, setInferenceSteps] = useState(20)
  const [seed, setSeed] = useState(-1)
  
  // Playback refs
  const videoRef = useRef(null)
  const audioRef = useRef(null)
  const resultVideoRef = useRef(null)
  
  // Generation state
  const [submitting, setSubmitting] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const [lastQueued, setLastQueued] = useState(null)
  const [result, setResult] = useState(null)

  // Handle video drop/select
  const handleVideoDrop = useCallback((e) => {
    e.preventDefault()
    const file = e.dataTransfer?.files?.[0] || e.target?.files?.[0]
    if (file && SUPPORTED_VIDEO_FORMATS.some(fmt => file.type.includes(fmt.split('/')[1]))) {
      setVideoFile(file)
      setVideoUrl(URL.createObjectURL(file))
      setUploadedVideoPath(null)
      setError(null)
      setLastQueued(null)
    } else if (file) {
      setError('Please upload a valid video file (MP4, WebM)')
    }
  }, [])

  // Handle audio drop/select
  const handleAudioDrop = useCallback((e) => {
    e.preventDefault()
    const file = e.dataTransfer?.files?.[0] || e.target?.files?.[0]
    if (file && SUPPORTED_AUDIO_FORMATS.some(fmt => file.type.includes(fmt.split('/')[1]))) {
      setAudioFile(file)
      setAudioUrl(URL.createObjectURL(file))
      setUploadedAudioPath(null)
      setError(null)
      setLastQueued(null)
    } else if (file) {
      setError('Please upload a valid audio file (WAV, MP3, FLAC)')
    }
  }, [])

  // Upload file
  const uploadFile = async (file) => {
    const formData = new FormData()
    formData.append('file', file)
    
    try {
      const res = await postForm(`${BACKEND_BASE}/upload`, formData)
      if (res.ok && res.data?.path) {
        return res.data.path
      }
      throw new Error(res.data?.detail || 'Upload failed')
    } catch (err) {
      throw new Error('Failed to upload file: ' + err.message)
    }
  }

  // Generate lip synced video
  const handleGenerate = async () => {
    if (!videoFile || !audioFile) {
      setError('Please provide both a video and audio file')
      return
    }
    
    setSubmitting(true)
    setUploading(true)
    setError(null)
    setLastQueued(null)
    setResult(null)
    
    try {
      // Upload video if needed
      let videoPath = uploadedVideoPath
      if (!videoPath) {
        videoPath = await uploadFile(videoFile)
        setUploadedVideoPath(videoPath)
      }
      
      // Upload audio if needed
      let audioPath = uploadedAudioPath
      if (!audioPath) {
        audioPath = await uploadFile(audioFile)
        setUploadedAudioPath(audioPath)
      }
      
      setUploading(false)
      
      // Request lip sync
      const res = await postJson(`${BACKEND_BASE}/lip-sync`, {
        video_path: videoPath,
        audio_path: audioPath,
        lips_expression: lipsExpression,
        inference_steps: inferenceSteps,
        seed: seed === -1 ? Math.floor(Math.random() * 2147483647) : seed,
      })
      
      if (!res.ok) {
        throw new Error(res.data?.detail || 'Lip sync request failed')
      }
      
      if (res.data?.prompt_id) {
        // Show queued confirmation
        setLastQueued({ promptId: res.data.prompt_id })
        
        // Notify queue indicator
        if (onJobSubmitted) onJobSubmitted({ prompt_id: res.data.prompt_id })
        
        if (DEBUG) console.debug('📋 LipSync queued:', res.data.prompt_id)
        
        // Don't wait for completion - job will appear in queue/history when done
      }
      
    } catch (err) {
      console.error('Lip sync error:', err)
      setError(err.message)
    } finally {
      setSubmitting(false)
      setUploading(false)
    }
  }

  // Clear files
  const clearVideo = () => {
    setVideoFile(null)
    setVideoUrl(null)
    setUploadedVideoPath(null)
  }

  const clearAudio = () => {
    setAudioFile(null)
    setAudioUrl(null)
    setUploadedAudioPath(null)
  }

  return (
    <div className="tool-container">
      {/* Video Input */}
      <div className="tool-section">
        <h3>
          <FileVideo size={18} />
          Input Video (with face)
        </h3>
        
        {!videoFile ? (
          <div
            className="drop-zone"
            onDrop={handleVideoDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => document.getElementById('video-file-input').click()}
          >
            <Upload size={32} />
            <p>Drop video file here or click to browse</p>
            <span className="supported-formats">MP4, WebM</span>
            <input
              id="video-file-input"
              type="file"
              accept="video/*"
              onChange={handleVideoDrop}
              style={{ display: 'none' }}
            />
          </div>
        ) : (
          <div className="media-preview">
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              className="preview-video"
            />
            <div className="file-info-row">
              <span className="filename">{videoFile.name}</span>
              <button className="icon-btn danger" onClick={clearVideo}>
                <Trash2 size={18} />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Audio Input */}
      <div className="tool-section">
        <h3>
          <FileAudio size={18} />
          Audio Track (speech/dialogue)
        </h3>
        
        {!audioFile ? (
          <div
            className="drop-zone"
            onDrop={handleAudioDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => document.getElementById('audio-file-input').click()}
          >
            <Upload size={32} />
            <p>Drop audio file here or click to browse</p>
            <span className="supported-formats">WAV, MP3, FLAC, OGG</span>
            <input
              id="audio-file-input"
              type="file"
              accept="audio/*"
              onChange={handleAudioDrop}
              style={{ display: 'none' }}
            />
          </div>
        ) : (
          <div className="audio-preview">
            <audio ref={audioRef} src={audioUrl} controls className="preview-audio" />
            <div className="file-info-row">
              <span className="filename">{audioFile.name}</span>
              <button className="icon-btn danger" onClick={clearAudio}>
                <Trash2 size={18} />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Settings */}
      <div className="tool-section">
        <h3>
          <Sliders size={18} />
          Settings
        </h3>
        
        <div className="setting-row">
          <label>Lips Expression</label>
          <div className="slider-row">
            <input
              type="range"
              min={1.0}
              max={3.0}
              step={0.1}
              value={lipsExpression}
              onChange={(e) => setLipsExpression(parseFloat(e.target.value))}
            />
            <span className="slider-value">{lipsExpression.toFixed(1)}</span>
          </div>
          <span className="setting-hint">Higher = more exaggerated lip movements</span>
        </div>

        <div className="setting-row">
          <label>Inference Steps</label>
          <div className="slider-row">
            <input
              type="range"
              min={10}
              max={50}
              step={5}
              value={inferenceSteps}
              onChange={(e) => setInferenceSteps(parseInt(e.target.value))}
            />
            <span className="slider-value">{inferenceSteps}</span>
          </div>
          <span className="setting-hint">More steps = better quality, slower</span>
        </div>

        <div className="setting-row">
          <label>Seed</label>
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(parseInt(e.target.value) || -1)}
            placeholder="-1 for random"
            className="seed-input"
          />
        </div>
      </div>

      {/* Queued notification */}
      {lastQueued && (
        <div className="queued-notice">
          ✅ Job queued! Check the Queue panel for progress.
        </div>
      )}

      {/* Generate Button */}
      <div className="tool-section">
        <button
          className="generate-btn"
          onClick={handleGenerate}
          disabled={submitting || !videoFile || !audioFile}
        >
          {submitting ? (
            <>
              <Loader2 size={20} className="spin" />
              <span>{uploading ? 'Uploading...' : 'Queueing...'}</span>
            </>
          ) : (
            <>
              <Video size={20} />
              <span>Sync Lips</span>
            </>
          )}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="error-message">
          <X size={16} />
          {error}
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="tool-section result-section">
          <h3>
            <Video size={18} />
            Lip Synced Result
          </h3>
          <div className="video-result">
            <video
              ref={resultVideoRef}
              src={result.url}
              controls
              className="result-video"
            />
            <div className="result-actions">
              <span className="filename">{result.filename}</span>
              <a
                href={result.url}
                download={result.filename}
                className="download-btn"
              >
                <Download size={18} />
                Download
              </a>
            </div>
          </div>
        </div>
      )}

      <style>{`
        .drop-zone {
          border: 2px dashed #4a4a4a;
          border-radius: 12px;
          padding: 32px;
          text-align: center;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .drop-zone:hover {
          border-color: #fbbf24;
          background: rgba(251, 191, 36, 0.05);
        }
        
        .drop-zone p {
          margin: 12px 0 4px;
          color: #ccc;
        }
        
        .supported-formats {
          font-size: 12px;
          color: #888;
        }
        
        .media-preview, .audio-preview {
          background: #1a1a1a;
          border-radius: 12px;
          padding: 16px;
        }
        
        .preview-video, .result-video {
          width: 100%;
          max-height: 300px;
          border-radius: 8px;
          background: #000;
        }
        
        .preview-audio {
          width: 100%;
        }
        
        .file-info-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-top: 12px;
        }
        
        .filename {
          color: #ccc;
          font-size: 13px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        
        .icon-btn {
          padding: 8px;
          border-radius: 8px;
          background: #2a2a2a;
          border: none;
          color: #fff;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .icon-btn:hover {
          background: #3a3a3a;
        }
        
        .icon-btn.danger:hover {
          background: rgba(239, 68, 68, 0.2);
          color: #ef4444;
        }
        
        .setting-row {
          margin-bottom: 16px;
        }
        
        .setting-row label {
          display: block;
          margin-bottom: 8px;
          color: #ccc;
          font-size: 13px;
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
          min-width: 50px;
          text-align: right;
          color: #fbbf24;
          font-weight: 500;
        }
        
        .setting-hint {
          display: block;
          font-size: 11px;
          color: #666;
          margin-top: 4px;
        }
        
        .seed-input {
          width: 100%;
          padding: 10px 12px;
          border-radius: 8px;
          background: #1a1a1a;
          border: 1px solid #2a2a2a;
          color: #fff;
          font-size: 14px;
        }
        
        .result-section {
          background: rgba(34, 197, 94, 0.1);
          border: 1px solid rgba(34, 197, 94, 0.3);
          border-radius: 12px;
          padding: 16px;
        }
        
        .video-result {
          margin-top: 12px;
        }
        
        .result-actions {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-top: 12px;
        }
        
        .download-btn {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 10px 16px;
          border-radius: 8px;
          background: #fbbf24;
          color: #000;
          text-decoration: none;
          font-weight: 500;
          transition: all 0.2s;
        }
        
        .download-btn:hover {
          background: #f59e0b;
        }
        
        .error-message {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 12px 16px;
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 8px;
          color: #ef4444;
          font-size: 13px;
        }
        
        .progress-bar {
          height: 4px;
          background: #2a2a2a;
          border-radius: 2px;
          margin-top: 12px;
          overflow: hidden;
        }
        
        .progress-fill {
          height: 100%;
          background: linear-gradient(90deg, #fbbf24, #f59e0b);
          transition: width 0.3s;
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
