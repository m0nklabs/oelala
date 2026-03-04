import React, { useState, useRef, useCallback, useMemo, useEffect } from 'react'
import {
  Mic, Upload, Play, Pause, Download, Loader2, X,
  FileAudio, Volume2, Trash2, Check
} from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm, postJson } from '../../api'
import { useAuth } from '../../contexts/AuthContext'
import { useToolSettings } from '../../hooks/useToolSettings'
import ResetDefaultsButton from '../../components/ResetDefaultsButton'

const SUPPORTED_AUDIO_FORMATS = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/flac', 'audio/ogg', 'audio/webm']

const F5_MODELS = [
  { value: 'F5v1', label: 'F5 v1 (English)', desc: 'Best quality English' },
  { value: 'F5', label: 'F5 Base (English)', desc: 'Standard English model' },
  { value: 'F5-DE', label: 'F5 German', desc: 'German language' },
  { value: 'F5-FR', label: 'F5 French', desc: 'French language' },
  { value: 'F5-ES', label: 'F5 Spanish', desc: 'Spanish language' },
  { value: 'F5-IT', label: 'F5 Italian', desc: 'Italian language' },
  { value: 'F5-JP', label: 'F5 Japanese', desc: 'Japanese language' },
  { value: 'E2', label: 'E2-TTS', desc: 'Alternative English model' },
]

const VC_DEFAULTS = { text: '', model: 'F5v1', speed: 1.0 }

export default function VoiceCloningTool({ onOutput, onJobSubmitted }) {
  const { user, requestLogin } = useAuth()
  const { initial, save: saveSettings, resetDefaults } = useToolSettings('voice_cloning', VC_DEFAULTS)

  // Voice sample state
  const [voiceSample, setVoiceSample] = useState(null)
  const [voiceSampleUrl, setVoiceSampleUrl] = useState(null)
  const [uploadedPath, setUploadedPath] = useState(null)

  // Text input
  const [text, setText] = useState(initial.text)

  // Model settings
  const [model, setModel] = useState(initial.model)
  const [speed, setSpeed] = useState(initial.speed)

  // Auto-save settings
  const settingsSnapshot = useMemo(() => ({ text, model, speed }), [text, model, speed])
  useEffect(() => { saveSettings(settingsSnapshot) }, [settingsSnapshot, saveSettings])

  const handleResetDefaults = useCallback(() => {
    const d = resetDefaults()
    setText(d.text); setModel(d.model); setSpeed(d.speed)
  }, [resetDefaults])

  // Recording state
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])
  const timerRef = useRef(null)

  // Playback
  const audioRef = useRef(null)
  const resultAudioRef = useRef(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isResultPlaying, setIsResultPlaying] = useState(false)

  // Generation state
  const [submitting, setSubmitting] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const [lastQueued, setLastQueued] = useState(null)
  const [result, setResult] = useState(null)

  // Handle file drop/select
  const handleDrop = useCallback((e) => {
    e.preventDefault()
    const file = e.dataTransfer?.files?.[0] || e.target?.files?.[0]
    if (file && SUPPORTED_AUDIO_FORMATS.some(fmt => file.type.includes(fmt.split('/')[1]))) {
      setVoiceSample(file)
      setVoiceSampleUrl(URL.createObjectURL(file))
      setUploadedPath(null) // Will upload when generating
      setError(null)
    } else if (file) {
      setError('Please upload a valid audio file (WAV, MP3, FLAC, OGG)')
    }
  }, [])

  // Start recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      })

      chunksRef.current = []
      mediaRecorderRef.current = mediaRecorder

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)
        }
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        const file = new File([blob], 'recording.webm', { type: 'audio/webm' })
        setVoiceSample(file)
        setVoiceSampleUrl(URL.createObjectURL(blob))
        setUploadedPath(null)
        stream.getTracks().forEach(track => track.stop())
      }

      mediaRecorder.start()
      setIsRecording(true)
      setRecordingTime(0)

      timerRef.current = setInterval(() => {
        setRecordingTime(t => t + 1)
      }, 1000)

    } catch (err) {
      setError('Failed to access microphone: ' + err.message)
    }
  }

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      clearInterval(timerRef.current)
    }
  }

  // Upload voice sample
  const uploadVoiceSample = async () => {
    if (!voiceSample) return null

    const formData = new FormData()
    formData.append('file', voiceSample)

    try {
      const res = await postForm(`${BACKEND_BASE}/upload`, formData)
      if (res.ok && res.data?.path) {
        setUploadedPath(res.data.path)
        return res.data.path
      }
      throw new Error(res.data?.detail || 'Upload failed')
    } catch (err) {
      throw new Error('Failed to upload voice sample: ' + err.message)
    }
  }

  // Generate cloned voice
  const handleGenerate = async () => {
    // Check if user is logged in
    if (!user) {
      requestLogin('Log in om te genereren')
      return
    }

    if (!voiceSample || !text.trim()) {
      setError('Please provide both a voice sample and text to speak')
      return
    }

    setSubmitting(true)
    setUploading(true)
    setError(null)
    setLastQueued(null)
    setResult(null)

    try {
      // Upload voice sample if needed
      let audioPath = uploadedPath
      if (!audioPath) {
        audioPath = await uploadVoiceSample()
      }

      setUploading(false)

      // Request voice cloning via F5-TTS
      const res = await postJson(`${BACKEND_BASE}/voice-clone`, {
        voice_sample_path: audioPath,
        text: text.trim(),
        model: model,
        speed: speed,
      })

      if (!res.ok) {
        throw new Error(res.data?.detail || 'Voice cloning request failed')
      }

      if (res.data?.prompt_id) {
        // Show queued confirmation
        setLastQueued({
          promptId: res.data.prompt_id,
          model: F5_MODELS.find(m => m.value === model)?.label || model
        })

        // Notify queue indicator
        if (onJobSubmitted) onJobSubmitted({ prompt_id: res.data.prompt_id })

        if (DEBUG) console.debug('📋 Voice cloning queued:', res.data.prompt_id)

        // Don't wait for completion - job will appear in queue/history when done
      }

    } catch (err) {
      console.error('Voice cloning error:', err)
      setError(err.message)
    } finally {
      setSubmitting(false)
      setUploading(false)
    }
  }

  // Clear voice sample
  const clearVoiceSample = () => {
    setVoiceSample(null)
    setVoiceSampleUrl(null)
    setUploadedPath(null)
    setLastQueued(null)
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
    }
    setIsPlaying(false)
  }

  // Toggle playback
  const togglePlay = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause()
      } else {
        audioRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const toggleResultPlay = () => {
    if (resultAudioRef.current) {
      if (isResultPlaying) {
        resultAudioRef.current.pause()
      } else {
        resultAudioRef.current.play()
      }
      setIsResultPlaying(!isResultPlaying)
    }
  }

  // Format time
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="tool-container">
      {/* Voice Sample Section */}
      <div className="tool-section">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h3 style={{ margin: 0 }}>
            <FileAudio size={18} />
            Voice Sample (5-30 seconds recommended)
          </h3>
          <ResetDefaultsButton onReset={handleResetDefaults} />
        </div>

        {!voiceSample ? (
          <div className="voice-input-options">
            {/* Upload Area */}
            <div
              className="drop-zone"
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => document.getElementById('voice-file-input').click()}
            >
              <Upload size={32} />
              <p>Drop audio file here or click to browse</p>
              <span className="supported-formats">WAV, MP3, FLAC, OGG</span>
              <input
                id="voice-file-input"
                type="file"
                accept="audio/*"
                onChange={handleDrop}
                style={{ display: 'none' }}
              />
            </div>

            <div className="divider-text">or</div>

            {/* Record Button */}
            <button
              className={`record-btn ${isRecording ? 'recording' : ''}`}
              onClick={isRecording ? stopRecording : startRecording}
            >
              {isRecording ? (
                <>
                  <div className="recording-indicator" />
                  <span>Stop Recording ({formatTime(recordingTime)})</span>
                </>
              ) : (
                <>
                  <Mic size={20} />
                  <span>Record Voice Sample</span>
                </>
              )}
            </button>
          </div>
        ) : (
          <div className="voice-preview">
            <div className="voice-file-info">
              <FileAudio size={24} />
              <div className="file-details">
                <span className="filename">{voiceSample.name}</span>
                <span className="filesize">{(voiceSample.size / 1024).toFixed(1)} KB</span>
              </div>
              <div className="voice-controls">
                <button className="icon-btn" onClick={togglePlay}>
                  {isPlaying ? <Pause size={18} /> : <Play size={18} />}
                </button>
                <button className="icon-btn danger" onClick={clearVoiceSample}>
                  <Trash2 size={18} />
                </button>
              </div>
            </div>
            <audio
              ref={audioRef}
              src={voiceSampleUrl}
              onEnded={() => setIsPlaying(false)}
            />
            {uploadedPath && (
              <div className="upload-status">
                <Check size={14} /> Uploaded
              </div>
            )}
          </div>
        )}
      </div>

      {/* Text Input */}
      <div className="tool-section">
        <h3>Text to Speak</h3>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter the text you want the cloned voice to speak..."
          rows={4}
          className="prompt-textarea"
        />
        <div className="char-count">{text.length} characters</div>
      </div>

      {/* Model Selection */}
      <div className="tool-section">
        <h3>Model</h3>
        <div className="model-grid">
          {F5_MODELS.map((m) => (
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

      {/* Speed Control */}
      <div className="tool-section">
        <h3>Speed</h3>
        <div className="slider-row">
          <input
            type="range"
            min={0.5}
            max={2.0}
            step={0.1}
            value={speed}
            onChange={(e) => setSpeed(parseFloat(e.target.value))}
          />
          <span className="slider-value">{speed.toFixed(1)}x</span>
        </div>
        <div className="slider-hints">
          <span>&gt;1.0 = slower</span>
          <span>&lt;1.0 = faster</span>
        </div>
      </div>

      {/* Queued notification */}
      {lastQueued && (
        <div className="queued-notice">
          ✅ Job queued! Check the Queue panel for progress.
          <span className="queued-mode">{lastQueued.model}</span>
        </div>
      )}

      {/* Generate Button */}
      <div className="tool-section">
        <button
          className="generate-btn"
          onClick={handleGenerate}
          disabled={submitting || !voiceSample || !text.trim()}
        >
          {submitting ? (
            <>
              <Loader2 size={20} className="spin" />
              <span>{uploading ? 'Uploading...' : 'Queueing...'}</span>
            </>
          ) : (
            <>
              <Volume2 size={20} />
              <span>Clone Voice</span>
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
            <Volume2 size={18} />
            Cloned Voice Result
          </h3>
          <div className="audio-result">
            <audio
              ref={resultAudioRef}
              src={result.url}
              onEnded={() => setIsResultPlaying(false)}
            />
            <div className="audio-controls">
              <button className="play-btn" onClick={toggleResultPlay}>
                {isResultPlaying ? <Pause size={24} /> : <Play size={24} />}
              </button>
              <span className="filename">{result.filename}</span>
              <a
                href={result.url}
                download={result.filename}
                className="download-btn"
              >
                <Download size={18} />
              </a>
            </div>
          </div>
        </div>
      )}

      <style>{`
        .voice-input-options {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

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

        .divider-text {
          text-align: center;
          color: #666;
          font-size: 13px;
        }

        .record-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          padding: 16px;
          border-radius: 12px;
          background: #2a2a2a;
          border: 2px solid #3a3a3a;
          color: #fff;
          font-size: 14px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .record-btn:hover {
          border-color: #ef4444;
          background: rgba(239, 68, 68, 0.1);
        }

        .record-btn.recording {
          border-color: #ef4444;
          background: rgba(239, 68, 68, 0.2);
        }

        .recording-indicator {
          width: 12px;
          height: 12px;
          border-radius: 50%;
          background: #ef4444;
          animation: pulse 1s infinite;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        .voice-preview {
          background: #1a1a1a;
          border-radius: 12px;
          padding: 16px;
        }

        .voice-file-info {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .file-details {
          flex: 1;
          display: flex;
          flex-direction: column;
        }

        .filename {
          color: #fff;
          font-size: 14px;
        }

        .filesize {
          color: #888;
          font-size: 12px;
        }

        .voice-controls {
          display: flex;
          gap: 8px;
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

        .upload-status {
          display: flex;
          align-items: center;
          gap: 6px;
          margin-top: 8px;
          color: #22c55e;
          font-size: 12px;
        }

        .char-count {
          text-align: right;
          font-size: 12px;
          color: #666;
          margin-top: 4px;
        }

        .model-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
          gap: 8px;
        }

        .model-btn {
          display: flex;
          flex-direction: column;
          align-items: flex-start;
          padding: 12px;
          border-radius: 8px;
          background: #1a1a1a;
          border: 2px solid #2a2a2a;
          color: #fff;
          cursor: pointer;
          transition: all 0.2s;
        }

        .model-btn:hover {
          border-color: #4a4a4a;
        }

        .model-btn.active {
          border-color: #fbbf24;
          background: rgba(251, 191, 36, 0.1);
        }

        .model-name {
          font-size: 13px;
          font-weight: 500;
        }

        .model-desc {
          font-size: 11px;
          color: #888;
          margin-top: 2px;
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

        .slider-hints {
          display: flex;
          justify-content: space-between;
          font-size: 11px;
          color: #666;
          margin-top: 4px;
        }

        .result-section {
          background: rgba(34, 197, 94, 0.1);
          border: 1px solid rgba(34, 197, 94, 0.3);
          border-radius: 12px;
          padding: 16px;
        }

        .audio-result {
          margin-top: 12px;
        }

        .audio-controls {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .play-btn {
          width: 48px;
          height: 48px;
          border-radius: 50%;
          background: #fbbf24;
          border: none;
          color: #000;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.2s;
        }

        .play-btn:hover {
          background: #f59e0b;
          transform: scale(1.05);
        }

        .download-btn {
          margin-left: auto;
          padding: 8px 16px;
          border-radius: 8px;
          background: #2a2a2a;
          color: #fff;
          text-decoration: none;
          display: flex;
          align-items: center;
          gap: 6px;
          transition: all 0.2s;
        }

        .download-btn:hover {
          background: #3a3a3a;
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
