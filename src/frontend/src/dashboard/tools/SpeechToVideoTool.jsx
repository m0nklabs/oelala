import React, { useState, useRef, useCallback } from 'react'
import {
  Video, Upload, Play, Pause, Download, Loader2, X,
  MessageSquare, Volume2, Mic, Settings2, ChevronDown
} from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm, postJson } from '../../api'
import { useAuth } from '../../contexts/AuthContext'

const SUPPORTED_VIDEO_FORMATS = ['video/mp4', 'video/webm', 'video/quicktime']

// TTS Models
const TTS_MODELS = [
  { id: 'f5v1', label: 'F5-TTS v1', description: 'Fast, high quality' },
  { id: 'e2', label: 'E2-TTS', description: 'More expressive' },
]

// Voice presets (from VoiceCloningTool)
const VOICE_PRESETS = [
  { id: 'custom', label: 'Upload Voice Sample', isCustom: true },
  { id: 'alloy', label: 'Alloy (Neutral)' },
  { id: 'echo', label: 'Echo (Male)' },
  { id: 'fable', label: 'Fable (British)' },
  { id: 'onyx', label: 'Onyx (Deep Male)' },
  { id: 'nova', label: 'Nova (Female)' },
  { id: 'shimmer', label: 'Shimmer (Soft Female)' },
]

export default function SpeechToVideoTool({ onOutput, onJobSubmitted }) {
  const { user, requestLogin } = useAuth()

  // Video state
  const [videoFile, setVideoFile] = useState(null)
  const [videoUrl, setVideoUrl] = useState(null)
  const [uploadedVideoPath, setUploadedVideoPath] = useState(null)

  // TTS state
  const [text, setText] = useState('')
  const [ttsModel, setTtsModel] = useState('f5v1')
  const [voicePreset, setVoicePreset] = useState('nova')
  const [voiceSampleFile, setVoiceSampleFile] = useState(null)
  const [voiceSampleUrl, setVoiceSampleUrl] = useState(null)

  // Lip sync settings
  const [lipsExpression, setLipsExpression] = useState(1.5)
  const [inferenceSteps, setInferenceSteps] = useState(20)

  // UI state
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [currentStep, setCurrentStep] = useState(null)
  const [error, setError] = useState(null)
  const [lastQueued, setLastQueued] = useState(null)

  // Refs
  const videoRef = useRef(null)
  const videoInputRef = useRef(null)
  const voiceInputRef = useRef(null)

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

  // Handle voice sample
  const handleVoiceSampleDrop = useCallback((e) => {
    e.preventDefault()
    const file = e.dataTransfer?.files?.[0] || e.target?.files?.[0]
    if (file && file.type.startsWith('audio/')) {
      setVoiceSampleFile(file)
      setVoiceSampleUrl(URL.createObjectURL(file))
      setError(null)
    } else if (file) {
      setError('Please upload a valid audio file for voice sample')
    }
  }, [])

  // Upload file helper
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
      throw new Error(`Upload failed: ${err.message}`)
    }
  }

  // Main generation handler - this is a 2-step process:
  // 1. Generate TTS audio
  // 2. Apply lip sync to video with generated audio
  const handleGenerate = async () => {
    // Check if user is logged in
    if (!user) {
      requestLogin('Log in om te genereren')
      return
    }

    if (!videoFile || !text.trim()) {
      setError('Please upload a video and enter text')
      return
    }

    setSubmitting(true)
    setError(null)
    setLastQueued(null)

    try {
      // Step 1: Upload video if needed
      setCurrentStep('Uploading video...')
      setUploading(true)

      let videoPath = uploadedVideoPath
      if (!videoPath) {
        videoPath = await uploadFile(videoFile)
        setUploadedVideoPath(videoPath)
      }

      // Upload voice sample if custom
      let voiceSamplePath = null
      if (voicePreset === 'custom' && voiceSampleFile) {
        setCurrentStep('Uploading voice sample...')
        voiceSamplePath = await uploadFile(voiceSampleFile)
      }

      setUploading(false)

      // Step 2: Generate TTS audio
      setCurrentStep('Generating speech...')

      const ttsFormData = new FormData()
      ttsFormData.append('text', text)
      ttsFormData.append('model', ttsModel)

      if (voicePreset === 'custom' && voiceSamplePath) {
        ttsFormData.append('voice_sample', voiceSamplePath)
      } else if (voicePreset !== 'custom') {
        ttsFormData.append('voice_preset', voicePreset)
      }

      if (DEBUG) console.log('🗣️ TTS request:', { text: text.slice(0, 50), model: ttsModel, voice: voicePreset })

      const ttsRes = await postForm(`${BACKEND_BASE}/voice-clone`, ttsFormData)

      if (!ttsRes.ok) {
        throw new Error(ttsRes.data?.detail || 'TTS generation failed')
      }

      // Get the generated audio path
      const audioPath = ttsRes.data?.path || ttsRes.data?.audio_path
      if (!audioPath) {
        throw new Error('TTS did not return audio path')
      }

      if (DEBUG) console.log('🎵 TTS audio generated:', audioPath)

      // Step 3: Apply lip sync
      setCurrentStep('Applying lip sync...')

      const lipSyncData = {
        video_path: videoPath,
        audio_path: audioPath,
        lips_expression: lipsExpression,
        inference_steps: inferenceSteps,
        seed: -1,
      }

      if (DEBUG) console.log('👄 Lip sync request:', lipSyncData)

      const lipSyncRes = await postJson(`${BACKEND_BASE}/lip-sync`, lipSyncData)

      if (!lipSyncRes.ok) {
        throw new Error(lipSyncRes.data?.detail || 'Lip sync failed')
      }

      // Show queued notification
      setLastQueued({
        promptId: lipSyncRes.data?.prompt_id,
        text: text.slice(0, 30) + (text.length > 30 ? '...' : '')
      })

      // Notify queue indicator
      if (onJobSubmitted) onJobSubmitted({ prompt_id: lipSyncRes.data?.prompt_id })

      if (DEBUG) console.log('✅ Speech-to-Video queued:', lipSyncRes.data?.prompt_id)

    } catch (err) {
      console.error('❌ Speech-to-Video error:', err)
      setError(err.message)
    } finally {
      setSubmitting(false)
      setUploading(false)
      setCurrentStep(null)
    }
  }

  const clearVideo = () => {
    setVideoFile(null)
    setVideoUrl(null)
    setUploadedVideoPath(null)
    setLastQueued(null)
  }

  const clearVoiceSample = () => {
    setVoiceSampleFile(null)
    setVoiceSampleUrl(null)
  }

  return (
    <div className="tool-container">
      {/* Video Upload */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <MessageSquare size={16} />
            Speech to Video
          </div>
        </div>

        <div
          className="upload-box"
          onClick={() => videoInputRef.current?.click()}
          onDrop={handleVideoDrop}
          onDragOver={(e) => e.preventDefault()}
          style={{ cursor: 'pointer' }}
        >
          <input
            ref={videoInputRef}
            type="file"
            accept="video/*"
            onChange={handleVideoDrop}
            style={{ display: 'none' }}
          />
          {videoUrl ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px', width: '100%' }}>
              <video
                ref={videoRef}
                src={videoUrl}
                controls
                muted
                style={{ maxHeight: '160px', borderRadius: '8px', maxWidth: '100%' }}
              />
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{videoFile?.name}</span>
                <button
                  onClick={(e) => { e.stopPropagation(); clearVideo() }}
                  className="icon-btn"
                  style={{ width: '24px', height: '24px', padding: '4px', color: '#ef4444' }}
                >
                  <X size={14} />
                </button>
              </div>
            </div>
          ) : (
            <>
              <Video size={32} className="text-muted" />
              <div className="text-muted">Drop video here or click to upload</div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>MP4, WebM supported</span>
            </>
          )}
        </div>
      </div>

      {/* Text Input */}
      <div className="grok-card">
        <div className="form-group">
          <label className="grok-section-label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <MessageSquare size={14} />
            Text to Speak
          </label>
          <textarea
            className="form-textarea"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter the text you want the character to say..."
            rows={4}
            style={{ minHeight: '80px' }}
          />
          <div style={{ textAlign: 'right', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            {text.length} characters
          </div>
        </div>
      </div>

      {/* TTS Model Selection */}
      <div className="grok-card">
        <div className="form-group">
          <label className="grok-section-label">TTS Model</label>
          <div className="grok-toggle-group">
            {TTS_MODELS.map(model => (
              <button
                key={model.id}
                onClick={() => setTtsModel(model.id)}
                className={`grok-toggle-btn ${ttsModel === model.id ? 'active' : ''}`}
              >
                <div style={{ fontWeight: 500 }}>{model.label}</div>
                <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>{model.description}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Voice Selection */}
        <div className="form-group">
          <label className="grok-section-label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Mic size={14} />
            Voice
          </label>
          <select
            className="form-input"
            value={voicePreset}
            onChange={(e) => setVoicePreset(e.target.value)}
            style={{ cursor: 'pointer' }}
          >
            {VOICE_PRESETS.map(voice => (
              <option key={voice.id} value={voice.id}>
                {voice.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Custom Voice Sample */}
      {voicePreset === 'custom' && (
        <div className="grok-card">
          <div
            className="upload-box"
            onClick={() => voiceInputRef.current?.click()}
            onDrop={handleVoiceSampleDrop}
            onDragOver={(e) => e.preventDefault()}
            style={{ cursor: 'pointer', padding: '20px' }}
          >
            <input
              ref={voiceInputRef}
              type="file"
              accept="audio/*"
              onChange={handleVoiceSampleDrop}
              style={{ display: 'none' }}
            />
            {voiceSampleUrl ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px', width: '100%' }}>
                <audio src={voiceSampleUrl} controls style={{ maxWidth: '100%' }} />
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{voiceSampleFile?.name}</span>
                  <button
                    onClick={(e) => { e.stopPropagation(); clearVoiceSample() }}
                    className="icon-btn"
                    style={{ width: '24px', height: '24px', padding: '4px', color: '#ef4444' }}
                  >
                    <X size={14} />
                  </button>
                </div>
              </div>
            ) : (
              <>
                <Volume2 size={24} className="text-muted" />
                <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Upload voice sample (5-15 sec recommended)</span>
              </>
            )}
          </div>
        </div>
      )}

      {/* Advanced Settings */}
      <div className="grok-card" style={{ padding: 0 }}>
        <div
          onClick={() => setShowAdvanced(!showAdvanced)}
          style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            padding: '16px 20px', cursor: 'pointer',
            color: 'var(--text-secondary)',
          }}
        >
          <Settings2 size={16} />
          <span style={{ fontSize: '0.85rem', fontWeight: 500 }}>Lip Sync Settings</span>
          <ChevronDown size={16} style={{
            marginLeft: 'auto',
            transform: showAdvanced ? 'rotate(180deg)' : 'rotate(0deg)',
            transition: 'transform 0.2s'
          }} />
        </div>

        {showAdvanced && (
          <div style={{ padding: '0 20px 20px', borderTop: '1px solid var(--border-color)' }}>
            <div className="form-group" style={{ marginTop: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <label className="grok-section-label" style={{ marginBottom: 0 }}>Lips Expression</label>
                <span className="nav-badge" style={{ fontSize: '0.8rem' }}>{lipsExpression.toFixed(1)}</span>
              </div>
              <div style={{ position: 'relative', height: '24px', marginBottom: '8px' }}>
                <input
                  type="range"
                  min={0.5}
                  max={3}
                  step={0.1}
                  value={lipsExpression}
                  onChange={(e) => setLipsExpression(parseFloat(e.target.value))}
                  style={{ width: '100%', opacity: 0, position: 'absolute', zIndex: 2, cursor: 'pointer' }}
                />
                <div style={{
                  position: 'absolute', top: '10px', left: 0, right: 0,
                  height: '4px', backgroundColor: '#333', borderRadius: '2px'
                }}>
                  <div style={{
                    width: `${((lipsExpression - 0.5) / 2.5) * 100}%`,
                    height: '100%', backgroundColor: 'var(--accent-color, #a855f7)', borderRadius: '2px'
                  }} />
                </div>
                <div style={{
                  position: 'absolute', top: '2px',
                  left: `calc(${((lipsExpression - 0.5) / 2.5) * 100}% - 10px)`,
                  width: '20px', height: '20px', backgroundColor: 'white',
                  borderRadius: '50%', boxShadow: '0 2px 4px rgba(0,0,0,0.3)'
                }} />
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Higher = more pronounced lip movement
              </div>
            </div>

            <div className="form-group">
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <label className="grok-section-label" style={{ marginBottom: 0 }}>Inference Steps</label>
                <span className="nav-badge" style={{ fontSize: '0.8rem' }}>{inferenceSteps}</span>
              </div>
              <div style={{ position: 'relative', height: '24px', marginBottom: '8px' }}>
                <input
                  type="range"
                  min={10}
                  max={50}
                  step={5}
                  value={inferenceSteps}
                  onChange={(e) => setInferenceSteps(parseInt(e.target.value))}
                  style={{ width: '100%', opacity: 0, position: 'absolute', zIndex: 2, cursor: 'pointer' }}
                />
                <div style={{
                  position: 'absolute', top: '10px', left: 0, right: 0,
                  height: '4px', backgroundColor: '#333', borderRadius: '2px'
                }}>
                  <div style={{
                    width: `${((inferenceSteps - 10) / 40) * 100}%`,
                    height: '100%', backgroundColor: 'var(--accent-color, #a855f7)', borderRadius: '2px'
                  }} />
                </div>
                <div style={{
                  position: 'absolute', top: '2px',
                  left: `calc(${((inferenceSteps - 10) / 40) * 100}% - 10px)`,
                  width: '20px', height: '20px', backgroundColor: 'white',
                  borderRadius: '50%', boxShadow: '0 2px 4px rgba(0,0,0,0.3)'
                }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                <span>10 (fast)</span>
                <span>50 (quality)</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Generate Button */}
      <button
        className="primary-btn"
        onClick={handleGenerate}
        disabled={submitting || !videoFile || !text.trim()}
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
      >
        {submitting ? (
          <>
            <Loader2 size={18} className="animate-spin" />
            {currentStep || 'Processing...'}
          </>
        ) : (
          <>
            <MessageSquare size={18} />
            Generate Speech Video
          </>
        )}
      </button>

      {/* Queued confirmation */}
      {lastQueued && (
        <div style={{
          padding: '12px', backgroundColor: 'rgba(34, 197, 94, 0.1)',
          border: '1px solid rgba(34, 197, 94, 0.3)', borderRadius: '8px',
          color: '#22c55e', marginTop: '12px', fontSize: '0.85rem',
        }}>
          ✅ Speech-to-Video queued! "{lastQueued.text}" - Check queue panel for progress
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{
          padding: '12px', backgroundColor: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '8px',
          color: '#ef4444', marginTop: '12px', fontSize: '0.85rem',
        }}>
          {error}
        </div>
      )}

      {/* Info */}
      <div className="info-badge" style={{ marginTop: '12px', textAlign: 'center' }}>
        This tool generates speech from your text using TTS, then applies lip sync to match the video.
      </div>
    </div>
  )
}
