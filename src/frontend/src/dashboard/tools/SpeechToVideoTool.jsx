import React, { useState, useRef, useCallback } from 'react'
import { 
  Video, Upload, Play, Pause, Download, Loader2, X, 
  MessageSquare, Volume2, Mic, Settings2, ChevronDown
} from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm, postJson } from '../../api'

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
    <div className="tool-container space-y-4 p-4">
      {/* Header */}
      <div className="text-center mb-4">
        <h2 className="text-xl font-bold text-white flex items-center justify-center gap-2">
          <MessageSquare className="w-6 h-6 text-purple-400" />
          Speech to Video
        </h2>
        <p className="text-gray-400 text-sm mt-1">
          Generate speech from text and sync it to a video
        </p>
      </div>

      {/* Video Upload */}
      <div
        onClick={() => videoInputRef.current?.click()}
        onDrop={handleVideoDrop}
        onDragOver={(e) => e.preventDefault()}
        className="border-2 border-dashed border-gray-600 rounded-lg p-6 text-center cursor-pointer hover:border-purple-500 transition-colors"
      >
        <input
          ref={videoInputRef}
          type="file"
          accept="video/*"
          onChange={handleVideoDrop}
          className="hidden"
        />
        {videoUrl ? (
          <div className="space-y-2">
            <video
              ref={videoRef}
              src={videoUrl}
              className="max-h-40 mx-auto rounded"
              controls
              muted
            />
            <div className="flex items-center justify-center gap-2">
              <span className="text-sm text-gray-400">{videoFile?.name}</span>
              <button
                onClick={(e) => { e.stopPropagation(); clearVideo() }}
                className="p-1 text-red-400 hover:text-red-300"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2 text-gray-400">
            <Video className="w-10 h-10" />
            <span>Drop video here or click to upload</span>
            <span className="text-xs">MP4, WebM supported</span>
          </div>
        )}
      </div>

      {/* Text Input */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          <MessageSquare className="w-4 h-4 inline mr-1" />
          Text to Speak
        </label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter the text you want the character to say..."
          className="w-full px-3 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 resize-none"
          rows={4}
        />
        <div className="text-xs text-gray-500 mt-1 text-right">
          {text.length} characters
        </div>
      </div>

      {/* TTS Model Selection */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          TTS Model
        </label>
        <div className="grid grid-cols-2 gap-2">
          {TTS_MODELS.map(model => (
            <button
              key={model.id}
              onClick={() => setTtsModel(model.id)}
              className={`px-3 py-2 text-sm rounded transition-colors text-left ${
                ttsModel === model.id
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              <div className="font-medium">{model.label}</div>
              <div className="text-xs opacity-70">{model.description}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Voice Selection */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          <Mic className="w-4 h-4 inline mr-1" />
          Voice
        </label>
        <select
          value={voicePreset}
          onChange={(e) => setVoicePreset(e.target.value)}
          className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white"
        >
          {VOICE_PRESETS.map(voice => (
            <option key={voice.id} value={voice.id}>
              {voice.label}
            </option>
          ))}
        </select>
      </div>

      {/* Custom Voice Sample */}
      {voicePreset === 'custom' && (
        <div
          onClick={() => voiceInputRef.current?.click()}
          onDrop={handleVoiceSampleDrop}
          onDragOver={(e) => e.preventDefault()}
          className="border-2 border-dashed border-gray-600 rounded-lg p-4 text-center cursor-pointer hover:border-purple-500 transition-colors"
        >
          <input
            ref={voiceInputRef}
            type="file"
            accept="audio/*"
            onChange={handleVoiceSampleDrop}
            className="hidden"
          />
          {voiceSampleUrl ? (
            <div className="space-y-2">
              <audio src={voiceSampleUrl} controls className="mx-auto" />
              <div className="flex items-center justify-center gap-2">
                <span className="text-sm text-gray-400">{voiceSampleFile?.name}</span>
                <button
                  onClick={(e) => { e.stopPropagation(); clearVoiceSample() }}
                  className="p-1 text-red-400 hover:text-red-300"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2 text-gray-400">
              <Volume2 className="w-6 h-6" />
              <span className="text-sm">Upload voice sample (5-15 sec recommended)</span>
            </div>
          )}
        </div>
      )}

      {/* Advanced Settings */}
      <div className="border border-gray-700 rounded-lg overflow-hidden">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full px-4 py-2 bg-gray-800 flex items-center justify-between text-gray-300 hover:bg-gray-750"
        >
          <span className="text-sm font-medium flex items-center gap-2">
            <Settings2 className="w-4 h-4" />
            Lip Sync Settings
          </span>
          <ChevronDown className={`w-4 h-4 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
        </button>
        
        {showAdvanced && (
          <div className="p-4 space-y-4 bg-gray-850">
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Lips Expression: {lipsExpression.toFixed(1)}
              </label>
              <input
                type="range"
                min={0.5}
                max={3}
                step={0.1}
                value={lipsExpression}
                onChange={(e) => setLipsExpression(parseFloat(e.target.value))}
                className="w-full accent-purple-500"
              />
              <span className="text-xs text-gray-500">Higher = more pronounced lip movement</span>
            </div>
            
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Inference Steps: {inferenceSteps}
              </label>
              <input
                type="range"
                min={10}
                max={50}
                step={5}
                value={inferenceSteps}
                onChange={(e) => setInferenceSteps(parseInt(e.target.value))}
                className="w-full accent-purple-500"
              />
              <span className="text-xs text-gray-500">Higher = better quality, slower</span>
            </div>
          </div>
        )}
      </div>

      {/* Generate Button */}
      <button
        onClick={handleGenerate}
        disabled={submitting || !videoFile || !text.trim()}
        className="w-full py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg font-semibold flex items-center justify-center gap-2 transition-colors"
      >
        {submitting ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            {currentStep || 'Processing...'}
          </>
        ) : (
          <>
            <MessageSquare className="w-5 h-5" />
            Generate Speech Video
          </>
        )}
      </button>

      {/* Queued confirmation */}
      {lastQueued && (
        <div className="p-3 bg-green-900/50 border border-green-700 rounded-lg text-green-200 text-sm">
          ✅ Speech-to-Video queued! "{lastQueued.text}" - Check queue panel for progress
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-200 text-sm">
          {error}
        </div>
      )}

      {/* Info */}
      <div className="text-xs text-gray-500 text-center">
        This tool generates speech from your text using TTS, then applies lip sync to match the video.
      </div>
    </div>
  )
}
