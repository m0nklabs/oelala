import React, { useState, useRef } from 'react'
import { Volume2, Music, Mic, Loader2, Play, Pause, Download, Settings, ChevronDown } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm, postJson } from '../../api'

const TTS_VOICES = [
  { value: 'alloy', label: 'Alloy', desc: 'Neutral, versatile' },
  { value: 'echo', label: 'Echo', desc: 'Warm, conversational' },
  { value: 'fable', label: 'Fable', desc: 'Expressive, dramatic' },
  { value: 'onyx', label: 'Onyx', desc: 'Deep, authoritative' },
  { value: 'nova', label: 'Nova', desc: 'Friendly, upbeat' },
  { value: 'shimmer', label: 'Shimmer', desc: 'Soft, gentle' },
]

const AUDIO_MODES = [
  { value: 'tts', label: 'Text to Speech', icon: <Mic size={18} />, desc: 'Generate voice from text' },
  { value: 'music', label: 'Music Generation', icon: <Music size={18} />, desc: 'Generate music/sounds' },
  { value: 'sfx', label: 'Sound Effects', icon: <Volume2 size={18} />, desc: 'Generate sound effects' },
]

const MUSIC_STYLES = [
  { value: 'ambient', label: 'Ambient' },
  { value: 'cinematic', label: 'Cinematic' },
  { value: 'electronic', label: 'Electronic' },
  { value: 'jazz', label: 'Jazz' },
  { value: 'classical', label: 'Classical' },
  { value: 'lofi', label: 'Lo-Fi' },
  { value: 'rock', label: 'Rock' },
  { value: 'hiphop', label: 'Hip-Hop' },
]

export default function AudioGenerationTool({ onOutput }) {
  const [mode, setMode] = useState('tts')
  const [text, setText] = useState('')
  const [voice, setVoice] = useState('nova')
  const [musicStyle, setMusicStyle] = useState('cinematic')
  const [duration, setDuration] = useState(10)
  const [showAdvanced, setShowAdvanced] = useState(false)
  
  // Advanced TTS
  const [speed, setSpeed] = useState(1.0)
  const [pitch, setPitch] = useState(1.0)
  
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [status, setStatus] = useState('')
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState(null)
  const [isPlaying, setIsPlaying] = useState(false)
  
  const audioRef = useRef(null)

  // Poll for completion
  const pollForCompletion = async (promptId, maxAttempts = 120) => {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      await new Promise(resolve => setTimeout(resolve, 1000))
      try {
        const res = await fetch(`${BACKEND_BASE}/comfyui/job/${promptId}`)
        if (!res.ok) continue
        const data = await res.json()
        
        if (data.status === 'pending') {
          setStatus('Queued...')
          setProgress(Math.min(10, attempt * 2))
        } else if (data.status === 'running') {
          setStatus('Generating audio...')
          setProgress(Math.min(90, 10 + attempt * 2))
        } else if (data.status === 'completed') {
          setProgress(100)
          return data
        } else if (data.status === 'failed') {
          throw new Error(data.error || 'Generation failed')
        }
      } catch (e) {
        if (e.message.includes('failed')) throw e
      }
    }
    throw new Error('Generation timed out')
  }

  const handleGenerate = async () => {
    if (!text.trim()) return
    
    setLoading(true)
    setError(null)
    setStatus('Starting...')
    setProgress(0)
    setResult(null)
    
    try {
      let endpoint = '/generate-audio'
      const payload = {
        text: text.trim(),
        mode,
      }
      
      if (mode === 'tts') {
        payload.voice = voice
        payload.speed = speed
        payload.pitch = pitch
      } else if (mode === 'music') {
        payload.style = musicStyle
        payload.duration = duration
      } else if (mode === 'sfx') {
        payload.duration = Math.min(duration, 5) // SFX shorter
      }
      
      if (DEBUG) console.debug('🎵 Audio request:', payload)
      
      const res = await postJson(`${BACKEND_BASE}${endpoint}`, payload)
      
      if (!res.ok) {
        throw new Error(res.data?.detail || 'Audio generation failed')
      }
      
      // Check if async (has prompt_id) or sync (has url directly)
      if (res.data?.prompt_id) {
        setStatus('Queued...')
        const completed = await pollForCompletion(res.data.prompt_id)
        
        if (completed.output_audio || completed.url) {
          const audioUrl = completed.output_audio || completed.url
          const fullUrl = audioUrl.startsWith('http') ? audioUrl : `${BACKEND_BASE}${audioUrl}`
          setResult({ url: fullUrl, filename: audioUrl.split('/').pop() })
        }
      } else if (res.data?.url) {
        const audioUrl = res.data.url
        const fullUrl = audioUrl.startsWith('http') ? audioUrl : `${BACKEND_BASE}${audioUrl}`
        setResult({ url: fullUrl, filename: audioUrl.split('/').pop() })
      }
      
      if (onOutput && result) {
        onOutput({
          kind: 'audio',
          url: result.url,
          filename: result.filename,
        })
      }
      
    } catch (err) {
      console.error('Audio error:', err)
      setError(err.message)
    } finally {
      setLoading(false)
      setStatus('')
      setProgress(0)
    }
  }

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

  return (
    <div className="tool-container">
      {/* Mode Selection */}
      <div className="tool-section">
        <h3>
          <Volume2 size={18} />
          Generation Mode
        </h3>
        <div className="mode-grid">
          {AUDIO_MODES.map((m) => (
            <button
              key={m.value}
              className={`mode-btn ${mode === m.value ? 'active' : ''}`}
              onClick={() => setMode(m.value)}
            >
              {m.icon}
              <span className="mode-name">{m.label}</span>
              <span className="mode-desc">{m.desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Input */}
      <div className="tool-section">
        <h3>
          {mode === 'tts' ? 'Text to Speak' : mode === 'music' ? 'Music Prompt' : 'Sound Description'}
        </h3>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={
            mode === 'tts' 
              ? 'Enter the text you want to convert to speech...'
              : mode === 'music'
              ? 'Describe the music you want to generate (e.g., "upbeat electronic dance track with heavy bass")'
              : 'Describe the sound effect (e.g., "thunder rumbling in the distance")'
          }
          rows={4}
          className="prompt-textarea"
        />
      </div>

      {/* TTS Voice Selection */}
      {mode === 'tts' && (
        <div className="tool-section">
          <h3>Voice</h3>
          <div className="voice-grid">
            {TTS_VOICES.map((v) => (
              <button
                key={v.value}
                className={`voice-btn ${voice === v.value ? 'active' : ''}`}
                onClick={() => setVoice(v.value)}
              >
                <span className="voice-name">{v.label}</span>
                <span className="voice-desc">{v.desc}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Music Style */}
      {mode === 'music' && (
        <div className="tool-section">
          <h3>Style</h3>
          <div className="style-grid">
            {MUSIC_STYLES.map((s) => (
              <button
                key={s.value}
                className={`style-btn ${musicStyle === s.value ? 'active' : ''}`}
                onClick={() => setMusicStyle(s.value)}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Duration (for music/sfx) */}
      {(mode === 'music' || mode === 'sfx') && (
        <div className="tool-section">
          <h3>Duration</h3>
          <div className="slider-row">
            <input
              type="range"
              min={mode === 'sfx' ? 1 : 5}
              max={mode === 'sfx' ? 10 : 30}
              value={duration}
              onChange={(e) => setDuration(parseInt(e.target.value))}
            />
            <span className="slider-value">{duration}s</span>
          </div>
        </div>
      )}

      {/* Advanced Settings */}
      {mode === 'tts' && (
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
              <div className="slider-row">
                <label>Speed</label>
                <input
                  type="range"
                  min={0.5}
                  max={2}
                  step={0.1}
                  value={speed}
                  onChange={(e) => setSpeed(parseFloat(e.target.value))}
                />
                <span className="slider-value">{speed.toFixed(1)}x</span>
              </div>
              <div className="slider-row">
                <label>Pitch</label>
                <input
                  type="range"
                  min={0.5}
                  max={2}
                  step={0.1}
                  value={pitch}
                  onChange={(e) => setPitch(parseFloat(e.target.value))}
                />
                <span className="slider-value">{pitch.toFixed(1)}x</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Progress */}
      {loading && (
        <div className="progress-section">
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
          <div className="progress-status">
            <Loader2 size={16} className="spin" />
            {status}
          </div>
        </div>
      )}

      {error && <div className="error-message">⚠️ {error}</div>}

      <button
        className="btn-primary btn-large"
        onClick={handleGenerate}
        disabled={!text.trim() || loading}
      >
        {loading ? (
          <>
            <Loader2 size={18} className="spin" />
            Generating...
          </>
        ) : (
          <>
            <Volume2 size={18} />
            Generate {mode === 'tts' ? 'Speech' : mode === 'music' ? 'Music' : 'Sound'}
          </>
        )}
      </button>

      {/* Result */}
      {result && (
        <div className="result-section">
          <h3>Result</h3>
          <div className="audio-player">
            <audio 
              ref={audioRef} 
              src={result.url}
              onEnded={() => setIsPlaying(false)}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
            />
            <button className="play-btn" onClick={togglePlay}>
              {isPlaying ? <Pause size={24} /> : <Play size={24} />}
            </button>
            <div className="audio-info">
              <span className="audio-filename">{result.filename}</span>
            </div>
            <a href={result.url} download className="download-btn">
              <Download size={18} />
            </a>
          </div>
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
        .mode-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 8px;
        }
        .mode-btn {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 6px;
          padding: 16px 12px;
          border: 1px solid var(--border-color, #444);
          border-radius: 8px;
          background: var(--bg-secondary, #1a1a1a);
          cursor: pointer;
          transition: all 0.2s;
        }
        .mode-btn:hover {
          border-color: var(--accent-color, #7c3aed);
        }
        .mode-btn.active {
          background: rgba(124, 58, 237, 0.2);
          border-color: var(--accent-color, #7c3aed);
        }
        .mode-name {
          font-size: 13px;
          font-weight: 500;
          color: var(--text-color, #fff);
        }
        .mode-desc {
          font-size: 10px;
          color: var(--text-muted, #888);
          text-align: center;
        }
        .prompt-textarea {
          width: 100%;
          padding: 12px;
          border-radius: 8px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          font-size: 14px;
          resize: none;
        }
        .voice-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 8px;
        }
        .voice-btn {
          padding: 10px;
          border: 1px solid var(--border-color, #444);
          border-radius: 8px;
          background: var(--bg-secondary, #1a1a1a);
          cursor: pointer;
          transition: all 0.2s;
          text-align: left;
        }
        .voice-btn:hover {
          border-color: var(--accent-color, #7c3aed);
        }
        .voice-btn.active {
          background: rgba(124, 58, 237, 0.2);
          border-color: var(--accent-color, #7c3aed);
        }
        .voice-name {
          display: block;
          font-size: 13px;
          font-weight: 500;
          color: var(--text-color, #fff);
        }
        .voice-desc {
          display: block;
          font-size: 10px;
          color: var(--text-muted, #888);
        }
        .style-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 6px;
        }
        .style-btn {
          padding: 8px 12px;
          border: 1px solid var(--border-color, #444);
          border-radius: 6px;
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          font-size: 12px;
          cursor: pointer;
          transition: all 0.2s;
        }
        .style-btn:hover {
          border-color: var(--accent-color, #7c3aed);
        }
        .style-btn.active {
          background: var(--accent-color, #7c3aed);
          border-color: var(--accent-color, #7c3aed);
        }
        .slider-row {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .slider-row label {
          min-width: 60px;
          font-size: 13px;
          color: var(--text-secondary, #aaa);
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
        .audio-player {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 16px;
          background: var(--bg-secondary, #1a1a1a);
          border-radius: 12px;
        }
        .play-btn {
          width: 48px;
          height: 48px;
          border-radius: 50%;
          border: none;
          background: var(--accent-color, #7c3aed);
          color: white;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: transform 0.2s;
        }
        .play-btn:hover {
          transform: scale(1.05);
        }
        .audio-info {
          flex: 1;
        }
        .audio-filename {
          font-size: 13px;
          color: var(--text-color, #fff);
        }
        .download-btn {
          padding: 8px;
          border-radius: 6px;
          background: var(--bg-tertiary, #252525);
          color: var(--text-color, #fff);
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .download-btn:hover {
          background: var(--border-color, #444);
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
