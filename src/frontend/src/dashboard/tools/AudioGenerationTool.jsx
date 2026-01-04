import React, { useState, useRef } from 'react'
import { Volume2, Music, Mic, Loader2, Play, Pause, Download, Settings, ChevronDown } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm } from '../../api'

const TTS_VOICES = [
  // Female voices
  { value: 'nova', label: 'Nova', desc: 'Friendly, upbeat', gender: 'female' },
  { value: 'shimmer', label: 'Shimmer', desc: 'Soft, gentle', gender: 'female' },
  { value: 'alloy', label: 'Alloy', desc: 'Neutral, versatile', gender: 'female' },
  // Male voices
  { value: 'echo', label: 'Echo', desc: 'Warm, conversational', gender: 'male' },
  { value: 'fable', label: 'Fable', desc: 'Expressive, dramatic', gender: 'male' },
  { value: 'onyx', label: 'Onyx', desc: 'Deep, authoritative', gender: 'male' },
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

export default function AudioGenerationTool({ onOutput, onJobSubmitted }) {
  const [mode, setMode] = useState('tts')
  const [text, setText] = useState('')
  const [voice, setVoice] = useState('nova')
  const [musicStyle, setMusicStyle] = useState('cinematic')
  const [duration, setDuration] = useState(10)
  const [showAdvanced, setShowAdvanced] = useState(false)
  
  // Advanced TTS
  const [speed, setSpeed] = useState(1.0)
  const [pitch, setPitch] = useState(1.0)
  
  const [submitting, setSubmitting] = useState(false)  // Brief state while submitting
  const [error, setError] = useState(null)
  const [lastQueued, setLastQueued] = useState(null)   // Track last queued job
  const [result, setResult] = useState(null)
  const [isPlaying, setIsPlaying] = useState(false)
  
  const audioRef = useRef(null)

  const handleGenerate = async () => {
    if (!text.trim()) return
    
    setSubmitting(true)
    setError(null)
    setLastQueued(null)
    
    try {
      let endpoint = '/generate-audio'
      
      // Build FormData - backend expects Form parameters
      const formData = new FormData()
      formData.append('text', text.trim())
      formData.append('mode', mode)
      
      if (mode === 'tts') {
        formData.append('voice', voice)
        formData.append('speed', speed.toString())
        formData.append('pitch', pitch.toString())
      } else if (mode === 'music') {
        formData.append('style', musicStyle)
        formData.append('duration', duration.toString())
      } else if (mode === 'sfx') {
        formData.append('duration', Math.min(duration, 10).toString()) // SFX max 10s
      }
      
      if (DEBUG) console.debug('🎵 Audio request:', { text: text.trim(), mode, voice, musicStyle, duration })
      
      const res = await postForm(`${BACKEND_BASE}${endpoint}`, formData)
      
      if (!res.ok) {
        // Better error extraction
        const errMsg = typeof res.data === 'object' 
          ? (res.data?.detail || JSON.stringify(res.data)) 
          : (res.data || 'Audio generation failed')
        throw new Error(errMsg)
      }
      
      // Job was queued - notify parent and show confirmation
      if (res.data?.prompt_id) {
        setLastQueued({
          promptId: res.data.prompt_id,
          mode,
          text: text.substring(0, 50) + (text.length > 50 ? '...' : '')
        })
        
        // Notify parent to refresh queue panel
        if (onJobSubmitted) {
          onJobSubmitted(res.data)
        }
        
        // Output will appear in queue/history when done - don't wait
        if (DEBUG) console.debug('🎵 Job queued:', res.data.prompt_id)
      } else if (res.data?.url) {
        // Sync result - show immediately
        const audioUrl = res.data.url
        const fullUrl = audioUrl.startsWith('http') ? audioUrl : `${BACKEND_BASE}${audioUrl}`
        setResult({ url: fullUrl, filename: audioUrl.split('/').pop() })
        
        if (onOutput) {
          onOutput({
            kind: 'audio',
            url: fullUrl,
            filename: audioUrl.split('/').pop(),
          })
        }
      }
      
    } catch (err) {
      console.error('Audio error:', err)
      setError(err.message)
    } finally {
      setSubmitting(false)
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
          {/* Female voices */}
          <div className="voice-group">
            <span className="voice-group-label">Female</span>
            <div className="voice-grid">
              {TTS_VOICES.filter(v => v.gender === 'female').map((v) => (
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
          {/* Male voices */}
          <div className="voice-group">
            <span className="voice-group-label">Male</span>
            <div className="voice-grid">
              {TTS_VOICES.filter(v => v.gender === 'male').map((v) => (
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

      {/* Queued notification */}
      {lastQueued && (
        <div className="queued-notice">
          ✅ Job queued! Check the Queue panel for progress.
          <span className="queued-mode">{lastQueued.mode.toUpperCase()}</span>
        </div>
      )}

      {error && <div className="error-message">⚠️ {error}</div>}

      <button
        className="btn-primary btn-large"
        onClick={handleGenerate}
        disabled={!text.trim() || submitting}
      >
        {submitting ? (
          <>
            <Loader2 size={18} className="spin" />
            Queueing...
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
        .voice-group {
          margin-bottom: 12px;
        }
        .voice-group:last-child {
          margin-bottom: 0;
        }
        .voice-group-label {
          display: block;
          font-size: 11px;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          color: var(--text-muted, #888);
          margin-bottom: 8px;
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
