import React, { useState, useRef, useMemo, useEffect, useCallback } from 'react'
import { Volume2, Music, Mic, Loader2, Play, Pause, Download, Settings, ChevronDown } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm } from '../../api'
import { useAuth } from '../../contexts/AuthContext'
import { useToolSettings } from '../../hooks/useToolSettings'
import useGeneration from '../../hooks/useGeneration'
import ResetDefaultsButton from '../../components/ResetDefaultsButton'

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

const AUDIO_DEFAULTS = {
  mode: 'tts', text: '', voice: 'nova', musicStyle: 'cinematic',
  duration: 10, speed: 1.0, pitch: 1.0,
}

export default function AudioGenerationTool({ onOutput, onJobSubmitted }) {
  const { user, requestLogin } = useAuth()
  const { initial, save: saveSettings, resetDefaults } = useToolSettings('audio_generation', AUDIO_DEFAULTS)

  const [mode, setMode] = useState(initial.mode)
  const [text, setText] = useState(initial.text)
  const [voice, setVoice] = useState(initial.voice)
  const [musicStyle, setMusicStyle] = useState(initial.musicStyle)
  const [duration, setDuration] = useState(initial.duration)
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Advanced TTS
  const [speed, setSpeed] = useState(initial.speed)
  const [pitch, setPitch] = useState(initial.pitch)

  const [error, setError] = useState(null)
  const [lastQueued, setLastQueued] = useState(null)   // Track last queued job
  const [result, setResult] = useState(null)
  const [isPlaying, setIsPlaying] = useState(false)

  const { generate, loading: submitting } = useGeneration({
    onSuccess: (data) => {
      if (data.detail) {
        setError(data.detail)
        return
      }

      const promptId = data.id || data.prompt_id
      if (promptId) {
        setLastQueued({
          promptId,
          text: text.substring(0, 40) + (text.length > 40 ? '...' : '')
        })
        if (onJobSubmitted) onJobSubmitted({ prompt_id: promptId })
      }
    },
    onError: (err) => setError(err)
  })

  // ── Auto-save settings ──────────────────────────────────────────
  const settingsSnapshot = useMemo(() => ({
    mode, text, voice, musicStyle, duration, speed, pitch,
  }), [mode, text, voice, musicStyle, duration, speed, pitch])
  useEffect(() => { saveSettings(settingsSnapshot) }, [settingsSnapshot, saveSettings])

  const handleResetDefaults = useCallback(() => {
    const d = resetDefaults()
    setMode(d.mode); setText(d.text); setVoice(d.voice); setMusicStyle(d.musicStyle)
    setDuration(d.duration); setSpeed(d.speed); setPitch(d.pitch)
  }, [resetDefaults])

  const audioRef = useRef(null)

  const handleGenerate = async () => {
    // Check if user is logged in
    if (!user) {
      requestLogin('Log in om te genereren')
      return
    }

    if (!text.trim()) return

    setError(null)
    setLastQueued(null)

    try {
      const durationVal = mode === 'sfx' ? Math.min(duration, 10) : duration
      
      const reqPayload = {
        operation: 'generate',
        target_type: 'audio',
        adapter_hint: 'local-mmaudio',
        compute_target: 'local',
        prompts: {
          positive: text.trim()
        },
        temporal: {
           num_frames: 0,
           fps: 0
        },
        settings: {
          audio_mode: mode,
          duration: parseFloat(durationVal)
        }
      }

      if (mode === 'tts') {
        reqPayload.settings.voice = voice
        reqPayload.settings.speed = parseFloat(speed)
        reqPayload.settings.pitch = parseFloat(pitch)
      } else if (mode === 'music') {
        reqPayload.settings.audio_style = musicStyle
      }
      
      if (DEBUG) console.debug('🎵 V2 Audio req:', typeof reqPayload, reqPayload)

      await generate(reqPayload)
    } catch (err) {
      console.error('Audio error:', err)
      setError(err.message)
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
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Volume2 size={16} />
            Generation Mode
          </div>
          <ResetDefaultsButton onReset={handleResetDefaults} />
        </div>
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
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">
            {mode === 'tts' ? 'Text to Speak' : mode === 'music' ? 'Music Prompt' : 'Sound Description'}
          </div>
        </div>
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
          className="form-textarea"
        />
      </div>

      {/* TTS Voice Selection */}
      {mode === 'tts' && (
        <div className="grok-card">
          <div className="grok-card-header">
            <div className="grok-card-title">Voice</div>
          </div>
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
        <div className="grok-card">
          <div className="grok-card-header">
            <div className="grok-card-title">Style</div>
          </div>
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
        <div className="grok-card">
          <div className="grok-card-header">
            <div className="grok-card-title">Duration</div>
          </div>
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
        <div className="grok-card" style={{ padding: 0, overflow: 'hidden' }}>
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            style={{
              width: '100%', padding: '14px 20px', background: 'transparent', border: 'none',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              color: 'var(--text-secondary)', cursor: 'pointer',
            }}
          >
            <span style={{ fontSize: '0.85rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Settings size={16} />
              Advanced
            </span>
            <ChevronDown size={16} style={{ transition: 'transform 0.2s', transform: showAdvanced ? 'rotate(180deg)' : 'none' }} />
          </button>

          {showAdvanced && (
            <div style={{ padding: '0 20px 20px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div>
                <label className="grok-section-label">Speed <span className="nav-badge">{speed.toFixed(1)}x</span></label>
                <input
                  type="range" className="form-range"
                  min={0.5} max={2} step={0.1}
                  value={speed}
                  onChange={(e) => setSpeed(parseFloat(e.target.value))}
                />
              </div>
              <div>
                <label className="grok-section-label">Pitch <span className="nav-badge">{pitch.toFixed(1)}x</span></label>
                <input
                  type="range" className="form-range"
                  min={0.5} max={2} step={0.1}
                  value={pitch}
                  onChange={(e) => setPitch(parseFloat(e.target.value))}
                />
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
