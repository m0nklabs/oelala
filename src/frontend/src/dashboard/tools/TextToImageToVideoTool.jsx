import React, { useState, useMemo, useEffect, useCallback } from 'react'
import { Type, Image as ImageIcon, Film, ArrowRight, Sparkles } from 'lucide-react'
import { BACKEND_BASE } from '../../config'
import { useAuth } from '../../contexts/AuthContext'
import { useToolSettings } from '../../hooks/useToolSettings'
import ResetDefaultsButton from '../../components/ResetDefaultsButton'

const T2I2V_DEFAULTS = { t2iPrompt: '', aspectRatio: '16:9', i2vPrompt: '', numFrames: 16 }

export default function TextToImageToVideoTool({ onOutput }) {
  const { user, requestLogin } = useAuth()
  const { initial, save: saveSettings, resetDefaults } = useToolSettings('t2i2v', T2I2V_DEFAULTS)

  // Step 1: Text to Image
  const [t2iPrompt, setT2iPrompt] = useState(initial.t2iPrompt)
  const [aspectRatio, setAspectRatio] = useState(initial.aspectRatio)
  const [isGeneratingImage, setIsGeneratingImage] = useState(false)
  const [generatedImage, setGeneratedImage] = useState(null)

  // Step 2: Image to Video
  const [i2vPrompt, setI2vPrompt] = useState(initial.i2vPrompt)
  const [numFrames, setNumFrames] = useState(initial.numFrames)
  const [isGeneratingVideo, setIsGeneratingVideo] = useState(false)

  // Auto-save settings
  const settingsSnapshot = useMemo(() => ({ t2iPrompt, aspectRatio, i2vPrompt, numFrames }), [t2iPrompt, aspectRatio, i2vPrompt, numFrames])
  useEffect(() => { saveSettings(settingsSnapshot) }, [settingsSnapshot, saveSettings])

  const handleResetDefaults = useCallback(() => {
    const d = resetDefaults()
    setT2iPrompt(d.t2iPrompt); setAspectRatio(d.aspectRatio); setI2vPrompt(d.i2vPrompt); setNumFrames(d.numFrames)
  }, [resetDefaults])

  const handleGenerateImage = async () => {
    // Check if user is logged in
    if (!user) {
      requestLogin('Log in om te genereren')
      return
    }

    if (!t2iPrompt.trim()) return
    setIsGeneratingImage(true)

    // TODO: Implement actual T2I backend call
    // For now, simulate a delay and use a placeholder or fail
    setTimeout(() => {
      setIsGeneratingImage(false)
      // Mock success for UI demonstration
      // setGeneratedImage('https://placehold.co/1280x720/1a1a1a/FFF?text=Generated+Image')
      alert("Text-to-Image backend is not yet connected.")
    }, 1500)
  }

  const handleGenerateVideo = async () => {
    if (!generatedImage) return
    setIsGeneratingVideo(true)
    // TODO: Implement I2V call with generatedImage
    setTimeout(() => setIsGeneratingVideo(false), 2000)
  }

  return (
    <div className="tool-container">
      {/* Step 1: Text to Image */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Step 1: Text to Image</div>
          <ResetDefaultsButton onReset={handleResetDefaults} />
          <ImageIcon size={16} className="text-muted" />
        </div>

        <div className="form-group">
          <label className="grok-section-label">Image Prompt</label>
          <textarea
            className="form-textarea"
            value={t2iPrompt}
            onChange={(e) => setT2iPrompt(e.target.value)}
            placeholder="Describe the image you want to generate..."
            rows={3}
            style={{ minHeight: '70px' }}
          />
        </div>

        <div className="form-group">
          <label className="grok-section-label">Aspect Ratio</label>
          <div className="grok-toggle-group">
            {[
              { label: '16:9' },
              { label: '9:16' },
              { label: '1:1' },
              { label: '21:9' },
            ].map((ratio) => (
              <button
                key={ratio.label}
                className={`grok-toggle-btn ${aspectRatio === ratio.label ? 'active' : ''}`}
                onClick={() => setAspectRatio(ratio.label)}
              >
                {ratio.label}
              </button>
            ))}
          </div>
        </div>

        <button
          className="primary-btn"
          onClick={handleGenerateImage}
          disabled={isGeneratingImage || !t2iPrompt.trim()}
          style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
        >
          {isGeneratingImage ? 'Generating Image...' : <><Sparkles size={16} /> Generate Image</>}
        </button>
      </div>

      {/* Step 2: Image to Video */}
      <div className={`grok-card ${!generatedImage ? 'opacity-50' : ''}`} style={{ transition: 'opacity 0.3s' }}>
        <div className="grok-card-header">
          <div className="grok-card-title">Step 2: Animate</div>
          <Film size={16} className="text-muted" />
        </div>

        {generatedImage ? (
          <div className="form-group">
            <img
              src={generatedImage}
              alt="Generated"
              style={{ width: '100%', borderRadius: '8px', border: '1px solid var(--border-color)', marginBottom: '16px' }}
            />
          </div>
        ) : (
          <div className="upload-box" style={{ padding: '24px', marginBottom: '16px', borderStyle: 'dashed' }}>
            <div className="text-muted">Generate an image above to continue</div>
          </div>
        )}

        <div className="form-group">
          <label className="grok-section-label">Motion Prompt (Optional)</label>
          <textarea
            className="form-textarea"
            value={i2vPrompt}
            onChange={(e) => setI2vPrompt(e.target.value)}
            placeholder="Describe how the image should move..."
            rows={2}
            disabled={!generatedImage}
            style={{ minHeight: '50px' }}
          />
        </div>

        <div className="form-group">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <label className="grok-section-label" style={{ marginBottom: 0 }}>Duration</label>
            <span className="nav-badge" style={{ fontSize: '0.8rem' }}>{numFrames} frames</span>
          </div>
          <div style={{ position: 'relative', height: '24px', marginBottom: '8px' }}>
            <input
              type="range"
              min="8"
              max="481"
              step="4"
              value={numFrames}
              onChange={(e) => setNumFrames(parseInt(e.target.value, 10))}
              disabled={!generatedImage}
              style={{ width: '100%', opacity: 0, position: 'absolute', zIndex: 2, cursor: generatedImage ? 'pointer' : 'not-allowed' }}
            />
            <div style={{
              position: 'absolute', top: '10px', left: 0, right: 0,
              height: '4px', backgroundColor: '#333', borderRadius: '2px'
            }}>
              <div style={{
                width: `${((numFrames - 8) / (481 - 8)) * 100}%`,
                height: '100%', backgroundColor: 'var(--accent-color, #a855f7)', borderRadius: '2px'
              }} />
            </div>
            <div style={{
              position: 'absolute', top: '2px',
              left: `calc(${((numFrames - 8) / (481 - 8)) * 100}% - 10px)`,
              width: '20px', height: '20px', backgroundColor: 'white',
              borderRadius: '50%', boxShadow: '0 2px 4px rgba(0,0,0,0.3)'
            }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            <span>8 frames</span>
            <span>481 frames</span>
          </div>
        </div>

        <button
          className="primary-btn"
          onClick={handleGenerateVideo}
          disabled={!generatedImage || isGeneratingVideo}
          style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
        >
          {isGeneratingVideo ? 'Generating Video...' : <><Film size={16} /> Generate Video</>}
        </button>
      </div>
    </div>
  )
}
