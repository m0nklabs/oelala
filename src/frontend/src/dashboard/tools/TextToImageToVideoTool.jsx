import React, { useState } from 'react'
import { Type, Image as ImageIcon, Film, ArrowRight, Sparkles } from 'lucide-react'
import { BACKEND_BASE } from '../../config'

export default function TextToImageToVideoTool({ onOutput }) {
  // Step 1: Text to Image
  const [t2iPrompt, setT2iPrompt] = useState('')
  const [aspectRatio, setAspectRatio] = useState('16:9')
  const [isGeneratingImage, setIsGeneratingImage] = useState(false)
  const [generatedImage, setGeneratedImage] = useState(null) // URL or Blob

  // Step 2: Image to Video
  const [i2vPrompt, setI2vPrompt] = useState('')
  const [numFrames, setNumFrames] = useState(16)
  const [isGeneratingVideo, setIsGeneratingVideo] = useState(false)

  const handleGenerateImage = async () => {
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
            style={{ backgroundColor: '#0f0f0f', border: 'none', resize: 'none' }}
          />
        </div>

        <div className="form-group">
          <label className="grok-section-label">Aspect Ratio</label>
          <div className="aspect-grid">
            {[
              { label: '16:9', icon: <div style={{ width: '24px', height: '14px', border: '2px solid currentColor', borderRadius: '2px' }} /> },
              { label: '9:16', icon: <div style={{ width: '14px', height: '24px', border: '2px solid currentColor', borderRadius: '2px' }} /> },
              { label: '1:1', icon: <div style={{ width: '20px', height: '20px', border: '2px solid currentColor', borderRadius: '2px' }} /> },
              { label: '21:9', icon: <div style={{ width: '28px', height: '12px', border: '2px solid currentColor', borderRadius: '2px' }} /> },
            ].map((ratio) => (
              <button
                key={ratio.label}
                className={`aspect-btn ${aspectRatio === ratio.label ? 'active' : ''}`}
                onClick={() => setAspectRatio(ratio.label)}
              >
                <div className="aspect-icon" style={{ background: 'transparent', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {ratio.icon}
                </div>
                <span className="aspect-label">{ratio.label}</span>
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
            style={{ backgroundColor: '#0f0f0f', border: 'none', resize: 'none' }}
          />
        </div>

        <div className="form-group">
          <label className="grok-section-label">Duration ({numFrames} frames)</label>
          <input
            type="range"
            min="8"
            max="32"
            step="4"
            value={numFrames}
            onChange={(e) => setNumFrames(parseInt(e.target.value, 10))}
            disabled={!generatedImage}
            style={{ width: '100%', accentColor: 'var(--text-primary)' }}
          />
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
