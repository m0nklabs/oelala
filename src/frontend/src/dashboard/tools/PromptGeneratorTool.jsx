import React, { useState, useCallback } from 'react'
import { Sparkles, Copy, RefreshCw, Loader2, Wand2, Send } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'

const STYLE_PRESETS = [
  { id: 'cinematic', label: '🎬 Cinematic', keywords: 'cinematic lighting, film grain, dramatic shadows, professional photography' },
  { id: 'anime', label: '🎌 Anime', keywords: 'anime style, vibrant colors, cel shading, Japanese animation' },
  { id: 'photorealistic', label: '📸 Photorealistic', keywords: 'photorealistic, highly detailed, 8k, sharp focus, professional photo' },
  { id: 'abstract', label: '🎨 Abstract', keywords: 'abstract art, geometric shapes, vibrant colors, artistic' },
  { id: 'vintage', label: '📼 Vintage', keywords: 'vintage aesthetic, retro, film photography, nostalgic, 1970s' },
  { id: 'cyberpunk', label: '🤖 Cyberpunk', keywords: 'cyberpunk, neon lights, futuristic, dystopian, high tech low life' },
  { id: 'fantasy', label: '🧙 Fantasy', keywords: 'fantasy art, magical, ethereal lighting, mystical, enchanted' },
  { id: 'minimalist', label: '⬜ Minimalist', keywords: 'minimalist, clean, simple, negative space, modern' },
  { id: 'horror', label: '👻 Horror', keywords: 'dark atmosphere, eerie, horror, unsettling, creepy' },
  { id: 'scifi', label: '🚀 Sci-Fi', keywords: 'science fiction, futuristic, space, advanced technology' },
]

const ENHANCEMENT_MODES = [
  { id: 'expand', label: 'Expand', description: 'Add more details and context' },
  { id: 'refine', label: 'Refine', description: 'Improve grammar and structure' },
  { id: 'variations', label: 'Variations', description: 'Generate 3 alternatives' },
]

export default function PromptGeneratorTool({ onSendToTool }) {
  const [input, setInput] = useState('')
  const [style, setStyle] = useState('')
  const [enhanceMode, setEnhanceMode] = useState('expand')
  const [includeNegative, setIncludeNegative] = useState(true)
  const [includeMotion, setIncludeMotion] = useState(false)

  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleGenerate = async () => {
    if (!input.trim()) return

    setLoading(true)
    setError(null)

    try {
      const res = await fetch(`${BACKEND_BASE}/generate-prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: input.trim(),
          style: style || null,
          mode: enhanceMode,
          include_negative: includeNegative,
          include_motion: includeMotion,
        }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Generation failed')
      }

      const data = await res.json()
      setResult(data)

      if (DEBUG) console.log('✨ Prompt result:', data)
    } catch (err) {
      console.error('Prompt generation error:', err)
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Quick template-based generation (no API needed)
  const handleQuickGenerate = () => {
    if (!input.trim()) return

    const basePrompt = input.trim()
    const stylePreset = STYLE_PRESETS.find(s => s.id === style)
    const styleKeywords = stylePreset ? `, ${stylePreset.keywords}` : ''

    const enhancedPrompt = `${basePrompt}${styleKeywords}, masterpiece, best quality, highly detailed`

    const negativePrompt = includeNegative
      ? 'ugly, deformed, blurry, low quality, bad anatomy, watermark, signature, text, cropped, worst quality'
      : ''

    const motionPrompt = includeMotion
      ? 'smooth camera motion, cinematic movement, fluid animation'
      : ''

    setResult({
      prompt: enhancedPrompt,
      negative_prompt: negativePrompt,
      motion_prompt: motionPrompt,
      variations: null,
    })
  }

  const handleCopy = (text) => {
    navigator.clipboard.writeText(text)
  }

  return (
    <div className="tool-container">
      <div className="tool-section">
        <h3>
          <Sparkles size={18} />
          Input Idea
        </h3>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Describe your image or video idea... (e.g., 'a cat wearing sunglasses')"
          rows={3}
          className="prompt-input"
        />
      </div>

      <div className="tool-section">
        <h3>Style Preset</h3>
        <div className="style-grid">
          {STYLE_PRESETS.map((s) => (
            <button
              key={s.id}
              className={`style-btn ${style === s.id ? 'active' : ''}`}
              onClick={() => setStyle(style === s.id ? '' : s.id)}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      <div className="tool-section">
        <h3>Options</h3>
        <div className="options-row">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={includeNegative}
              onChange={(e) => setIncludeNegative(e.target.checked)}
            />
            Generate negative prompt
          </label>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={includeMotion}
              onChange={(e) => setIncludeMotion(e.target.checked)}
            />
            Include motion prompts (for video)
          </label>
        </div>
      </div>

      <div className="button-row">
        <button
          className="btn-primary btn-large"
          onClick={handleQuickGenerate}
          disabled={!input.trim()}
        >
          <Wand2 size={18} />
          Quick Generate
        </button>
        <button
          className="btn-secondary btn-large"
          onClick={handleGenerate}
          disabled={!input.trim() || loading}
          title="Uses AI for smarter enhancement (requires LLM)"
        >
          {loading ? (
            <>
              <Loader2 size={18} className="spin" />
              Generating...
            </>
          ) : (
            <>
              <Sparkles size={18} />
              AI Enhance
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}

      {result && (
        <div className="results-section">
          <div className="result-card">
            <div className="result-header">
              <h4>✨ Enhanced Prompt</h4>
              <button className="btn-icon" onClick={() => handleCopy(result.prompt)}>
                <Copy size={16} />
              </button>
            </div>
            <p className="result-text">{result.prompt}</p>
          </div>

          {result.negative_prompt && (
            <div className="result-card">
              <div className="result-header">
                <h4>🚫 Negative Prompt</h4>
                <button className="btn-icon" onClick={() => handleCopy(result.negative_prompt)}>
                  <Copy size={16} />
                </button>
              </div>
              <p className="result-text muted">{result.negative_prompt}</p>
            </div>
          )}

          {result.motion_prompt && (
            <div className="result-card">
              <div className="result-header">
                <h4>🎬 Motion Prompt</h4>
                <button className="btn-icon" onClick={() => handleCopy(result.motion_prompt)}>
                  <Copy size={16} />
                </button>
              </div>
              <p className="result-text">{result.motion_prompt}</p>
            </div>
          )}

          {result.variations && result.variations.length > 0 && (
            <div className="result-card">
              <h4>🔄 Variations</h4>
              {result.variations.map((v, i) => (
                <div key={i} className="variation-item">
                  <p className="result-text">{v}</p>
                  <button className="btn-icon" onClick={() => handleCopy(v)}>
                    <Copy size={16} />
                  </button>
                </div>
              ))}
            </div>
          )}

          {onSendToTool && (
            <button className="btn-primary" onClick={() => onSendToTool(result)}>
              <Send size={16} />
              Send to Generator
            </button>
          )}
        </div>
      )}

      <style>{`
        .prompt-input {
          width: 100%;
          padding: 12px;
          border-radius: 8px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          font-family: inherit;
          font-size: 14px;
          resize: vertical;
        }
        .style-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
          gap: 8px;
        }
        .style-btn {
          padding: 10px 12px;
          border: 1px solid var(--border-color, #444);
          border-radius: 8px;
          background: transparent;
          color: var(--text-color, #fff);
          cursor: pointer;
          transition: all 0.2s;
          font-size: 13px;
        }
        .style-btn:hover {
          border-color: var(--accent-color, #7c3aed);
          background: rgba(124, 58, 237, 0.1);
        }
        .style-btn.active {
          background: var(--accent-color, #7c3aed);
          border-color: var(--accent-color, #7c3aed);
        }
        .options-row {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .checkbox-label {
          display: flex;
          align-items: center;
          gap: 8px;
          cursor: pointer;
        }
        .checkbox-label input {
          width: 16px;
          height: 16px;
        }
        .button-row {
          display: flex;
          gap: 12px;
          margin-top: 16px;
        }
        .btn-large {
          flex: 1;
          padding: 14px 20px;
          font-size: 15px;
        }
        .results-section {
          margin-top: 24px;
          display: flex;
          flex-direction: column;
          gap: 16px;
        }
        .result-card {
          background: var(--bg-secondary, #1a1a1a);
          border: 1px solid var(--border-color, #444);
          border-radius: 12px;
          padding: 16px;
        }
        .result-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
        }
        .result-header h4 {
          margin: 0;
          font-size: 14px;
        }
        .result-text {
          margin: 0;
          line-height: 1.5;
          word-break: break-word;
        }
        .result-text.muted {
          color: var(--text-muted, #888);
        }
        .variation-item {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 12px;
          padding: 8px 0;
          border-bottom: 1px solid var(--border-color, #333);
        }
        .variation-item:last-child {
          border-bottom: none;
        }
        .btn-icon {
          background: none;
          border: none;
          color: var(--text-muted, #888);
          cursor: pointer;
          padding: 4px;
          border-radius: 4px;
        }
        .btn-icon:hover {
          color: var(--text-color, #fff);
          background: var(--bg-hover, #333);
        }
        .error-message {
          padding: 12px;
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 8px;
          color: #ef4444;
          margin-top: 12px;
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
