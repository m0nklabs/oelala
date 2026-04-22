import React, { useState, useCallback, useMemo, useEffect } from 'react'
import { Sparkles, Copy, RefreshCw, Loader2, Wand2, Send, Flame } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import useLLMEnhance from '../../hooks/useLLMEnhance'
import LLMQueueIndicator from '../../components/LLMQueueIndicator'
import { PROMPT_LLM_MODELS, NSFW_LLM_MODELS, DEFAULT_PROMPT_LLM, DEFAULT_NSFW_LLM } from '../../constants/llmModels'
import { useNSFW } from '../../contexts/NSFWContext'
import { useAuth } from '../../contexts/AuthContext'
import { useToolSettings } from '../../hooks/useToolSettings'
import ResetDefaultsButton from '../../components/ResetDefaultsButton'
import CameraMotionSelector, { getCameraMotionPrefix } from '../../components/CameraMotionSelector'

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

const NSFW_STYLE_PRESETS = [
  { id: 'sensual', label: '💋 Sensual', keywords: 'sensual, intimate, soft lighting, romantic atmosphere' },
  { id: 'glamour', label: '✨ Glamour', keywords: 'glamour photography, studio lighting, alluring, provocative pose' },
  { id: 'boudoir', label: '🛏️ Boudoir', keywords: 'boudoir photography, intimate setting, soft fabrics, warm lighting' },
  { id: 'artistic_nude', label: '🎨 Art Nude', keywords: 'artistic nude, fine art, dramatic lighting, sculptural body' },
  { id: 'fetish', label: '⛓️ Fetish', keywords: 'fetish, leather, latex, dominant, submissive, dark aesthetic' },
  { id: 'hentai', label: '🎌 Hentai', keywords: 'hentai, anime style, ecchi, explicit, Japanese illustration' },
]

const NSFW_INTENSITY_LABELS = [
  '', // 0 unused
  '💋 Suggestive',
  '🔥 Softcore',
  '🔞 Nude',
  '💥 Hardcore',
  '⚡ Extreme',
]

const ENHANCEMENT_MODES = [
  { id: 'expand', label: 'Expand', description: 'Add more details and context' },
  { id: 'refine', label: 'Refine', description: 'Improve grammar and structure' },
  { id: 'variations', label: 'Variations', description: 'Generate 3 alternatives' },
]

// PROMPT_LLM_MODELS imported from shared constants/llmModels.js

const PROMPTGEN_DEFAULTS = {
  input: '', style: '', enhanceMode: 'expand', includeNegative: true,
  includeMotion: false, cameraMotion: '', nsfwMode: false, nsfwIntensity: 3, enhanceModel: DEFAULT_PROMPT_LLM,
}

export default function PromptGeneratorTool({ onSendToTool }) {
  const { nsfwEnabled } = useNSFW()
  const { user, requestLogin } = useAuth()
  const { initial, save: saveSettings, resetDefaults } = useToolSettings('prompt_generator', PROMPTGEN_DEFAULTS)
  const [input, setInput] = useState(initial.input)
  const [style, setStyle] = useState(initial.style)
  const [enhanceMode, setEnhanceMode] = useState(initial.enhanceMode)
  const [includeNegative, setIncludeNegative] = useState(initial.includeNegative)
  const [includeMotion, setIncludeMotion] = useState(initial.includeMotion)
  const [cameraMotion, setCameraMotion] = useState(initial.cameraMotion || '')
  const [nsfwMode, setNsfwMode] = useState(initial.nsfwMode)
  const [nsfwIntensity, setNsfwIntensity] = useState(initial.nsfwIntensity)

  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [enhanceModel, setEnhanceModel] = useState(initial.enhanceModel)

  // ── Auto-save settings ──────────────────────────────────────────
  const settingsSnapshot = useMemo(() => ({
    input, style, enhanceMode, includeNegative, includeMotion, cameraMotion,
    nsfwMode, nsfwIntensity, enhanceModel,
  }), [input, style, enhanceMode, includeNegative, includeMotion, cameraMotion,
    nsfwMode, nsfwIntensity, enhanceModel])
  useEffect(() => { saveSettings(settingsSnapshot) }, [settingsSnapshot, saveSettings])

  const handleResetDefaults = useCallback(() => {
    const d = resetDefaults()
    setInput(d.input); setStyle(d.style); setEnhanceMode(d.enhanceMode)
    setIncludeNegative(d.includeNegative); setIncludeMotion(d.includeMotion)
    setCameraMotion(d.cameraMotion || '')
    setNsfwMode(d.nsfwMode); setNsfwIntensity(d.nsfwIntensity); setEnhanceModel(d.enhanceModel)
  }, [resetDefaults])

  // Auto-switch model list when toggling NSFW
  const activeModels = nsfwMode ? NSFW_LLM_MODELS : PROMPT_LLM_MODELS
  const activeStyles = nsfwMode ? NSFW_STYLE_PRESETS : STYLE_PRESETS

  const handleNsfwToggle = (enabled) => {
    setNsfwMode(enabled)
    setStyle('') // reset style when switching
    setEnhanceModel(enabled ? DEFAULT_NSFW_LLM : DEFAULT_PROMPT_LLM)
  }

  // LLM prompt enhancement queue
  const llm = useLLMEnhance()

  const handleGenerate = async () => {
    if (!user) { requestLogin('Log in om prompts te genereren'); return }
    if (!input.trim()) return

    setLoading(true)
    setError(null)

    const result = await llm.enhance({
      input: input.trim(),
      style: style || null,
      mode: enhanceMode,
      include_negative: includeNegative,
      include_motion: includeMotion,
      model: enhanceModel,
      ...(nsfwMode && { nsfw_intensity: nsfwIntensity }),
    })

    if (result) {
      // Prepend camera motion prefix to generated prompt if selected
      const motionPrefix = getCameraMotionPrefix(cameraMotion)
      if (motionPrefix && result.prompt) {
        result.prompt = motionPrefix + result.prompt
      }
      setResult(result)
      if (DEBUG) console.log('✨ Prompt result:', result)
    } else if (llm.error) {
      setError(llm.error)
    }
    setLoading(false)
  }

  // Quick template-based generation (no API needed)
  const handleQuickGenerate = () => {
    if (!input.trim()) return

    const basePrompt = input.trim()
    const allStyles = [...STYLE_PRESETS, ...NSFW_STYLE_PRESETS]
    const stylePreset = allStyles.find(s => s.id === style)
    const styleKeywords = stylePreset ? `, ${stylePreset.keywords}` : ''

    // Prepend camera motion prefix if selected
    const motionPrefix = getCameraMotionPrefix(cameraMotion)
    const enhancedPrompt = `${motionPrefix}${basePrompt}${styleKeywords}, masterpiece, best quality, highly detailed`

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
      {/* NSFW Toggle — only visible when NSFW is enabled globally */}
      {nsfwEnabled && (
        <div className="grok-card">
          <div className="grok-card-header">
            <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Flame size={16} color={nsfwMode ? '#ef4444' : undefined} />
              NSFW Mode
            </div>
            <button
              onClick={() => handleNsfwToggle(!nsfwMode)}
              style={{
                padding: '6px 14px', borderRadius: '20px',   cursor: 'pointer',
                fontSize: '12px', fontWeight: 600, transition: 'all 0.2s',
                background: nsfwMode ? '#ef4444' : 'var(--bg-secondary, #1a1a1a)',
                color: nsfwMode ? '#fff' : 'var(--text-muted, #888)',
                border: `1px solid ${nsfwMode ? '#ef4444' : 'var(--border-color, #444)'}`,
              }}
            >
              {nsfwMode ? '🔥 ON' : 'OFF'}
            </button>
          </div>
          {nsfwMode && (
            <div style={{ marginTop: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <span style={{ fontSize: '12px', color: 'var(--text-muted, #888)' }}>Intensity</span>
                <span style={{ fontSize: '13px', fontWeight: 600, color: nsfwIntensity >= 4 ? '#ef4444' : nsfwIntensity >= 2 ? '#f59e0b' : 'var(--text-color)' }}>
                  {NSFW_INTENSITY_LABELS[nsfwIntensity]}
                </span>
              </div>
              <input
                type="range" min={1} max={5} value={nsfwIntensity}
                onChange={(e) => setNsfwIntensity(parseInt(e.target.value))}
                style={{ width: '100%', accentColor: '#ef4444' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-muted, #666)', marginTop: '2px' }}>
                <span>Suggestive</span><span>Extreme</span>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Sparkles size={16} />
            Input Idea
          </div>
          <ResetDefaultsButton onReset={handleResetDefaults} />
        </div>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={nsfwMode
            ? 'Describe your NSFW scene idea... (e.g., "woman in lingerie, bedroom")'
            : 'Describe your image or video idea... (e.g., \'a cat wearing sunglasses\')'}
          rows={3}
          className="form-textarea"
        />
      </div>

      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Style Preset</div>
        </div>
        <div className="style-grid">
          {activeStyles.map((s) => (
            <button
              key={s.id}
              className={`style-btn ${style === s.id ? 'active' : ''}`}
              onClick={() => setStyle(style === s.id ? '' : s.id)}
              style={nsfwMode && style === s.id ? { background: '#ef4444', borderColor: '#ef4444' } : {}}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Options</div>
        </div>
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

      {/* Camera Motion Selector — available for both SFW and NSFW */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Camera Motion</div>
        </div>
        <CameraMotionSelector
          value={cameraMotion}
          onChange={setCameraMotion}
        />
        <p style={{ margin: '6px 0 0', fontSize: '11px', color: 'var(--text-muted, #666)' }}>
          Selected motion is prepended to the generated prompt — ready for T2V / I2V.
        </p>
      </div>

      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">AI Model</div>
        </div>
        <select
          value={enhanceModel}
          onChange={(e) => setEnhanceModel(e.target.value)}
          className="form-select"
        >
          {activeModels.map((m) => (
            <option key={m.id} value={m.id}>{m.label}</option>
          ))}
        </select>
        <p style={{ margin: '6px 0 0', fontSize: '12px', color: 'var(--text-muted, #888)' }}>
          {activeModels.find((m) => m.id === enhanceModel)?.description}
        </p>
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
              <LLMQueueIndicator queuePosition={llm.queuePosition} isLoading={llm.isLoading} />
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
