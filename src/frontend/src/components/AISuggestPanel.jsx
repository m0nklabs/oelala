import React, { useState, useCallback } from 'react'
import { Wand2, Loader2, Check, X, ChevronDown, ChevronUp, AlertCircle } from 'lucide-react'
import { apiFetch } from '../api'
import { DEBUG } from '../config'

/**
 * AISuggestPanel — LLM-powered settings optimizer for I2V / T2V tools.
 *
 * Sends current form state + available LoRA registry to the /ai-suggest
 * endpoint, displays suggestions as toggleable checkboxes, and applies
 * accepted suggestions back to the parent form via onApply callback.
 */
export default function AISuggestPanel({
  tool = 'i2v',           // 'i2v' | 't2v'
  prompt = '',
  negativePrompt = '',
  modelMode = '',
  resolution = '',
  steps = 6,
  cfg = 3.0,
  fps = 16,
  duration = 5,
  loras = [],             // [{high, low, strength}]
  availableLoras = {},     // {high_noise: [], low_noise: [], general: []}
  onApply,                 // ({prompt, negativePrompt, loras, steps, cfg, fps, resolution}) => void
}) {
  const [loading, setLoading] = useState(false)
  const [suggestions, setSuggestions] = useState(null)
  const [error, setError] = useState('')
  const [expanded, setExpanded] = useState(true)

  // Track which suggestions are checked (by id)
  const [checked, setChecked] = useState({})

  const toggleCheck = useCallback((id) => {
    setChecked(prev => ({ ...prev, [id]: !prev[id] }))
  }, [])

  const handleAnalyze = useCallback(async () => {
    if (!prompt.trim()) {
      setError('Write a prompt first before analyzing')
      return
    }

    setLoading(true)
    setError('')
    setSuggestions(null)

    try {
      // Build lora payload — flatten high/low to filenames
      const loraPayload = loras
        .filter(l => l.high || l.low)
        .map(l => ({
          filename: l.high || l.low || '',
          strength: l.strength ?? 1.0,
        }))

      const body = {
        prompt,
        negative_prompt: negativePrompt,
        tool,
        model_mode: modelMode,
        resolution,
        steps,
        cfg,
        fps,
        duration,
        loras: loraPayload,
      }

      if (DEBUG) console.log('🤖 AI Suggest: analyzing settings', body)

      const resp = await apiFetch('/ai-suggest', {
        method: 'POST',
        body: JSON.stringify(body),
      })

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(err.detail || `Server error: ${resp.status}`)
      }

      const data = await resp.json()
      if (DEBUG) console.log('🤖 AI Suggest: received', data.suggestions?.length, 'suggestions')

      setSuggestions(data.suggestions || [])
      // Default all to checked
      const initialChecked = {}
      for (const s of (data.suggestions || [])) {
        initialChecked[s.id] = s.checked !== false
      }
      setChecked(initialChecked)

    } catch (err) {
      console.error('🤖 AI Suggest error:', err)
      setError(err.message || 'Failed to get suggestions')
    } finally {
      setLoading(false)
    }
  }, [prompt, negativePrompt, tool, modelMode, resolution, steps, cfg, fps, duration, loras])

  const handleApply = useCallback(() => {
    if (!suggestions || !onApply) return

    const selected = suggestions.filter(s => checked[s.id])
    if (selected.length === 0) return

    // Build changes object
    const changes = {
      promptAppend: [],
      promptReplace: [],
      negativeAppend: [],
      lorasToAdd: [],
      loraStrengthChanges: {},
      settingChanges: {},
    }

    for (const s of selected) {
      const a = s.apply
      switch (s.type) {
        case 'prompt_add':
          if (a.text) changes.promptAppend.push(a.text)
          break
        case 'lora_trigger':
          // Only add trigger words for LoRAs that are already active (not being added in same batch)
          if (a.text) {
            const loraBeingAdded = selected.some(
              other => other.type === 'lora_add' && other.apply?.filename === a.lora_filename
            )
            if (!loraBeingAdded) {
              changes.promptAppend.push(a.text)
            }
            // If the LoRA is being added, trigger words are included in lora_add.trigger_words
          }
          break
        case 'prompt_replace':
          if (a.find && a.replace !== undefined) {
            changes.promptReplace.push({ find: a.find, replace: a.replace })
          }
          break
        case 'negative_add':
          if (a.text) changes.negativeAppend.push(a.text)
          break
        case 'lora_add':
          if (a.filename) {
            changes.lorasToAdd.push({
              filename: a.filename,
              strength: a.strength ?? 1.0,
              noise_level: a.noise_level || '',
              trigger_words: a.trigger_words || [],
            })
          }
          break
        case 'lora_strength':
          if (a.filename && a.new_strength !== undefined) {
            changes.loraStrengthChanges[a.filename] = a.new_strength
          }
          break
        case 'setting_change':
          if (a.setting && a.value !== undefined) {
            changes.settingChanges[a.setting] = a.value
          }
          break
        default:
          break
      }
    }

    if (DEBUG) console.log('🤖 AI Suggest: applying changes', changes)
    onApply(changes)
    setSuggestions(null)
    setChecked({})
  }, [suggestions, checked, onApply])

  const checkedCount = suggestions
    ? suggestions.filter(s => checked[s.id]).length
    : 0

  const priorityColor = {
    high: '#ef4444',
    medium: '#f59e0b',
    low: '#6b7280',
  }

  const typeIcon = {
    prompt_add: '✏️',
    prompt_replace: '🔄',
    negative_add: '🚫',
    lora_add: '➕',
    lora_strength: '🎚️',
    lora_trigger: '🏷️',
    setting_change: '⚙️',
  }

  return (
    <div style={{
      borderRadius: '10px',
      border: '1px solid rgba(139, 92, 246, 0.3)',
      background: 'rgba(139, 92, 246, 0.05)',
      overflow: 'hidden',
      marginBottom: '12px',
    }}>
      {/* Header / Trigger Button */}
      {!suggestions ? (
        <button
          onClick={handleAnalyze}
          disabled={loading || !prompt.trim()}
          style={{
            width: '100%',
            padding: '12px 16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
            background: loading
              ? 'rgba(139, 92, 246, 0.15)'
              : 'linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(168, 85, 247, 0.2))',
            border: 'none',
            color: loading ? '#a78bfa' : '#c4b5fd',
            fontSize: '0.9rem',
            fontWeight: 600,
            cursor: loading ? 'wait' : (!prompt.trim() ? 'not-allowed' : 'pointer'),
            opacity: !prompt.trim() && !loading ? 0.5 : 1,
            transition: 'all 0.2s',
          }}
        >
          {loading ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              Analyzing your settings...
            </>
          ) : (
            <>
              <Wand2 size={16} />
              AI Suggest — Optimize Settings
            </>
          )}
        </button>
      ) : (
        <>
          {/* Suggestions Header */}
          <div
            onClick={() => setExpanded(!expanded)}
            style={{
              padding: '10px 16px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              cursor: 'pointer',
              background: 'rgba(139, 92, 246, 0.1)',
              borderBottom: expanded ? '1px solid rgba(139, 92, 246, 0.2)' : 'none',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Wand2 size={14} style={{ color: '#a78bfa' }} />
              <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#c4b5fd' }}>
                {suggestions.length} suggestion{suggestions.length !== 1 ? 's' : ''}
              </span>
              <span style={{ fontSize: '0.8rem', color: '#8b5cf6' }}>
                ({checkedCount} selected)
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <button
                onClick={(e) => { e.stopPropagation(); setSuggestions(null); setChecked({}); }}
                style={{
                  padding: '4px 8px',
                  background: 'rgba(239, 68, 68, 0.2)',
                  border: '1px solid rgba(239, 68, 68, 0.3)',
                  borderRadius: '4px',
                  color: '#f87171',
                  fontSize: '0.75rem',
                  cursor: 'pointer',
                }}
              >
                Dismiss
              </button>
              {expanded ? <ChevronUp size={14} color="#8b5cf6" /> : <ChevronDown size={14} color="#8b5cf6" />}
            </div>
          </div>

          {/* Suggestion List */}
          {expanded && (
            <div style={{ padding: '8px 12px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {suggestions.map((s) => (
                <label
                  key={s.id}
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '10px',
                    padding: '8px 10px',
                    borderRadius: '8px',
                    background: checked[s.id] ? 'rgba(139, 92, 246, 0.08)' : 'transparent',
                    border: `1px solid ${checked[s.id] ? 'rgba(139, 92, 246, 0.25)' : 'rgba(255,255,255,0.06)'}`,
                    cursor: 'pointer',
                    transition: 'all 0.15s',
                  }}
                >
                  <input
                    type="checkbox"
                    checked={!!checked[s.id]}
                    onChange={() => toggleCheck(s.id)}
                    style={{ marginTop: '3px', accentColor: '#8b5cf6', flexShrink: 0 }}
                  />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '2px' }}>
                      <span style={{ fontSize: '0.85rem' }}>{typeIcon[s.type] || '💡'}</span>
                      <span style={{
                        fontSize: '0.85rem',
                        fontWeight: 600,
                        color: checked[s.id] ? '#e2e8f0' : '#94a3b8',
                      }}>
                        {s.title}
                      </span>
                      <span style={{
                        fontSize: '0.65rem',
                        padding: '1px 6px',
                        borderRadius: '3px',
                        background: `${priorityColor[s.priority] || '#6b7280'}22`,
                        color: priorityColor[s.priority] || '#6b7280',
                        fontWeight: 600,
                        textTransform: 'uppercase',
                      }}>
                        {s.priority}
                      </span>
                    </div>
                    <div style={{
                      fontSize: '0.78rem',
                      color: '#64748b',
                      lineHeight: 1.4,
                    }}>
                      {s.description}
                    </div>
                  </div>
                </label>
              ))}

              {/* Apply Button */}
              {suggestions.length > 0 && (
                <button
                  onClick={handleApply}
                  disabled={checkedCount === 0}
                  style={{
                    marginTop: '4px',
                    padding: '10px 16px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '8px',
                    background: checkedCount > 0
                      ? 'linear-gradient(135deg, #7c3aed, #a855f7)'
                      : '#333',
                    border: 'none',
                    borderRadius: '8px',
                    color: '#fff',
                    fontSize: '0.9rem',
                    fontWeight: 600,
                    cursor: checkedCount > 0 ? 'pointer' : 'not-allowed',
                    opacity: checkedCount > 0 ? 1 : 0.5,
                  }}
                >
                  <Check size={16} />
                  Apply {checkedCount} Suggestion{checkedCount !== 1 ? 's' : ''}
                </button>
              )}

              {suggestions.length === 0 && (
                <div style={{
                  padding: '16px',
                  textAlign: 'center',
                  color: '#64748b',
                  fontSize: '0.85rem',
                }}>
                  No suggestions — your settings look good! 👍
                </div>
              )}
            </div>
          )}
        </>
      )}

      {/* Error */}
      {error && (
        <div style={{
          padding: '8px 12px',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          fontSize: '0.8rem',
          color: '#f87171',
          background: 'rgba(239, 68, 68, 0.1)',
        }}>
          <AlertCircle size={14} />
          {error}
        </div>
      )}
    </div>
  )
}
