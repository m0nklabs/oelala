import React, { useState } from 'react'
import { X, Wand2, Image as ImageIcon, FileText, Settings2 } from 'lucide-react'
import './MediaImportModal.css'

/**
 * Modal that lets the user choose which fields to import from a ComfyUI metadata parse result.
 *
 * Props:
 *   item          {object}  – media item (for display: filename, type)
 *   parsedData    {object}  – result of parseComfyWorkflow(): { positive, negative, steps, cfg, sampler, scheduler, seed }
 *   availableFields {string[]} – which fields to offer for this tool, e.g. ['image','positive','negative','steps','cfg']
 *   onApply       (selected) => void   – called with the filtered parsedData the user wants
 *   onClose       () => void
 */
export default function MediaImportModal({ item, parsedData, availableFields, onApply, onClose }) {
  const [checked, setChecked] = useState(() => {
    // Default: check everything that has a value
    const defaults = {}
    for (const f of availableFields) {
      defaults[f] = parsedData?.[f] !== undefined || f === 'image'
    }
    return defaults
  })

  if (!parsedData && !availableFields?.includes('image')) return null

  const hasAnyData = availableFields.some(f => f === 'image' || parsedData?.[f] !== undefined)
  if (!hasAnyData) return null

  const toggle = (field) => setChecked(prev => ({ ...prev, [field]: !prev[field] }))

  const handleApply = () => {
    const selected = {}
    for (const [field, on] of Object.entries(checked)) {
      if (!on) continue
      if (field === 'image') {
        selected.image = true  // caller already has the image blob
      } else if (parsedData?.[field] !== undefined) {
        selected[field] = parsedData[field]
      }
    }
    onApply(selected)
    onClose()
  }

  const FIELD_META = {
    image:     { icon: <ImageIcon size={14} />, label: 'Image',           group: 'source' },
    positive:  { icon: <FileText  size={14} />, label: 'Positive prompt', group: 'prompts' },
    negative:  { icon: <FileText  size={14} />, label: 'Negative prompt', group: 'prompts' },
    steps:     { icon: <Settings2 size={14} />, label: 'Steps',           group: 'advanced' },
    cfg:       { icon: <Settings2 size={14} />, label: 'Prompt Strength / CFG', group: 'advanced' },
    sampler:   { icon: <Settings2 size={14} />, label: 'Sampler',         group: 'advanced' },
    scheduler: { icon: <Settings2 size={14} />, label: 'Scheduler',       group: 'advanced' },
    seed:      { icon: <Settings2 size={14} />, label: 'Seed',            group: 'advanced' },
    loras:     { icon: <Wand2     size={14} />, label: 'LoRA Stack',      group: 'loras' },
  }

  const groups = ['source', 'prompts', 'advanced', 'loras']
  const groupLabel = { source: 'Source', prompts: 'Prompts', advanced: 'Advanced settings', loras: 'LoRA Stack' }

  return (
    <div className="media-import-overlay" onClick={onClose}>
      <div className="media-import-modal" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="media-import-header">
          <div className="media-import-title">
            <Wand2 size={18} />
            <span>Import from previous generation</span>
          </div>
          <button className="media-import-close" onClick={onClose}><X size={18} /></button>
        </div>

        {/* Source label */}
        {item?.filename && (
          <p className="media-import-source">
            From: <span>{item.filename}</span>
          </p>
        )}

        {/* Field groups */}
        <div className="media-import-fields">
          {groups.map(group => {
            const fields = availableFields.filter(f => FIELD_META[f]?.group === group)
            if (!fields.length) return null
            const anyVisible = fields.some(f => f === 'image' || parsedData?.[f] !== undefined)
            if (!anyVisible) return null
            return (
              <div key={group} className="media-import-group">
                <div className="media-import-group-label">{groupLabel[group]}</div>
                {fields.map(field => {
                  const meta = FIELD_META[field] || {}
                  const isVideo = item?.type === 'video'
                  const value = field === 'image'
                    ? isVideo ? '(use start image from video)' : '(use this image as input)'
                    : parsedData?.[field]
                  if (value === undefined) return null
                  // Special rendering for LoRA array
                  if (field === 'loras' && Array.isArray(value)) {
                    return (
                      <label key={field} className={`media-import-row ${checked[field] ? 'checked' : ''}`}>
                        <input
                          type="checkbox"
                          checked={!!checked[field]}
                          onChange={() => toggle(field)}
                        />
                        <span className="media-import-field-icon">{meta.icon}</span>
                        <span className="media-import-field-label">{meta.label}</span>
                        <span className="media-import-field-value" style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                          {value.map((lora, li) => (
                            <span key={li} style={{ fontSize: '0.75rem' }}>
                              {lora.high || lora.low} @ {lora.strength}
                            </span>
                          ))}
                        </span>
                      </label>
                    )
                  }
                  return (
                    <label key={field} className={`media-import-row ${checked[field] ? 'checked' : ''}`}>
                      <input
                        type="checkbox"
                        checked={!!checked[field]}
                        onChange={() => toggle(field)}
                      />
                      <span className="media-import-field-icon">{meta.icon}</span>
                      <span className="media-import-field-label">{meta.label}</span>
                      <span className="media-import-field-value">
                        {typeof value === 'string' && value.length > 80
                          ? value.slice(0, 80) + '…'
                          : String(value)}
                      </span>
                    </label>
                  )
                })}
              </div>
            )
          })}
        </div>

        {/* Footer */}
        <div className="media-import-footer">
          <button className="media-import-btn cancel" onClick={onClose}>Cancel</button>
          <button
            className="media-import-btn apply"
            onClick={handleApply}
            disabled={!Object.values(checked).some(Boolean)}
          >
            Apply to tool
          </button>
        </div>
      </div>
    </div>
  )
}
