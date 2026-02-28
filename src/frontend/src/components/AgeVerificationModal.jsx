/**
 * Age Verification Modal
 * Shown when a user tries to enable NSFW content without prior age verification.
 * Stores verification in localStorage; cleared on logout.
 */

import React, { useState } from 'react'
import { ShieldCheck, AlertTriangle, X } from 'lucide-react'
import './AgeVerificationModal.css'

export default function AgeVerificationModal({ onConfirm, onCancel }) {
  const [checked, setChecked] = useState(false)

  return (
    <div className="age-modal-overlay" onClick={onCancel}>
      <div className="age-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="age-modal-header">
          <div className="age-modal-icon">
            <AlertTriangle size={32} color="#f59e0b" />
          </div>
          <button className="age-modal-close" onClick={onCancel} aria-label="Cancel">
            <X size={18} />
          </button>
        </div>

        {/* Title */}
        <h2 className="age-modal-title">🔞 Adult Content</h2>
        <p className="age-modal-subtitle">
          You are about to enable NSFW content. This platform may contain
          explicit adult material intended for mature audiences only.
        </p>

        {/* Warning box */}
        <div className="age-modal-warning">
          <ul>
            <li>Content may include nudity and explicit imagery</li>
            <li>Access is strictly for adults aged 18 and over</li>
            <li>By continuing you confirm you are legally allowed to view such content in your jurisdiction</li>
          </ul>
        </div>

        {/* Checkbox */}
        <label className="age-modal-checkbox-label">
          <input
            type="checkbox"
            checked={checked}
            onChange={(e) => setChecked(e.target.checked)}
            className="age-modal-checkbox"
          />
          <span>
            I confirm that I am <strong>18 years of age or older</strong> and I
            consent to viewing adult content.
          </span>
        </label>

        {/* Actions */}
        <div className="age-modal-actions">
          <button className="age-modal-btn age-modal-btn-cancel" onClick={onCancel}>
            Cancel
          </button>
          <button
            className="age-modal-btn age-modal-btn-confirm"
            onClick={onConfirm}
            disabled={!checked}
          >
            <ShieldCheck size={16} />
            Confirm & Enable NSFW
          </button>
        </div>
      </div>
    </div>
  )
}
