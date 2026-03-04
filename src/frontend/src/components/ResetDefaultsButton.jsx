/**
 * ResetDefaultsButton — Small icon button to reset tool settings to OOTB defaults.
 *
 * Props:
 *   onReset  — callback that applies defaults to all state (receives defaults object)
 *   label    — optional tooltip text (default: "Reset to defaults")
 */

import React, { useState } from 'react'
import { RotateCcw } from 'lucide-react'

export default function ResetDefaultsButton({ onReset, label = 'Reset to defaults' }) {
  const [confirming, setConfirming] = useState(false)

  const handleClick = () => {
    if (confirming) {
      onReset?.()
      setConfirming(false)
    } else {
      setConfirming(true)
      setTimeout(() => setConfirming(false), 3000) // auto-cancel after 3s
    }
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      title={confirming ? 'Click again to confirm' : label}
      className={`p-1.5 rounded-lg transition-colors ${
        confirming
          ? 'bg-amber-500/20 text-amber-400 hover:bg-amber-500/30'
          : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/5'
      }`}
    >
      <RotateCcw size={14} className={confirming ? 'animate-spin' : ''} />
    </button>
  )
}
