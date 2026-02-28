import React from 'react'
import LoraBrowser from '../../components/LoraBrowser'

/**
 * LoRA Browser tool — standalone dashboard tool for browsing all LoRA models.
 * Shows a searchable, filterable grid of all available LoRAs with metadata.
 */
export default function LoRABrowserTool() {
  const handleSelect = (lora) => {
    // In standalone mode, copy the path to clipboard for easy use
    if (navigator.clipboard) {
      navigator.clipboard.writeText(lora.path)
        .then(() => {
          // Could add a toast notification here
          console.log(`Copied LoRA path: ${lora.path}`)
        })
        .catch(() => {})
    }
  }

  return (
    <div className="h-full flex flex-col">
      <LoraBrowser onSelect={handleSelect} mode="browse" />
    </div>
  )
}
