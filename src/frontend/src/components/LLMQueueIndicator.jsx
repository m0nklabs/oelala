import React from 'react'

/**
 * Small inline indicator showing LLM queue position.
 * Shows "Queue #N" when waiting, "Processing..." when active.
 */
export default function LLMQueueIndicator({ queuePosition, isLoading }) {
  if (!isLoading || queuePosition === null || queuePosition === undefined) return null

  if (queuePosition === -1) {
    return (
      <span style={{
        fontSize: '11px',
        color: '#34d399',
        fontWeight: 500,
        marginLeft: '6px',
        animation: 'pulse 1.5s infinite',
      }}>
        Processing...
      </span>
    )
  }

  if (queuePosition === 0) {
    return (
      <span style={{
        fontSize: '11px',
        color: '#fbbf24',
        fontWeight: 500,
        marginLeft: '6px',
      }}>
        Next up...
      </span>
    )
  }

  return (
    <span style={{
      fontSize: '11px',
      color: '#a78bfa',
      fontWeight: 500,
      marginLeft: '6px',
    }}>
      Queue #{queuePosition + 1}
    </span>
  )
}
