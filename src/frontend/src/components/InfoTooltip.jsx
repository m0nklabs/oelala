import React, { useState, useRef, useEffect } from 'react'
import { HelpCircle } from 'lucide-react'

/**
 * Reusable info tooltip component ("wolkje" / cloud popup).
 * Shows a (?) icon that displays an info popup on hover.
 *
 * @param {string} text - Plain text tooltip content
 * @param {React.ReactNode} children - Rich JSX content (overrides text)
 * @param {number} size - Icon size (default: 14)
 * @param {number} width - Tooltip width in px (default: 280)
 * @param {'top'|'bottom'} position - Tooltip position relative to icon (default: 'top')
 * @param {object} style - Additional style for the wrapper span
 */
export default function InfoTooltip({ text, children, size = 14, width = 280, position = 'top', style = {} }) {
  const [visible, setVisible] = useState(false)
  const [flipped, setFlipped] = useState(false)
  const wrapRef = useRef(null)
  const tipRef = useRef(null)

  useEffect(() => {
    if (visible && tipRef.current) {
      const rect = tipRef.current.getBoundingClientRect()
      // Flip to bottom if tooltip overflows top of viewport
      if (position === 'top' && rect.top < 8) {
        setFlipped(true)
      } else if (position === 'bottom' && rect.bottom > window.innerHeight - 8) {
        setFlipped(true)
      } else {
        setFlipped(false)
      }
    }
  }, [visible, position])

  const isTop = (position === 'top' && !flipped) || (position === 'bottom' && flipped)

  const tooltipStyle = {
    position: 'absolute',
    ...(isTop
      ? { bottom: 'calc(100% + 10px)' }
      : { top: 'calc(100% + 10px)' }),
    left: '50%',
    transform: 'translateX(-50%)',
    width: `${width}px`,
    padding: '10px 14px',
    borderRadius: '10px',
    backgroundColor: 'var(--bg-tertiary, #1e1e2e)',
    border: '1px solid var(--border-color, #333)',
    boxShadow: '0 4px 16px rgba(0,0,0,0.35)',
    fontSize: '0.78rem',
    lineHeight: '1.5',
    color: 'var(--text-secondary, #ccc)',
    opacity: visible ? 1 : 0,
    visibility: visible ? 'visible' : 'hidden',
    transition: 'opacity 0.2s, visibility 0.2s',
    zIndex: 100,
    pointerEvents: 'none',
  }

  const arrowStyle = {
    position: 'absolute',
    ...(isTop
      ? { bottom: '-6px', borderRight: '1px solid var(--border-color, #333)', borderBottom: '1px solid var(--border-color, #333)' }
      : { top: '-6px', borderLeft: '1px solid var(--border-color, #333)', borderTop: '1px solid var(--border-color, #333)' }),
    left: '50%',
    transform: 'translateX(-50%) rotate(45deg)',
    width: '10px',
    height: '10px',
    backgroundColor: 'var(--bg-tertiary, #1e1e2e)',
  }

  return (
    <span
      ref={wrapRef}
      style={{ position: 'relative', display: 'inline-flex', cursor: 'help', ...style }}
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => { setVisible(false); setFlipped(false) }}
    >
      <HelpCircle size={size} style={{ opacity: 0.5 }} />
      <div ref={tipRef} style={tooltipStyle}>
        <div style={arrowStyle} />
        {children || text}
      </div>
    </span>
  )
}
