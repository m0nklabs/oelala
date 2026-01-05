/**
 * Site Footer
 * Compact legal links bar
 */

import React from 'react'

export default function Footer({ onShowLegal }) {
  return (
    <footer
      style={{
        padding: '8px 16px',
        borderTop: '1px solid rgba(255,255,255,0.05)',
        background: 'transparent',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        gap: '8px',
        fontSize: '11px',
        color: '#4b5563',
      }}
    >
      <span>© {new Date().getFullYear()} oelala.xyz</span>
      <span style={{ opacity: 0.3 }}>•</span>
      <button onClick={() => onShowLegal?.('privacy')} style={linkStyle}>Privacy</button>
      <span style={{ opacity: 0.3 }}>•</span>
      <button onClick={() => onShowLegal?.('terms')} style={linkStyle}>Terms</button>
      <span style={{ opacity: 0.3 }}>•</span>
      <button onClick={() => onShowLegal?.('dmca')} style={linkStyle}>DMCA</button>
    </footer>
  )
}

const linkStyle = {
  background: 'transparent',
  border: 'none',
  color: '#4b5563',
  fontSize: '11px',
  cursor: 'pointer',
  padding: 0,
}
linkStyle[':hover'] = { color: '#6b7280' }
