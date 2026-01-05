/**
 * Site Footer
 * Compact legal links bar
 */

import React from 'react'

export default function Footer({ onShowLegal }) {
  return (
    <footer
      style={{
        padding: '4px 12px',
        borderTop: '1px solid rgba(255,255,255,0.03)',
        background: 'transparent',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        gap: '6px',
        fontSize: '9px',
        color: '#3b4555',
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
  color: '#3b4555',
  fontSize: '9px',
  cursor: 'pointer',
  padding: 0,
}
linkStyle[':hover'] = { color: '#6b7280' }
