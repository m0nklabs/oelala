/**
 * Site Footer
 * Contains legal links and copyright notice
 */

import React from 'react'

export default function Footer({ onShowLegal }) {
  return (
    <footer
      style={{
        padding: '16px 24px',
        borderTop: '1px solid #1f2937',
        background: '#0d1117',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '12px',
      }}
    >
      {/* Copyright */}
      <div style={{ color: '#6b7280', fontSize: '13px' }}>
        © {new Date().getFullYear()} oelala.xyz — AI-powered creativity
      </div>
      
      {/* Legal Links */}
      <div style={{ display: 'flex', gap: '20px' }}>
        <button
          onClick={() => onShowLegal?.('privacy')}
          style={linkStyle}
        >
          Privacy
        </button>
        <button
          onClick={() => onShowLegal?.('terms')}
          style={linkStyle}
        >
          Terms of Service
        </button>
        <button
          onClick={() => onShowLegal?.('dmca')}
          style={linkStyle}
        >
          DMCA
        </button>
        <a
          href="mailto:support@oelala.xyz"
          style={linkStyle}
        >
          Contact
        </a>
      </div>
    </footer>
  )
}

const linkStyle = {
  background: 'transparent',
  border: 'none',
  color: '#6b7280',
  fontSize: '13px',
  cursor: 'pointer',
  textDecoration: 'none',
  transition: 'color 0.2s',
}
