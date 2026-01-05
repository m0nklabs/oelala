/**
 * Insufficient Credits Modal
 * Shown when user tries to generate but doesn't have enough credits
 */

import React from 'react'
import { AlertCircle, Coins, X } from 'lucide-react'

export default function InsufficientCreditsModal({ 
  required, 
  available, 
  packages = [], 
  onClose, 
  onPurchase 
}) {
  const deficit = required - available

  return (
    <>
      {/* Backdrop */}
      <div
        style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0, 0, 0, 0.7)',
          backdropFilter: 'blur(4px)',
          zIndex: 1100,
        }}
        onClick={onClose}
      />

      {/* Modal */}
      <div
        style={{
          position: 'fixed',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '90%',
          maxWidth: 500,
          background: 'var(--bg-card, #1a1a2e)',
          borderRadius: 16,
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
          zIndex: 1101,
          overflow: 'hidden',
        }}
      >
        {/* Header */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '20px 24px',
            borderBottom: '1px solid var(--border-color, #2d2d4a)',
            background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.05))',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div
              style={{
                width: 40,
                height: 40,
                borderRadius: 10,
                background: 'rgba(239, 68, 68, 0.2)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <AlertCircle size={20} color="#ef4444" />
            </div>
            <h2 style={{ margin: 0, fontSize: '1.2rem', color: 'var(--text-primary, white)' }}>
              Insufficient Credits
            </h2>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-muted, #888)',
              cursor: 'pointer',
              padding: 8,
              borderRadius: 8,
            }}
          >
            <X size={20} />
          </button>
        </div>

        {/* Body */}
        <div style={{ padding: '24px' }}>
          {/* Balance Info */}
          <div
            style={{
              padding: '16px',
              background: 'rgba(239, 68, 68, 0.05)',
              borderRadius: 12,
              border: '1px solid rgba(239, 68, 68, 0.2)',
              marginBottom: 20,
            }}
          >
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted, #888)', marginBottom: 8 }}>
              You need <strong style={{ color: '#ef4444' }}>{deficit} more credits</strong> to complete this generation.
            </div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.9rem' }}>
              <div>
                <span style={{ color: 'var(--text-muted, #888)' }}>Required:</span>{' '}
                <strong style={{ color: 'white' }}>{required} credits</strong>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted, #888)' }}>Available:</span>{' '}
                <strong style={{ color: '#a78bfa' }}>{available} credits</strong>
              </div>
            </div>
          </div>

          {/* Suggested Package */}
          {packages.length > 0 && (
            <>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-muted, #888)', marginBottom: 12 }}>
                Recommended package:
              </div>
              {(() => {
                // Find smallest package that covers the deficit
                const suitable = packages
                  .filter(p => p.credits >= deficit)
                  .sort((a, b) => a.credits - b.credits)[0]
                
                const recommended = suitable || packages[0]
                
                return (
                  <div
                    onClick={() => onPurchase(recommended)}
                    style={{
                      padding: '16px',
                      borderRadius: 12,
                      border: '2px solid #7c3aed',
                      background: 'rgba(124, 58, 237, 0.1)',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                      marginBottom: 16,
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = 'rgba(124, 58, 237, 0.15)'
                      e.currentTarget.style.transform = 'scale(1.02)'
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'rgba(124, 58, 237, 0.1)'
                      e.currentTarget.style.transform = 'scale(1)'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                      <div style={{ fontWeight: 600, color: 'white' }}>{recommended.name}</div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <Coins size={16} style={{ color: '#fbbf24' }} />
                        <span style={{ fontSize: '1.2rem', fontWeight: 700, color: '#a78bfa' }}>
                          {recommended.credits.toLocaleString()}
                        </span>
                      </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <div style={{ fontSize: '0.85rem', color: 'var(--text-muted, #888)' }}>
                        {new Intl.NumberFormat('nl-NL', {
                          style: 'currency',
                          currency: recommended.currency || 'EUR',
                        }).format(recommended.price_cents / 100)}
                      </div>
                      <div
                        style={{
                          padding: '4px 12px',
                          borderRadius: 20,
                          background: '#7c3aed',
                          color: 'white',
                          fontSize: '0.75rem',
                          fontWeight: 600,
                        }}
                      >
                        Buy Now →
                      </div>
                    </div>
                  </div>
                )
              })()}
            </>
          )}

          {/* Footer */}
          <div style={{ display: 'flex', gap: 12 }}>
            <button
              onClick={onClose}
              style={{
                flex: 1,
                padding: '12px',
                borderRadius: 8,
                border: '1px solid var(--border-color, #2d2d4a)',
                background: 'var(--bg-input, #252540)',
                color: 'var(--text-primary, white)',
                fontSize: '0.9rem',
                fontWeight: 500,
                cursor: 'pointer',
              }}
            >
              Cancel
            </button>
            <button
              onClick={() => {
                onClose()
                onPurchase()
              }}
              style={{
                flex: 1,
                padding: '12px',
                borderRadius: 8,
                border: 'none',
                background: 'linear-gradient(135deg, #7c3aed, #a855f7)',
                color: 'white',
                fontSize: '0.9rem',
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              View All Packages
            </button>
          </div>
        </div>
      </div>
    </>
  )
}
