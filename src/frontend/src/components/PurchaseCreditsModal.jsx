/**
 * Purchase Credits Modal
 * Shows available credit packages and handles Stripe checkout
 */

import React, { useState } from 'react'
import { useCredits } from '../contexts/CreditsContext'
import { X, Coins, Sparkles, Check, Loader2, ExternalLink } from 'lucide-react'

const formatPrice = (cents, currency = 'EUR') => {
  return new Intl.NumberFormat('nl-NL', {
    style: 'currency',
    currency: currency,
  }).format(cents / 100)
}

const getBadgeStyle = (badge) => {
  if (badge === 'POPULAR') {
    return {
      background: 'linear-gradient(135deg, #7c3aed, #a855f7)',
      color: 'white',
    }
  }
  if (badge === 'BEST VALUE') {
    return {
      background: 'linear-gradient(135deg, #059669, #10b981)',
      color: 'white',
    }
  }
  return { background: '#374151', color: '#9ca3af' }
}

export default function PurchaseCreditsModal({ onClose }) {
  const { packages, balance, purchaseCredits, error, clearError } = useCredits()
  const [selectedPackage, setSelectedPackage] = useState(null)
  const [purchasing, setPurchasing] = useState(false)

  const handlePurchase = async (pkg) => {
    setSelectedPackage(pkg.id)
    setPurchasing(true)
    clearError()

    const checkoutUrl = await purchaseCredits(pkg.id)
    
    if (checkoutUrl) {
      // Redirect to Stripe Checkout
      window.location.href = checkoutUrl
    } else {
      setPurchasing(false)
      setSelectedPackage(null)
    }
  }

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
          maxWidth: 600,
          maxHeight: '90vh',
          background: 'var(--bg-card, #1a1a2e)',
          borderRadius: 16,
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
          zIndex: 1101,
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
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
            background: 'linear-gradient(135deg, rgba(124,58,237,0.1), rgba(168,85,247,0.05))',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div
              style={{
                width: 40,
                height: 40,
                borderRadius: 10,
                background: 'linear-gradient(135deg, #7c3aed, #a855f7)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Coins size={20} color="white" />
            </div>
            <div>
              <h2 style={{ margin: 0, fontSize: '1.2rem', color: 'var(--text-primary, white)' }}>
                Buy Credits
              </h2>
              <p style={{ margin: 0, fontSize: '0.8rem', color: 'var(--text-muted, #888)' }}>
                Current balance: <strong style={{ color: '#a78bfa' }}>{balance} credits</strong>
              </p>
            </div>
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
              transition: 'background 0.15s',
            }}
          >
            <X size={20} />
          </button>
        </div>

        {/* Error Message */}
        {error && (
          <div
            style={{
              margin: '16px 24px 0',
              padding: '12px 16px',
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: 8,
              color: '#f87171',
              fontSize: '0.85rem',
            }}
          >
            {error}
          </div>
        )}

        {/* Packages Grid */}
        <div
          style={{
            padding: '24px',
            overflowY: 'auto',
            flex: 1,
          }}
        >
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
              gap: 16,
            }}
          >
            {packages.map((pkg) => {
              const isSelected = selectedPackage === pkg.id
              const pricePerCredit = pkg.price_cents / pkg.credits
              const basePrice = 0.05 // €0.05 per credit base
              const savings = Math.round((1 - pricePerCredit / 100 / basePrice) * 100)

              return (
                <div
                  key={pkg.id}
                  onClick={() => !purchasing && handlePurchase(pkg)}
                  style={{
                    position: 'relative',
                    padding: '20px 16px',
                    borderRadius: 12,
                    border: isSelected
                      ? '2px solid #7c3aed'
                      : '1px solid var(--border-color, #2d2d4a)',
                    background: isSelected
                      ? 'rgba(124, 58, 237, 0.1)'
                      : 'var(--bg-input, #252540)',
                    cursor: purchasing ? 'wait' : 'pointer',
                    transition: 'all 0.2s ease',
                    opacity: purchasing && !isSelected ? 0.5 : 1,
                  }}
                >
                  {/* Badge */}
                  {pkg.badge && (
                    <div
                      style={{
                        position: 'absolute',
                        top: -10,
                        right: 12,
                        padding: '4px 10px',
                        borderRadius: 20,
                        fontSize: '0.65rem',
                        fontWeight: 600,
                        textTransform: 'uppercase',
                        letterSpacing: '0.5px',
                        ...getBadgeStyle(pkg.badge),
                      }}
                    >
                      {pkg.badge}
                    </div>
                  )}

                  {/* Package Name */}
                  <div
                    style={{
                      fontSize: '0.9rem',
                      fontWeight: 600,
                      color: 'var(--text-primary, white)',
                      marginBottom: 8,
                    }}
                  >
                    {pkg.name}
                  </div>

                  {/* Credits */}
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                      marginBottom: 12,
                    }}
                  >
                    <Sparkles size={16} style={{ color: '#fbbf24' }} />
                    <span
                      style={{
                        fontSize: '1.5rem',
                        fontWeight: 700,
                        color: '#a78bfa',
                      }}
                    >
                      {pkg.credits.toLocaleString()}
                    </span>
                  </div>

                  {/* Price */}
                  <div
                    style={{
                      fontSize: '1.1rem',
                      fontWeight: 600,
                      color: 'var(--text-primary, white)',
                      marginBottom: 4,
                    }}
                  >
                    {formatPrice(pkg.price_cents, pkg.currency)}
                  </div>

                  {/* Per Credit */}
                  <div
                    style={{
                      fontSize: '0.75rem',
                      color: 'var(--text-muted, #888)',
                    }}
                  >
                    {formatPrice(pricePerCredit, pkg.currency)}/credit
                    {savings > 0 && (
                      <span style={{ color: '#10b981', marginLeft: 6 }}>
                        Save {savings}%
                      </span>
                    )}
                  </div>

                  {/* Loading State */}
                  {isSelected && purchasing && (
                    <div
                      style={{
                        position: 'absolute',
                        inset: 0,
                        background: 'rgba(0,0,0,0.5)',
                        borderRadius: 12,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                      }}
                    >
                      <Loader2 size={24} className="spin" style={{ color: '#a78bfa' }} />
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          {/* Info */}
          <div
            style={{
              marginTop: 24,
              padding: '16px',
              background: 'rgba(124, 58, 237, 0.05)',
              borderRadius: 12,
              border: '1px solid rgba(124, 58, 237, 0.2)',
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: 12,
                fontSize: '0.8rem',
                color: 'var(--text-muted, #888)',
              }}
            >
              <Check size={16} style={{ color: '#10b981', flexShrink: 0, marginTop: 2 }} />
              <div>
                <strong style={{ color: 'var(--text-primary, white)' }}>Credits never expire.</strong>
                <br />
                Use them whenever you want. Secure payment via Stripe.
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div
          style={{
            padding: '16px 24px',
            borderTop: '1px solid var(--border-color, #2d2d4a)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontSize: '0.75rem',
            color: 'var(--text-muted, #888)',
          }}
        >
          <span>Payments processed securely by Stripe</span>
          <a
            href="https://stripe.com"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              color: 'inherit',
              textDecoration: 'none',
            }}
          >
            <ExternalLink size={12} />
            stripe.com
          </a>
        </div>
      </div>

      {/* CSS for spin animation */}
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .spin {
          animation: spin 1s linear infinite;
        }
      `}</style>
    </>
  )
}
