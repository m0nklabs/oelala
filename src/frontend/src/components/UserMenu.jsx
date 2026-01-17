import React, { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { useCredits } from '../contexts/CreditsContext'
import { LogIn, User, LogOut, Loader2, ChevronDown, Coins, Plus, RefreshCw } from 'lucide-react'
import PurchaseCreditsModal from './PurchaseCreditsModal'
import StorageQuota from './StorageQuota'

export default function UserMenu() {
  const { user, loading, signInWithGoogle, signOut, switchAccount } = useAuth()
  const { balance, loading: creditsLoading } = useCredits()
  const [showDropdown, setShowDropdown] = useState(false)
  const [showPurchaseModal, setShowPurchaseModal] = useState(false)

  if (loading) {
    return (
      <div className="user-menu loading">
        <Loader2 size={16} className="spin" />
      </div>
    )
  }

  if (!user) {
    return (
      <button
        className="login-btn"
        onClick={signInWithGoogle}
        title="Sign in with Google"
      >
        <LogIn size={16} />
        <span>Login</span>
      </button>
    )
  }

  return (
    <div className="user-menu" style={{ position: 'relative', display: 'flex', alignItems: 'center', gap: '8px' }}>
      {/* Credits Display */}
      <div
        className="credits-display"
        onClick={() => setShowPurchaseModal(true)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          padding: '6px 10px',
          borderRadius: '6px',
          background: 'linear-gradient(135deg, #7c3aed22, #a855f722)',
          border: '1px solid #7c3aed44',
          color: '#a78bfa',
          fontSize: '0.85rem',
          fontWeight: 600,
          cursor: 'pointer',
          transition: 'all 0.2s ease',
        }}
        title="Click to buy credits"
      >
        <Coins size={14} />
        <span>{creditsLoading ? '...' : balance}</span>
        <Plus size={12} style={{ opacity: 0.7 }} />
      </div>

      {/* User Button */}
      <button
        className="user-info-btn"
        onClick={() => setShowDropdown(!showDropdown)}
        title={user.email}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          padding: '6px 10px',
          borderRadius: '6px',
          border: '1px solid var(--border-color)',
          background: 'var(--bg-input)',
          color: 'var(--text-secondary)',
          fontSize: '0.8rem',
          cursor: 'pointer',
          transition: 'all 0.2s ease',
        }}
      >
        {user.user_metadata?.avatar_url ? (
          <img
            src={user.user_metadata.avatar_url}
            alt="Avatar"
            className="user-avatar"
            style={{ width: 24, height: 24, borderRadius: '50%' }}
          />
        ) : (
          <User size={16} />
        )}
        <span className="user-name" style={{ maxWidth: 100, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {user.user_metadata?.full_name || user.email?.split('@')[0]}
        </span>
        <ChevronDown size={14} style={{ opacity: 0.6 }} />
      </button>

      {showDropdown && (
        <div
          className="user-dropdown"
          style={{
            position: 'absolute',
            top: '100%',
            right: 0,
            marginTop: 4,
            minWidth: 200,
            background: 'var(--bg-panel)',
            border: '1px solid var(--border-color)',
            borderRadius: 8,
            boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
            zIndex: 1000,
            overflow: 'hidden',
          }}
        >
          {/* User Info */}
          <div style={{ padding: '12px 14px', borderBottom: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: '0.85rem', fontWeight: 500, color: 'var(--text-primary)' }}>
              {user.user_metadata?.full_name || 'User'}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 2 }}>
              {user.email}
            </div>
          </div>

          {/* Credits Info */}
          <div style={{ padding: '12px 14px', borderBottom: '1px solid var(--border-color)', background: 'rgba(124,58,237,0.05)' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Credits Balance</span>
              <span style={{ fontSize: '0.95rem', fontWeight: 600, color: '#a78bfa' }}>
                <Coins size={14} style={{ marginRight: 4, verticalAlign: 'middle' }} />
                {balance}
              </span>
            </div>
            <button
              onClick={() => { setShowPurchaseModal(true); setShowDropdown(false); }}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 6,
                width: '100%',
                padding: '8px 12px',
                border: 'none',
                borderRadius: 6,
                background: 'linear-gradient(135deg, #7c3aed, #a855f7)',
                color: 'white',
                fontSize: '0.8rem',
                fontWeight: 500,
                cursor: 'pointer',
                transition: 'opacity 0.15s',
              }}
              onMouseEnter={(e) => e.target.style.opacity = '0.9'}
              onMouseLeave={(e) => e.target.style.opacity = '1'}
            >
              <Plus size={14} />
              Buy Credits
            </button>
          </div>

          {/* Storage Quota */}
          <StorageQuota />

          {/* Switch Account */}
          <button
            onClick={() => { switchAccount('google'); setShowDropdown(false); }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              width: '100%',
              padding: '10px 14px',
              border: 'none',
              borderBottom: '1px solid var(--border-color)',
              background: 'transparent',
              color: 'var(--text-secondary)',
              fontSize: '0.85rem',
              cursor: 'pointer',
              transition: 'background 0.15s',
            }}
            onMouseEnter={(e) => e.target.style.background = 'rgba(124,58,237,0.1)'}
            onMouseLeave={(e) => e.target.style.background = 'transparent'}
          >
            <RefreshCw size={16} />
            Switch account
          </button>

          {/* Sign Out */}
          <button
            onClick={() => { signOut(); setShowDropdown(false); }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              width: '100%',
              padding: '10px 14px',
              border: 'none',
              background: 'transparent',
              color: '#ef4444',
              fontSize: '0.85rem',
              cursor: 'pointer',
              transition: 'background 0.15s',
            }}
            onMouseEnter={(e) => e.target.style.background = 'rgba(239,68,68,0.1)'}
            onMouseLeave={(e) => e.target.style.background = 'transparent'}
          >
            <LogOut size={16} />
            Sign out
          </button>
        </div>
      )}

      {/* Click outside to close dropdown */}
      {showDropdown && (
        <div
          style={{ position: 'fixed', inset: 0, zIndex: 999 }}
          onClick={() => setShowDropdown(false)}
        />
      )}

      {/* Purchase Credits Modal */}
      {showPurchaseModal && (
        <PurchaseCreditsModal onClose={() => setShowPurchaseModal(false)} />
      )}
    </div>
  )
}
