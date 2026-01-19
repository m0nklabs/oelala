import React, { useState, useEffect } from 'react'
import { HardDrive, AlertTriangle, ArrowUpCircle } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import { BACKEND_BASE } from '../config'

/**
 * Storage quota display component for user dropdown.
 * Shows storage usage bar, warning for >80%, and upgrade CTA for >95%.
 */
export default function StorageQuota() {
  const { user, token } = useAuth()
  const [quota, setQuota] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!user || !token) {
      setLoading(false)
      return
    }

    async function fetchQuota() {
      try {
        const res = await fetch(`${BACKEND_BASE}/user/storage-quota`, {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        })

        if (!res.ok) {
          throw new Error('Failed to fetch quota')
        }

        const data = await res.json()
        if (data.success) {
          setQuota(data.data)
        } else {
          setError(data.error || 'Unknown error')
        }
      } catch (err) {
        console.error('Failed to fetch storage quota:', err)
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }

    fetchQuota()
  }, [user, token])

  if (loading) {
    return (
      <div style={{ padding: '12px 14px', borderBottom: '1px solid var(--border-color)' }}>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Loading storage info...
        </div>
      </div>
    )
  }

  if (error || !quota) {
    return null // Don't show anything if quota fetch failed
  }

  const { used_percent, human_used, human_limit, warning, upgrade_needed, tier } = quota

  // Progress bar color based on usage
  const getProgressColor = () => {
    if (upgrade_needed) return '#ef4444' // Red
    if (warning) return '#f59e0b' // Orange
    return '#10b981' // Green
  }

  return (
    <div
      style={{
        padding: '12px 14px',
        borderBottom: '1px solid var(--border-color)',
        background: warning ? 'rgba(245,158,11,0.05)' : 'transparent',
      }}
    >
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: 8,
      }}>
        <span style={{
          fontSize: '0.75rem',
          color: 'var(--text-muted)',
          display: 'flex',
          alignItems: 'center',
          gap: 4,
        }}>
          <HardDrive size={12} />
          Storage ({tier.toUpperCase()})
        </span>
        <span style={{
          fontSize: '0.7rem',
          color: warning ? '#f59e0b' : 'var(--text-muted)',
        }}>
          {human_used} / {human_limit}
        </span>
      </div>

      {/* Progress Bar */}
      <div style={{
        width: '100%',
        height: 6,
        background: 'var(--bg-input)',
        borderRadius: 3,
        overflow: 'hidden',
        marginBottom: 6,
      }}>
        <div style={{
          width: `${Math.min(used_percent, 100)}%`,
          height: '100%',
          background: getProgressColor(),
          borderRadius: 3,
          transition: 'width 0.3s ease',
        }} />
      </div>

      {/* Usage Percentage */}
      <div style={{
        fontSize: '0.7rem',
        color: 'var(--text-muted)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <span>{used_percent.toFixed(1)}% used</span>

        {warning && !upgrade_needed && (
          <span style={{
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            color: '#f59e0b',
            fontSize: '0.65rem',
          }}>
            <AlertTriangle size={10} />
            Running low
          </span>
        )}
      </div>

      {/* Upgrade CTA when near limit */}
      {upgrade_needed && (
        <button
          onClick={() => {
            // TODO: Open upgrade modal or navigate to pricing
            console.log('Open upgrade modal')
          }}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 6,
            width: '100%',
            padding: '8px 12px',
            marginTop: 8,
            border: 'none',
            borderRadius: 6,
            background: 'linear-gradient(135deg, #f59e0b, #d97706)',
            color: 'white',
            fontSize: '0.75rem',
            fontWeight: 500,
            cursor: 'pointer',
            transition: 'opacity 0.15s',
          }}
          onMouseEnter={(e) => e.target.style.opacity = '0.9'}
          onMouseLeave={(e) => e.target.style.opacity = '1'}
        >
          <ArrowUpCircle size={14} />
          Upgrade for more storage
        </button>
      )}
    </div>
  )
}
