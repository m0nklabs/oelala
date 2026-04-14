import React, { useState, useEffect } from 'react'
import { useAuth } from '../../contexts/AuthContext'
import { apiFetch } from '../../api'
import { HardDrive, RefreshCw, AlertCircle, CheckCircle2, Database } from 'lucide-react'

export default function AdminStorageNodesTab() {
  const { isAdmin } = useAuth()
  const [storageHealth, setStorageHealth] = useState(null)
  const [buckets, setBuckets] = useState([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState(null)

  const fetchStorageHealth = async () => {
    try {
      setRefreshing(true)
      const response = await apiFetch('/api/admin/system/health')
      if (response.ok) {
        const data = await response.json()
        const storage = data.services?.storage || null
        setStorageHealth(storage)

        // Extract bucket info if available
        if (storage?.buckets) {
          setBuckets(storage.buckets)
        }
        setError(null)
      } else {
        setError('Failed to fetch storage health')
      }
    } catch (err) {
      console.error('Error fetching storage health:', err)
      setError(err.message)
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }

  useEffect(() => {
    if (isAdmin) {
      fetchStorageHealth()
    }
  }, [isAdmin])

  // Auto-refresh every 30 seconds
  useEffect(() => {
    if (!isAdmin) return
    const interval = setInterval(fetchStorageHealth, 30000)
    return () => clearInterval(interval)
  }, [isAdmin])

  if (!isAdmin) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center' }}>
        <AlertCircle size={48} style={{ color: '#ef4444', marginBottom: '1rem' }} />
        <h3>Access Denied</h3>
      </div>
    )
  }

  const isOnline = storageHealth?.status === 'online'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ fontSize: '1.2rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-primary)', margin: 0 }}>
          <HardDrive size={20} style={{ color: 'var(--accent-color)' }} />
          MinIO Storage
        </h2>
        <button
          onClick={fetchStorageHealth}
          disabled={refreshing}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            padding: '0.4rem 0.8rem',
            background: 'var(--surface-color-light)',
            border: '1px solid var(--border-color)',
            borderRadius: '6px',
            color: 'var(--text-secondary)',
            fontSize: '0.85rem',
            cursor: refreshing ? 'not-allowed' : 'pointer',
            opacity: refreshing ? 0.7 : 1,
            transition: 'all 0.2s',
          }}
        >
          <RefreshCw size={14} className={refreshing ? 'spin' : ''} />
          {refreshing ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {error && (
        <div style={{ padding: '0.75rem', background: 'rgba(239, 68, 68, 0.1)', color: '#ef4444', borderRadius: '8px', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {loading ? (
        <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          Loading storage status...
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {/* MinIO Status Card */}
          <div style={{
            background: 'var(--surface-color)',
            border: '1px solid var(--border-color)',
            borderRadius: '12px',
            padding: '1rem',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
              <div style={{
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                background: isOnline ? '#10b981' : '#ef4444',
                boxShadow: isOnline ? '0 0 8px rgba(16, 185, 129, 0.4)' : 'none'
              }} />
              <div>
                <div style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '0.95rem' }}>
                  MinIO Server
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.1rem' }}>
                  {isOnline ? 'Online' : 'Offline'} — Port {storageHealth?.port || 9000}
                  {storageHealth?.backend ? ` (${storageHealth.backend})` : ''}
                </div>
              </div>
            </div>

            {/* Bucket Info */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem' }}>
              {['oelala-generated', 'oelala-comfyui', 'oelala-avatars', 'oelala-users'].map((bucket) => (
                <div
                  key={bucket}
                  style={{
                    padding: '0.75rem',
                    background: 'var(--surface-color-light)',
                    borderRadius: '8px',
                    border: '1px solid var(--border-color)',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.25rem' }}>
                    <Database size={14} style={{ color: 'var(--accent-color)' }} />
                    <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                      {bucket}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.75rem', color: isOnline ? '#10b981' : 'var(--text-muted)' }}>
                    {isOnline ? (
                      <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                        <CheckCircle2 size={12} /> Active
                      </span>
                    ) : (
                      'Unavailable'
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
