import React, { useState, useEffect, useCallback } from 'react'
import { useAuth } from '../../contexts/AuthContext'
import { BACKEND_BASE } from '../../config'
import { apiFetch } from '../../api'
import {
  Flag, Shield, Eye, EyeOff, Check, X, AlertTriangle,
  ChevronDown, ChevronUp, Clock, User, FileText, RefreshCw
} from 'lucide-react'

const REASON_LABELS = {
  inappropriate: 'Inappropriate',
  copyright: 'Copyright',
  spam: 'Spam',
  harassment: 'Harassment',
  underage: 'Underage',
  other: 'Other',
}

const REASON_COLORS = {
  inappropriate: '#f59e0b',
  copyright: '#3b82f6',
  spam: '#6b7280',
  harassment: '#ef4444',
  underage: '#dc2626',
  other: '#8b5cf6',
}

function StatCard({ icon, label, value, color }) {
  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border-color)',
      borderRadius: '12px',
      padding: '1rem',
      display: 'flex',
      alignItems: 'center',
      gap: '0.75rem',
    }}>
      <div style={{
        width: '40px',
        height: '40px',
        borderRadius: '10px',
        background: `${color}20`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: color,
      }}>
        {icon}
      </div>
      <div>
        <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          {value}
        </div>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{label}</div>
      </div>
    </div>
  )
}

export default function AdminModerationTab() {
  const { session } = useAuth()
  const [stats, setStats] = useState(null)
  const [queue, setQueue] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [filterStatus, setFilterStatus] = useState('pending')
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const [expandedItem, setExpandedItem] = useState(null)
  const [actionLoading, setActionLoading] = useState(null) // media_id being acted on
  const [selectedItems, setSelectedItems] = useState(new Set())
  const [bulkActionLoading, setBulkActionLoading] = useState(false)
  const [showAuditLog, setShowAuditLog] = useState(false)
  const [auditLog, setAuditLog] = useState([])
  const [auditLoading, setAuditLoading] = useState(false)

  const authHeaders = useCallback(() => ({
    Authorization: `Bearer ${session?.access_token}`,
    'Content-Type': 'application/json',
  }), [session])

  // Fetch moderation stats
  const fetchStats = useCallback(async () => {
    try {
      const resp = await apiFetch('/api/admin/moderation/stats')
      if (resp.ok) {
        setStats(await resp.json())
      }
    } catch (err) {
      console.error('Failed to fetch moderation stats:', err)
    }
  }, [authHeaders])

  // Fetch moderation queue
  const fetchQueue = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const resp = await apiFetch(
        `/api/admin/moderation/queue?status=${filterStatus}&page=${page}&per_page=20`
      )
      if (!resp.ok) throw new Error('Failed to load queue')
      const data = await resp.json()
      setQueue(data.items || [])
      setTotal(data.total || 0)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [filterStatus, page, authHeaders])

  useEffect(() => {
    fetchStats()
    fetchQueue()
  }, [fetchStats, fetchQueue])

  // Take moderation action
  const handleAction = async (mediaId, action, reason = null, reportId = null) => {
    setActionLoading(mediaId)
    try {
      const resp = await apiFetch(`/api/admin/moderation/${mediaId}/action`, {
        method: 'POST',
        body: JSON.stringify({ action, reason, report_id: reportId }),
      })
      if (!resp.ok) throw new Error('Action failed')
      // Refresh
      await Promise.all([fetchQueue(), fetchStats()])
    } catch (err) {
      console.error('Moderation action failed:', err)
    } finally {
      setActionLoading(null)
    }
  }

  // Bulk action
  const handleBulkAction = async (action) => {
    if (selectedItems.size === 0) return
    setBulkActionLoading(true)
    try {
      const resp = await apiFetch('/api/admin/moderation/bulk-action', {
        method: 'POST',
        body: JSON.stringify({
          media_ids: Array.from(selectedItems),
          action,
          reason: `Bulk ${action} by admin`,
        }),
      })
      if (!resp.ok) throw new Error('Bulk action failed')
      setSelectedItems(new Set())
      await Promise.all([fetchQueue(), fetchStats()])
    } catch (err) {
      console.error('Bulk action failed:', err)
    } finally {
      setBulkActionLoading(false)
    }
  }

  // Fetch audit log
  const fetchAuditLog = async () => {
    setAuditLoading(true)
    try {
      const resp = await apiFetch('/api/admin/moderation/log/actions?per_page=50')
      if (resp.ok) {
        const data = await resp.json()
        setAuditLog(data.items || [])
      }
    } catch (err) {
      console.error('Failed to fetch audit log:', err)
    } finally {
      setAuditLoading(false)
    }
  }

  const toggleSelect = (mediaId) => {
    setSelectedItems(prev => {
      const next = new Set(prev)
      if (next.has(mediaId)) next.delete(mediaId)
      else next.add(mediaId)
      return next
    })
  }

  const formatDate = (dateStr) => {
    if (!dateStr) return '—'
    const d = new Date(dateStr)
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
  }

  return (
    <div>
      {/* Stats */}
      {stats && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
          gap: '0.75rem',
          marginBottom: '1.5rem',
        }}>
          <StatCard icon={<AlertTriangle size={20} />} label="Pending Reports" value={stats.pending_reports} color="#ef4444" />
          <StatCard icon={<Check size={20} />} label="Reviewed Today" value={stats.reviewed_today} color="#10b981" />
          <StatCard icon={<EyeOff size={20} />} label="Hidden Content" value={stats.total_hidden} color="#f59e0b" />
          <StatCard icon={<X size={20} />} label="Rejected" value={stats.total_rejected} color="#dc2626" />
          <StatCard icon={<Flag size={20} />} label="Total Reports" value={stats.total_reports} color="#6366f1" />
        </div>
      )}

      {/* Toolbar */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '1rem',
        flexWrap: 'wrap',
        gap: '0.5rem',
      }}>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          {/* Filter by report status */}
          <select
            value={filterStatus}
            onChange={e => { setFilterStatus(e.target.value); setPage(1) }}
            style={{
              padding: '0.5rem 0.75rem',
              background: 'var(--bg-card)',
              border: '1px solid var(--border-color)',
              borderRadius: '8px',
              color: 'var(--text-primary)',
              fontSize: '0.9rem',
            }}
          >
            <option value="pending">Pending</option>
            <option value="reviewed">Reviewed</option>
            <option value="dismissed">Dismissed</option>
          </select>

          <button
            onClick={() => { fetchQueue(); fetchStats() }}
            style={{
              padding: '0.5rem',
              background: 'var(--bg-card)',
              border: '1px solid var(--border-color)',
              borderRadius: '8px',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
            }}
            title="Refresh"
          >
            <RefreshCw size={16} />
          </button>

          <button
            onClick={() => { setShowAuditLog(!showAuditLog); if (!showAuditLog) fetchAuditLog() }}
            style={{
              padding: '0.5rem 0.75rem',
              background: showAuditLog ? 'var(--accent-color)' : 'var(--bg-card)',
              border: '1px solid var(--border-color)',
              borderRadius: '8px',
              color: showAuditLog ? 'white' : 'var(--text-secondary)',
              cursor: 'pointer',
              fontSize: '0.85rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
            }}
          >
            <FileText size={14} />
            Audit Log
          </button>
        </div>

        {/* Bulk actions */}
        {selectedItems.size > 0 && (
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              {selectedItems.size} selected
            </span>
            <button
              onClick={() => handleBulkAction('approve')}
              disabled={bulkActionLoading}
              style={{
                padding: '0.4rem 0.75rem',
                background: '#10b981',
                border: 'none',
                borderRadius: '6px',
                color: 'white',
                cursor: 'pointer',
                fontSize: '0.85rem',
              }}
            >
              Approve All
            </button>
            <button
              onClick={() => handleBulkAction('hide')}
              disabled={bulkActionLoading}
              style={{
                padding: '0.4rem 0.75rem',
                background: '#f59e0b',
                border: 'none',
                borderRadius: '6px',
                color: 'white',
                cursor: 'pointer',
                fontSize: '0.85rem',
              }}
            >
              Hide All
            </button>
            <button
              onClick={() => handleBulkAction('reject')}
              disabled={bulkActionLoading}
              style={{
                padding: '0.4rem 0.75rem',
                background: '#ef4444',
                border: 'none',
                borderRadius: '6px',
                color: 'white',
                cursor: 'pointer',
                fontSize: '0.85rem',
              }}
            >
              Reject All
            </button>
          </div>
        )}
      </div>

      {/* Audit Log View */}
      {showAuditLog && (
        <div style={{
          background: 'var(--bg-card)',
          border: '1px solid var(--border-color)',
          borderRadius: '12px',
          padding: '1rem',
          marginBottom: '1.5rem',
          maxHeight: '300px',
          overflowY: 'auto',
        }}>
          <h3 style={{ margin: '0 0 0.75rem', fontSize: '1rem', color: 'var(--text-primary)' }}>
            Moderation Audit Log
          </h3>
          {auditLoading ? (
            <p style={{ color: 'var(--text-muted)', textAlign: 'center' }}>Loading...</p>
          ) : auditLog.length === 0 ? (
            <p style={{ color: 'var(--text-muted)', textAlign: 'center' }}>No moderation actions yet</p>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                  <th style={{ textAlign: 'left', padding: '0.5rem', color: 'var(--text-muted)' }}>Date</th>
                  <th style={{ textAlign: 'left', padding: '0.5rem', color: 'var(--text-muted)' }}>Action</th>
                  <th style={{ textAlign: 'left', padding: '0.5rem', color: 'var(--text-muted)' }}>Media ID</th>
                  <th style={{ textAlign: 'left', padding: '0.5rem', color: 'var(--text-muted)' }}>Reason</th>
                </tr>
              </thead>
              <tbody>
                {auditLog.map(a => (
                  <tr key={a.id} style={{ borderBottom: '1px solid var(--border-color)' }}>
                    <td style={{ padding: '0.5rem', color: 'var(--text-secondary)' }}>{formatDate(a.created_at)}</td>
                    <td style={{ padding: '0.5rem' }}>
                      <span style={{
                        padding: '2px 8px',
                        borderRadius: '4px',
                        fontSize: '0.8rem',
                        fontWeight: 600,
                        background: a.action === 'approve' ? '#10b98120' : a.action === 'reject' ? '#ef444420' : '#f59e0b20',
                        color: a.action === 'approve' ? '#10b981' : a.action === 'reject' ? '#ef4444' : '#f59e0b',
                      }}>
                        {a.action}
                      </span>
                    </td>
                    <td style={{ padding: '0.5rem', color: 'var(--text-muted)', fontFamily: 'monospace', fontSize: '0.75rem' }}>
                      {a.media_id?.slice(0, 8)}...
                    </td>
                    <td style={{ padding: '0.5rem', color: 'var(--text-secondary)' }}>{a.reason || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Queue */}
      {error && (
        <div style={{
          padding: '1rem',
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: '8px',
          color: '#ef4444',
          marginBottom: '1rem',
        }}>
          {error}
        </div>
      )}

      {loading ? (
        <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
          Loading moderation queue...
        </div>
      ) : queue.length === 0 ? (
        <div style={{
          textAlign: 'center',
          padding: '3rem',
          background: 'var(--bg-card)',
          borderRadius: '12px',
          border: '1px solid var(--border-color)',
        }}>
          <Shield size={48} style={{ color: 'var(--text-muted)', marginBottom: '1rem' }} />
          <p style={{ color: 'var(--text-muted)', fontSize: '1.1rem', margin: 0 }}>
            {filterStatus === 'pending'
              ? 'No pending reports — all clear! 🎉'
              : `No ${filterStatus} reports found`}
          </p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {queue.map(item => (
            <div
              key={item.media_id}
              style={{
                background: 'var(--bg-card)',
                border: `1px solid ${item.moderation_status === 'hidden' ? '#f59e0b40' : 'var(--border-color)'}`,
                borderRadius: '12px',
                overflow: 'hidden',
              }}
            >
              {/* Header */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  padding: '0.75rem 1rem',
                  cursor: 'pointer',
                }}
                onClick={() => setExpandedItem(expandedItem === item.media_id ? null : item.media_id)}
              >
                {/* Checkbox */}
                <input
                  type="checkbox"
                  checked={selectedItems.has(item.media_id)}
                  onChange={(e) => { e.stopPropagation(); toggleSelect(item.media_id) }}
                  style={{ accentColor: 'var(--accent-color)', width: '18px', height: '18px' }}
                />

                {/* Media preview thumbnail */}
                {item.storage_path && (
                  <div style={{
                    width: '48px',
                    height: '48px',
                    borderRadius: '8px',
                    overflow: 'hidden',
                    flexShrink: 0,
                    background: '#1a1a1a',
                  }}>
                    {item.media_type === 'video' ? (
                      <video
                        src={`${BACKEND_BASE}/api/gallery/${item.media_id}/file`}
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                        muted
                      />
                    ) : (
                      <img
                        src={`${BACKEND_BASE}/api/gallery/${item.media_id}/file`}
                        alt=""
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      />
                    )}
                  </div>
                )}

                {/* Info */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    marginBottom: '2px',
                  }}>
                    <span style={{
                      fontWeight: 600,
                      color: 'var(--text-primary)',
                      fontSize: '0.95rem',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}>
                      {item.title}
                    </span>
                    {item.is_nsfw && (
                      <span style={{
                        padding: '1px 6px',
                        background: '#ef444430',
                        color: '#ef4444',
                        borderRadius: '4px',
                        fontSize: '0.7rem',
                        fontWeight: 700,
                      }}>
                        NSFW
                      </span>
                    )}
                    <span style={{
                      padding: '1px 6px',
                      background: item.moderation_status === 'hidden' ? '#f59e0b30' : item.moderation_status === 'rejected' ? '#ef444430' : '#10b98130',
                      color: item.moderation_status === 'hidden' ? '#f59e0b' : item.moderation_status === 'rejected' ? '#ef4444' : '#10b981',
                      borderRadius: '4px',
                      fontSize: '0.7rem',
                      fontWeight: 600,
                    }}>
                      {item.moderation_status}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', gap: '0.75rem' }}>
                    <span>{item.media_type}</span>
                    <span style={{ color: '#ef4444', fontWeight: 600 }}>
                      {item.report_count} report{item.report_count !== 1 ? 's' : ''}
                    </span>
                    <span>{formatDate(item.created_at)}</span>
                  </div>
                </div>

                {/* Quick actions */}
                <div style={{ display: 'flex', gap: '0.4rem' }} onClick={e => e.stopPropagation()}>
                  <button
                    onClick={() => handleAction(item.media_id, 'approve')}
                    disabled={actionLoading === item.media_id}
                    title="Approve — dismiss reports"
                    style={{
                      padding: '6px 10px',
                      background: '#10b981',
                      border: 'none',
                      borderRadius: '6px',
                      color: 'white',
                      cursor: 'pointer',
                      fontSize: '0.8rem',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px',
                      opacity: actionLoading === item.media_id ? 0.5 : 1,
                    }}
                  >
                    <Check size={14} /> OK
                  </button>
                  <button
                    onClick={() => handleAction(item.media_id, 'hide')}
                    disabled={actionLoading === item.media_id}
                    title="Hide — remove from gallery"
                    style={{
                      padding: '6px 10px',
                      background: '#f59e0b',
                      border: 'none',
                      borderRadius: '6px',
                      color: 'white',
                      cursor: 'pointer',
                      fontSize: '0.8rem',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px',
                      opacity: actionLoading === item.media_id ? 0.5 : 1,
                    }}
                  >
                    <EyeOff size={14} /> Hide
                  </button>
                  <button
                    onClick={() => handleAction(item.media_id, 'reject')}
                    disabled={actionLoading === item.media_id}
                    title="Reject — permanently remove"
                    style={{
                      padding: '6px 10px',
                      background: '#ef4444',
                      border: 'none',
                      borderRadius: '6px',
                      color: 'white',
                      cursor: 'pointer',
                      fontSize: '0.8rem',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px',
                      opacity: actionLoading === item.media_id ? 0.5 : 1,
                    }}
                  >
                    <X size={14} /> Reject
                  </button>
                </div>

                {expandedItem === item.media_id ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
              </div>

              {/* Expanded detail */}
              {expandedItem === item.media_id && (
                <div style={{
                  borderTop: '1px solid var(--border-color)',
                  padding: '1rem',
                  background: 'rgba(0,0,0,0.2)',
                }}>
                  {/* Media preview */}
                  <div style={{
                    marginBottom: '1rem',
                    borderRadius: '8px',
                    overflow: 'hidden',
                    maxHeight: '300px',
                    background: '#111',
                    display: 'flex',
                    justifyContent: 'center',
                  }}>
                    {item.media_type === 'video' ? (
                      <video
                        src={`${BACKEND_BASE}/api/gallery/${item.media_id}/file`}
                        controls
                        style={{ maxHeight: '300px', maxWidth: '100%' }}
                      />
                    ) : (
                      <img
                        src={`${BACKEND_BASE}/api/gallery/${item.media_id}/file`}
                        alt={item.title}
                        style={{ maxHeight: '300px', maxWidth: '100%', objectFit: 'contain' }}
                      />
                    )}
                  </div>

                  {/* Reports list */}
                  <h4 style={{ margin: '0 0 0.5rem', color: 'var(--text-primary)', fontSize: '0.95rem' }}>
                    Reports ({item.reports.length})
                  </h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {item.reports.map(r => (
                      <div
                        key={r.id}
                        style={{
                          padding: '0.75rem',
                          background: 'var(--bg-card)',
                          border: '1px solid var(--border-color)',
                          borderRadius: '8px',
                          display: 'flex',
                          alignItems: 'flex-start',
                          gap: '0.75rem',
                        }}
                      >
                        <span style={{
                          padding: '2px 8px',
                          background: `${REASON_COLORS[r.reason] || '#666'}20`,
                          color: REASON_COLORS[r.reason] || '#666',
                          borderRadius: '4px',
                          fontSize: '0.8rem',
                          fontWeight: 600,
                          flexShrink: 0,
                        }}>
                          {REASON_LABELS[r.reason] || r.reason}
                        </span>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          {r.description && (
                            <p style={{ margin: '0 0 4px', color: 'var(--text-primary)', fontSize: '0.9rem' }}>
                              {r.description}
                            </p>
                          )}
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', gap: '0.75rem' }}>
                            <span>
                              <User size={12} style={{ verticalAlign: 'middle', marginRight: '4px' }} />
                              {r.reporter_id?.slice(0, 8)}...
                            </span>
                            <span>
                              <Clock size={12} style={{ verticalAlign: 'middle', marginRight: '4px' }} />
                              {formatDate(r.created_at)}
                            </span>
                          </div>
                        </div>
                        {r.status === 'pending' && (
                          <button
                            onClick={() => handleAction(item.media_id, 'dismiss_report', null, r.id)}
                            disabled={actionLoading === item.media_id}
                            title="Dismiss this report"
                            style={{
                              padding: '4px 8px',
                              background: 'transparent',
                              border: '1px solid var(--border-color)',
                              borderRadius: '4px',
                              color: 'var(--text-muted)',
                              cursor: 'pointer',
                              fontSize: '0.75rem',
                            }}
                          >
                            Dismiss
                          </button>
                        )}
                      </div>
                    ))}
                  </div>

                  {/* Creator info */}
                  <div style={{
                    marginTop: '0.75rem',
                    padding: '0.5rem 0.75rem',
                    background: 'var(--bg-card)',
                    borderRadius: '8px',
                    fontSize: '0.85rem',
                    color: 'var(--text-muted)',
                  }}>
                    <strong>Creator:</strong>{' '}
                    <span style={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                      {item.creator_id?.slice(0, 12)}...
                    </span>
                  </div>
                </div>
              )}
            </div>
          ))}

          {/* Pagination */}
          {total > 20 && (
            <div style={{
              display: 'flex',
              justifyContent: 'center',
              gap: '0.5rem',
              marginTop: '1rem',
            }}>
              <button
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page === 1}
                style={{
                  padding: '0.5rem 1rem',
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '8px',
                  color: 'var(--text-primary)',
                  cursor: page === 1 ? 'not-allowed' : 'pointer',
                  opacity: page === 1 ? 0.5 : 1,
                }}
              >
                Previous
              </button>
              <span style={{ padding: '0.5rem', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                Page {page} of {Math.ceil(total / 20)}
              </span>
              <button
                onClick={() => setPage(p => p + 1)}
                disabled={page >= Math.ceil(total / 20)}
                style={{
                  padding: '0.5rem 1rem',
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '8px',
                  color: 'var(--text-primary)',
                  cursor: page >= Math.ceil(total / 20) ? 'not-allowed' : 'pointer',
                  opacity: page >= Math.ceil(total / 20) ? 0.5 : 1,
                }}
              >
                Next
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
