import React, { useState, useEffect } from 'react'
import { useAuth } from '../../contexts/AuthContext'
import { apiFetch } from '../../api'
import { HardDrive, Server, Activity, Users, FileDigit, RefreshCw, AlertCircle, CheckCircle2 } from 'lucide-react'

export default function AdminStorageNodesTab() {
  const { isAdmin } = useAuth()
  const [nodes, setNodes] = useState([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState(null)

  const fetchNodes = async () => {
    try {
      setRefreshing(true)
      const response = await apiFetch('/api/storage-nodes/')
      if (response.ok) {
        const data = await response.json()
        setNodes(data)
        setError(null)
      } else {
        setError('Failed to fetch storage nodes')
      }
    } catch (err) {
      console.error('Error fetching storage nodes:', err)
      setError(err.message)
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }

  useEffect(() => {
    if (isAdmin) {
      fetchNodes()
    }
  }, [isAdmin])

  // Auto-refresh every 30 seconds
  useEffect(() => {
    if (!isAdmin) return
    const interval = setInterval(fetchNodes, 30000)
    return () => clearInterval(interval)
  }, [isAdmin])

  const formatBytes = (bytes) => {
    if (bytes === 0 || !bytes) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const formatTimeAgo = (dateString) => {
    if (!dateString) return 'Never'
    const date = new Date(dateString + 'Z') // assuming UTC
    const now = new Date()
    const seconds = Math.floor((now - date) / 1000)
    
    if (seconds < 60) return `${seconds}s ago`
    const minutes = Math.floor(seconds / 60)
    if (minutes < 60) return `${minutes}m ago`
    const hours = Math.floor(minutes / 60)
    if (hours < 24) return `${hours}h ago`
    const days = Math.floor(hours / 24)
    return `${days}d ago`
  }

  const isOnline = (lastHeartbeat) => {
    if (!lastHeartbeat) return false
    const date = new Date(lastHeartbeat + 'Z')
    const now = new Date()
    const seconds = Math.floor((now - date) / 1000)
    // Consider offline if no heartbeat for more than 2 minutes (120 seconds)
    return seconds < 120
  }

  if (!isAdmin) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center' }}>
        <AlertCircle size={48} style={{ color: '#ef4444', marginBottom: '1rem' }} />
        <h3>Access Denied</h3>
        <p>You need administrator privileges to view this page.</p>
      </div>
    )
  }

  return (
    <div style={{ background: 'var(--surface-color)', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border-color)', marginBottom: '2rem' }}>
      <div style={{ 
        padding: '1.5rem', 
        borderBottom: '1px solid var(--border-color)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-color)' }}>
          <HardDrive size={24} style={{ color: 'var(--accent-color)' }} />
          Storage Node Cluster
        </h2>
        <button
          onClick={fetchNodes}
          disabled={refreshing}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.5rem 1rem',
            background: 'var(--bg-color)',
            border: '1px solid var(--border-color)',
            borderRadius: '6px',
            color: 'var(--text-color)',
            cursor: refreshing ? 'not-allowed' : 'pointer',
            opacity: refreshing ? 0.7 : 1,
            transition: 'all 0.2s'
          }}
        >
          <RefreshCw size={16} className={refreshing ? 'spin' : ''} />
          {refreshing ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {error && (
        <div style={{ padding: '1rem', margin: '1rem', background: 'rgba(239, 68, 68, 0.1)', color: '#ef4444', borderRadius: '8px', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <AlertCircle size={20} />
          {error}
        </div>
      )}

      {loading ? (
        <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          Loading storage nodes...
        </div>
      ) : nodes.length === 0 ? (
        <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          <Server size={48} style={{ opacity: 0.2, marginBottom: '1rem' }} />
          <p>No storage nodes registered.</p>
        </div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ background: 'var(--bg-color)', borderBottom: '2px solid var(--border-color)' }}>
                <th style={{ padding: '1rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Node ID</th>
                <th style={{ padding: '1rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Status</th>
                <th style={{ padding: '1rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Storage Usage</th>
                <th style={{ padding: '1rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Host info</th>
                <th style={{ padding: '1rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Version</th>
                <th style={{ padding: '1rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Last Heartbeat</th>
              </tr>
            </thead>
            <tbody>
              {nodes.map((node) => {
                const online = isOnline(node.last_heartbeat)
                const percentUsed = node.total_space_bytes ? (node.used_space_bytes / node.total_space_bytes) * 100 : 0
                
                return (
                  <tr key={node.id} style={{ borderBottom: '1px solid var(--border-color)' }}>
                    <td style={{ padding: '1rem', fontFamily: 'monospace', color: 'var(--text-color)' }}>
                      {node.node_id}
                    </td>
                    <td style={{ padding: '1rem' }}>
                      <span style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.25rem',
                        padding: '0.25rem 0.5rem',
                        borderRadius: '999px',
                        fontSize: '0.85rem',
                        fontWeight: 500,
                        background: online ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                        color: online ? '#10b981' : '#ef4444'
                      }}>
                        {online ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}
                        {online ? 'Online' : 'Offline'}
                      </span>
                    </td>
                    <td style={{ padding: '1rem' }}>
                      <div style={{ marginBottom: '0.25rem', fontSize: '0.9rem', color: 'var(--text-color)' }}>
                        {formatBytes(node.used_space_bytes)} / {formatBytes(node.total_space_bytes)}
                      </div>
                      <div style={{ width: '100%', height: '6px', background: 'var(--bg-color)', borderRadius: '3px', overflow: 'hidden' }}>
                        <div style={{ 
                          height: '100%', 
                          width: `${Math.min(percentUsed, 100)}%`, 
                          background: percentUsed > 90 ? '#ef4444' : percentUsed > 75 ? '#f59e0b' : '#10b981',
                          borderRadius: '3px'
                        }} />
                      </div>
                    </td>
                    <td style={{ padding: '1rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                      <div>IP: {node.ip_address}</div>
                      <div>Hostname: {node.hostname}</div>
                      <div>OS: {node.os_type} ({node.architecture})</div>
                    </td>
                    <td style={{ padding: '1rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                      {node.version || 'unknown'}
                    </td>
                    <td style={{ padding: '1rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                      {formatTimeAgo(node.last_heartbeat)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}