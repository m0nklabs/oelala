import React, { useState, useEffect } from 'react'
import { useAuth } from '../../contexts/AuthContext'
import { apiFetch } from '../../api'
import { HardDrive, Server, RefreshCw, AlertCircle, CheckCircle2, ChevronDown, ChevronRight } from 'lucide-react'

export default function AdminStorageNodesTab() {
  const { isAdmin } = useAuth()
  const [nodes, setNodes] = useState([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState(null)
  const [expandedNodes, setExpandedNodes] = useState({})

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
    
    if (seconds < 60) return `${seconds}s`
    const minutes = Math.floor(seconds / 60)
    if (minutes < 60) return `${minutes}m`
    const hours = Math.floor(minutes / 60)
    if (hours < 24) return `${hours}h`
    const days = Math.floor(hours / 24)
    return `${days}d`
  }

  const isOnline = (lastHeartbeat) => {
    if (!lastHeartbeat) return false
    const date = new Date(lastHeartbeat + 'Z')
    const now = new Date()
    const seconds = Math.floor((now - date) / 1000)
    // Consider offline if no heartbeat for more than 2 minutes (120 seconds)
    return seconds < 120
  }

  const toggleNode = (nodeId) => {
    setExpandedNodes(prev => ({
      ...prev,
      [nodeId]: !prev[nodeId]
    }))
  }

  if (!isAdmin) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center' }}>
        <AlertCircle size={48} style={{ color: '#ef4444', marginBottom: '1rem' }} />
        <h3>Access Denied</h3>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ fontSize: '1.2rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-primary)', margin: 0 }}>
          <HardDrive size={20} style={{ color: 'var(--accent-color)' }} />
          Storage Network
        </h2>
        <button
          onClick={fetchNodes}
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
          Loading network status...
        </div>
      ) : nodes.length === 0 ? (
        <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)', background: 'var(--surface-color)', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
          <Server size={32} style={{ opacity: 0.2, margin: '0 auto 1rem auto' }} />
          <p style={{ margin: 0, fontSize: '0.9rem' }}>No storage nodes registered.</p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {nodes.map((node) => {
            const online = isOnline(node.last_heartbeat_at)
            const percentUsed = node.total_bytes ? (node.used_bytes / node.total_bytes) * 100 : 0
            const isExpanded = expandedNodes[node.node_id]
            
            return (
              <div 
                key={node.node_id} 
                style={{ 
                  background: 'var(--surface-color)', 
                  border: '1px solid var(--border-color)', 
                  borderRadius: '12px',
                  overflow: 'hidden'
                }}
              >
                {/* Node Header */}
                <div 
                  onClick={() => toggleNode(node.node_id)}
                  style={{ 
                    padding: '1rem',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    background: isExpanded ? 'var(--surface-color-light)' : 'transparent',
                    transition: 'background 0.2s'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <div style={{ 
                      width: '10px', 
                      height: '10px', 
                      borderRadius: '50%', 
                      background: online ? '#10b981' : '#ef4444',
                      boxShadow: online ? '0 0 8px rgba(16, 185, 129, 0.4)' : 'none'
                    }} />
                    <div>
                      <div style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '0.95rem' }}>
                        {node.hostname || node.node_id.substring(0, 8)}
                      </div>
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.1rem' }}>
                        Last seen: {formatTimeAgo(node.last_heartbeat_at)}
                      </div>
                    </div>
                  </div>
                  
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-secondary)' }}>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '0.8rem', fontWeight: 600, color: percentUsed > 90 ? '#ef4444' : 'var(--text-primary)' }}>
                        {formatBytes(node.used_bytes)}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        / {formatBytes(node.total_bytes)}
                      </div>
                    </div>
                    {isExpanded ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
                  </div>
                </div>

                {/* Progress Bar (Always visible) */}
                <div style={{ width: '100%', height: '4px', background: 'var(--bg-color)', position: 'relative' }}>
                  <div style={{ 
                    position: 'absolute',
                    top: 0, left: 0, bottom: 0,
                    width: `${Math.min(percentUsed, 100)}%`, 
                    background: percentUsed > 90 ? '#ef4444' : percentUsed > 75 ? '#f59e0b' : '#10b981',
                    transition: 'width 0.5s ease-out'
                  }} />
                </div>

                {/* Expanded Details */}
                {isExpanded && (
                  <div style={{ padding: '1rem', borderTop: '1px solid var(--border-color)', fontSize: '0.85rem' }}>
                    <div style={{ display: 'grid', gridTemplateColumns: '80px 1fr', gap: '0.5rem', marginBottom: '0.5rem' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Node ID:</span>
                      <span style={{ fontFamily: 'monospace', color: 'var(--text-secondary)' }}>{node.node_id}</span>
                      
                      <span style={{ color: 'var(--text-muted)' }}>IP Addr:</span>
                      <span style={{ color: 'var(--text-primary)' }}>{node.ip_address}</span>
                      
                      {node.public_url && (
                        <>
                          <span style={{ color: 'var(--text-muted)' }}>Public URL:</span>
                          <span style={{ color: 'var(--text-primary)' }}>
                            <a href={node.public_url} target="_blank" rel="noreferrer" style={{ color: 'var(--primary-color)' }}>
                              {node.public_url.replace('https://', '')}
                            </a>
                          </span>
                        </>
                      )}
                      
                      <span style={{ color: 'var(--text-muted)' }}>OS:</span>
                      <span style={{ color: 'var(--text-primary)' }}>{node.os_type} ({node.architecture})</span>
                      
                      <span style={{ color: 'var(--text-muted)' }}>Version:</span>
                      <span style={{ color: 'var(--text-primary)' }}>{node.version || 'v0.1.0'}</span>
                      
                      <span style={{ color: 'var(--text-muted)' }}>Percent:</span>
                      <span style={{ color: percentUsed > 90 ? '#ef4444' : 'var(--text-primary)', fontWeight: 600 }}>
                        {percentUsed.toFixed(1)}% used
                      </span>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}