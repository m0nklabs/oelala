import React, { useEffect, useState, useRef, useCallback } from 'react'
import { Terminal, X, Maximize2, Minimize2, Wifi, WifiOff } from 'lucide-react'
import { BACKEND_BASE } from '../config'

// Convert HTTP URL to WebSocket URL
const getWsUrl = () => {
  // BACKEND_BASE is like 'http://192.168.1.2:7998'
  const wsProtocol = BACKEND_BASE.startsWith('https') ? 'wss:' : 'ws:'
  const hostPart = BACKEND_BASE.replace(/^https?:\/\//, '')
  return `${wsProtocol}//${hostPart}/ws/logs`
}

export default function LogViewer() {
  const [logs, setLogs] = useState([])
  const [isOpen, setIsOpen] = useState(true)
  const [isExpanded, setIsExpanded] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const bottomRef = useRef(null)
  const wsRef = useRef(null)
  const reconnectTimeoutRef = useRef(null)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(getWsUrl())
    wsRef.current = ws

    ws.onopen = () => {
      setIsConnected(true)
      console.log('📡 Log WebSocket connected')
    }

    ws.onmessage = (event) => {
      try {
        const log = JSON.parse(event.data)
        setLogs(prev => {
          const newLogs = [...prev, log]
          // Keep last 500 logs to prevent memory bloat
          return newLogs.slice(-500)
        })
      } catch (e) {
        console.error('Failed to parse log', e)
      }
    }

    ws.onclose = () => {
      setIsConnected(false)
      console.log('📡 Log WebSocket disconnected')
      // Reconnect after 3 seconds
      reconnectTimeoutRef.current = setTimeout(() => {
        if (isOpen) connect()
      }, 3000)
    }

    ws.onerror = (err) => {
      console.error('WebSocket error', err)
      ws.close()
    }
  }, [isOpen])

  useEffect(() => {
    if (isOpen) {
      connect()
    } else {
      wsRef.current?.close()
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
    }

    return () => {
      wsRef.current?.close()
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
    }
  }, [isOpen, connect])

  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs])

  if (!isOpen) {
    return (
      <button 
        onClick={() => setIsOpen(true)}
        style={{
          position: 'fixed',
          bottom: '20px',
          right: '20px',
          backgroundColor: '#1a1a1a',
          border: '1px solid #333',
          borderRadius: '50%',
          width: '48px',
          height: '48px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          zIndex: 100,
          boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
        }}
      >
        <Terminal size={20} color="#a3a3a3" />
      </button>
    )
  }

  return (
    <div style={{
      position: 'fixed',
      bottom: '20px',
      right: '20px',
      width: isExpanded ? '800px' : '400px',
      height: isExpanded ? '600px' : '300px',
      backgroundColor: '#0a0a0a',
      border: '1px solid #333',
      borderRadius: '8px',
      display: 'flex',
      flexDirection: 'column',
      zIndex: 100,
      boxShadow: '0 10px 30px rgba(0,0,0,0.8)',
      transition: 'all 0.2s ease'
    }}>
      <div style={{
        padding: '8px 12px',
        borderBottom: '1px solid #333',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        backgroundColor: '#121212',
        borderTopLeftRadius: '8px',
        borderTopRightRadius: '8px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8rem', fontWeight: 600, color: '#a3a3a3' }}>
          <Terminal size={14} />
          <span>Server Logs</span>
          {isConnected ? (
            <Wifi size={12} color="#22c55e" title="Connected" />
          ) : (
            <WifiOff size={12} color="#ef4444" title="Disconnected" />
          )}
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button 
            onClick={() => setIsExpanded(!isExpanded)}
            style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: '#666' }}
          >
            {isExpanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
          </button>
          <button 
            onClick={() => setIsOpen(false)}
            style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: '#666' }}
          >
            <X size={14} />
          </button>
        </div>
      </div>
      
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '12px',
        fontFamily: 'monospace',
        fontSize: '0.75rem',
        color: '#d4d4d4',
        lineHeight: '1.4'
      }}>
        {logs.map((log, i) => (
          <div key={i} style={{ marginBottom: '4px', display: 'flex', gap: '8px' }}>
            <span style={{ color: '#525252', flexShrink: 0 }}>{log.timestamp?.split('T')[1]?.split('.')[0] || ''}</span>
            <span style={{ 
              color: log.level === 'ERROR' ? '#ef4444' : 
                     log.level === 'WARNING' ? '#eab308' : '#a3a3a3' 
            }}>
              {log.message}
            </span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
