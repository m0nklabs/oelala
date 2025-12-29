import React, { useMemo, useState } from 'react'
import { BACKEND_BASE } from '../config'
import { useVideoHistory } from './useVideoHistory'
import { Download, ExternalLink, History, Film, X } from 'lucide-react'

export default function OutputPanel({ output, refreshToken, onSelectHistoryVideo }) {
  const [historyOpen, setHistoryOpen] = useState(false)
  const { videos, loading, error } = useVideoHistory(refreshToken)

  const content = useMemo(() => {
    if (!output) {
      return (
        <div className="placeholder-state">
          <div className="placeholder-icon">
            <Film />
          </div>
          <h3>Ready to Create</h3>
          <p className="muted">Configure parameters and click Generate</p>
        </div>
      )
    }

    if (output.kind === 'video') {
      return (
        <div className="media-container">
          <video className="media-preview" controls src={output.url} autoPlay loop />
          <div className="media-info">
            <div className="media-meta">
              {output.filename || 'Generated Video'}
            </div>
            <div className="media-actions">
              {output.url && (
                <a className="icon-btn" href={output.url} download={output.filename || undefined} title="Download">
                  <Download size={18} />
                </a>
              )}
              {output.backendUrl && (
                <a className="icon-btn" href={output.backendUrl} target="_blank" rel="noreferrer" title="Open in new tab">
                  <ExternalLink size={18} />
                </a>
              )}
            </div>
          </div>
        </div>
      )
    }

    if (output.kind === 'image') {
      return (
        <div className="media-container">
          <img 
            className="media-preview" 
            src={output.url} 
            alt="Generated" 
            onError={(e) => {
              console.error('Image load failed:', output.url)
              e.target.style.display = 'none'
              e.target.parentNode.innerHTML += `<div style="padding:20px;color:red">Failed to load image: ${output.url}</div>`
            }}
          />
          <div className="media-info">
            <div className="media-meta">
              {output.filename || 'Generated Image'}
            </div>
            <div className="media-actions">
              {output.url && (
                <a className="icon-btn" href={output.url} download={output.filename || undefined} title="Download">
                  <Download size={18} />
                </a>
              )}
              {output.backendUrl && (
                <a className="icon-btn" href={output.backendUrl} target="_blank" rel="noreferrer" title="Open in new tab">
                  <ExternalLink size={18} />
                </a>
              )}
            </div>
          </div>
        </div>
      )
    }

    if (output.kind === 'lora') {
      return (
        <div className="media-container" style={{ padding: '24px' }}>
          <h3>LoRA Training Complete</h3>
          <div className="media-meta" style={{ marginTop: '16px' }}>
            <p>ID: {output.training_id}</p>
            <p>Path: {output.lora_path}</p>
          </div>
        </div>
      )
    }

    return null
  }, [output])

  return (
    <section className="output-panel">
      <div style={{ position: 'absolute', top: 20, right: 20, zIndex: 10 }}>
        <button 
          className="icon-btn" 
          onClick={() => setHistoryOpen(!historyOpen)}
          title="History"
        >
          <History size={20} />
        </button>
      </div>

      {content}

      {historyOpen && (
        <div className="history">
          <div className="history-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>History</span>
            <button className="icon-btn" onClick={() => setHistoryOpen(false)}>
              <X size={18} />
            </button>
          </div>
          <div className="history-list">
            {loading && <div style={{ padding: 20, textAlign: 'center' }} className="muted">Loading...</div>}
            {error && <div className="error">{error}</div>}
            {!loading && !error && videos.length === 0 && (
              <div style={{ padding: 20, textAlign: 'center' }} className="muted">No history yet</div>
            )}
            {videos.map((vid) => (
              <button
                key={vid.filename}
                className="history-item"
                onClick={() => {
                  onSelectHistoryVideo({
                    kind: 'video',
                    url: `${BACKEND_BASE}/outputs/${vid.filename}`,
                    backendUrl: `${BACKEND_BASE}/outputs/${vid.filename}`,
                    filename: vid.filename,
                  })
                  // Optional: close history on select
                  // setHistoryOpen(false) 
                }}
              >
                <div className="history-item-title">{vid.filename}</div>
                <div className="history-item-sub">
                  {new Date(vid.mtime * 1000).toLocaleString()}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </section>
  )
}
