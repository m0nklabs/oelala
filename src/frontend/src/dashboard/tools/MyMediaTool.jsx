import React, { useCallback, useEffect, useState, useRef } from 'react'
import { RefreshCw, Download, X, ChevronLeft, ChevronRight } from 'lucide-react'
import { BACKEND_BASE } from '../../config'

export default function MyMediaTool({ filter = 'all' }) {
  const [mediaList, setMediaList] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [stats, setStats] = useState({ videos: 0, images: 0 })
  const [selectedIndex, setSelectedIndex] = useState(null)
  const containerRef = useRef(null)

  const fetchMedia = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const res = await fetch(`${BACKEND_BASE}/list-comfyui-media?type=${filter}`)
      if (!res.ok) throw new Error('Failed to fetch media')
      const data = await res.json()
      setMediaList(data.media || [])
      setStats({ videos: data.videos || 0, images: data.images || 0 })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [filter])

  useEffect(() => {
    fetchMedia()
  }, [fetchMedia])

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (selectedIndex === null) return
      if (e.key === 'Escape') setSelectedIndex(null)
      if (e.key === 'ArrowLeft') setSelectedIndex(prev => prev > 0 ? prev - 1 : mediaList.length - 1)
      if (e.key === 'ArrowRight') setSelectedIndex(prev => prev < mediaList.length - 1 ? prev + 1 : 0)
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedIndex, mediaList.length])

  const handleDownload = (item, e) => {
    e?.stopPropagation()
    const link = document.createElement('a')
    link.href = `${BACKEND_BASE}${item.url}`
    link.download = item.filename
    link.click()
  }

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  }

  const selectedItem = selectedIndex !== null ? mediaList[selectedIndex] : null

  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100%',
      backgroundColor: 'var(--bg-primary)'
    }}>
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .media-item {
          position: relative;
          border-radius: 12px;
          overflow: hidden;
          cursor: pointer;
          transition: transform 0.2s ease, box-shadow 0.2s ease;
          background: #0a0a0a;
        }
        .media-item:hover {
          transform: scale(1.03);
          box-shadow: 0 8px 24px rgba(0,0,0,0.4);
          z-index: 10;
        }
        .media-item:hover .media-overlay {
          opacity: 1;
        }
        .media-item video {
          width: 100%;
          height: 100%;
          object-fit: cover;
          display: block;
        }
        .media-item img {
          width: 100%;
          height: 100%;
          object-fit: cover;
          display: block;
        }
        .media-overlay {
          position: absolute;
          bottom: 0;
          left: 0;
          right: 0;
          padding: 12px;
          background: linear-gradient(transparent, rgba(0,0,0,0.85));
          opacity: 0;
          transition: opacity 0.2s ease;
          display: flex;
          justify-content: space-between;
          align-items: flex-end;
        }
        .media-filename {
          font-size: 0.75rem;
          color: #fff;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          max-width: 70%;
        }
        .media-size {
          font-size: 0.7rem;
          color: rgba(255,255,255,0.7);
        }
        .download-btn {
          padding: 6px;
          border-radius: 6px;
          background: rgba(255,255,255,0.15);
          border: none;
          color: #fff;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: background 0.2s;
        }
        .download-btn:hover {
          background: rgba(255,255,255,0.3);
        }
        .lightbox-overlay {
          position: fixed;
          inset: 0;
          background: rgba(0,0,0,0.95);
          z-index: 1000;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .lightbox-content {
          max-width: 90vw;
          max-height: 85vh;
          position: relative;
        }
        .lightbox-content video,
        .lightbox-content img {
          max-width: 90vw;
          max-height: 85vh;
          border-radius: 12px;
        }
        .lightbox-nav {
          position: absolute;
          top: 50%;
          transform: translateY(-50%);
          width: 48px;
          height: 48px;
          border-radius: 50%;
          background: rgba(255,255,255,0.1);
          border: none;
          color: #fff;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: background 0.2s;
        }
        .lightbox-nav:hover {
          background: rgba(255,255,255,0.2);
        }
        .lightbox-close {
          position: absolute;
          top: 20px;
          right: 20px;
          width: 40px;
          height: 40px;
          border-radius: 50%;
          background: rgba(255,255,255,0.1);
          border: none;
          color: #fff;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1001;
        }
        .lightbox-close:hover {
          background: rgba(255,255,255,0.2);
        }
        .lightbox-info {
          position: absolute;
          bottom: 20px;
          left: 50%;
          transform: translateX(-50%);
          background: rgba(0,0,0,0.7);
          padding: 12px 20px;
          border-radius: 8px;
          display: flex;
          gap: 16px;
          align-items: center;
        }
        .masonry-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
          gap: 12px;
          padding: 16px;
        }
        @media (min-width: 1200px) {
          .masonry-grid {
            grid-template-columns: repeat(5, 1fr);
          }
        }
        @media (max-width: 800px) {
          .masonry-grid {
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 8px;
          }
        }
      `}</style>

      {/* Minimal Header */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        padding: '12px 16px',
        borderBottom: '1px solid var(--border-color)',
        backgroundColor: 'var(--bg-secondary)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
            {filter === 'all' ? 'All Media' : filter === 'video' ? 'Videos' : 'Images'}
          </span>
          <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            🎬 {stats.videos} videos • 🖼️ {stats.images} images
          </span>
        </div>
        <button
          onClick={fetchMedia}
          disabled={loading}
          style={{
            padding: '8px',
            borderRadius: '6px',
            border: 'none',
            background: 'transparent',
            color: 'var(--text-muted)',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center'
          }}
          title="Refresh"
        >
          <RefreshCw size={18} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
        </button>
      </div>

      {/* Error */}
      {error && (
        <div style={{ 
          padding: '12px 16px', 
          backgroundColor: 'rgba(239, 68, 68, 0.1)', 
          color: '#ef4444',
          textAlign: 'center'
        }}>
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div style={{ 
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-muted)' 
        }}>
          <RefreshCw size={40} style={{ animation: 'spin 1s linear infinite', marginBottom: '16px' }} />
          <div>Loading media...</div>
        </div>
      )}

      {/* Empty State */}
      {!loading && mediaList.length === 0 && (
        <div style={{ 
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-muted)' 
        }}>
          <div style={{ fontSize: '4rem', marginBottom: '16px', opacity: 0.5 }}>📁</div>
          <div style={{ fontSize: '1.2rem', marginBottom: '8px' }}>No {filter === 'all' ? 'media' : filter + 's'} yet</div>
          <div style={{ fontSize: '0.9rem', opacity: 0.7 }}>Generated content will appear here</div>
        </div>
      )}

      {/* Media Grid - Grok Imagine Style */}
      {!loading && mediaList.length > 0 && (
        <div 
          ref={containerRef}
          className="masonry-grid"
          style={{ 
            flex: 1, 
            overflowY: 'auto',
            overflowX: 'hidden'
          }}
        >
          {mediaList.map((item, idx) => (
            <div
              key={idx}
              className="media-item"
              onClick={() => setSelectedIndex(idx)}
              style={{
                aspectRatio: '3/4',
                height: '320px'
              }}
            >
              {item.type === 'video' ? (
                <video
                  src={`${BACKEND_BASE}${item.url}`}
                  autoPlay
                  loop
                  muted
                  playsInline
                  preload="metadata"
                />
              ) : (
                <img
                  src={`${BACKEND_BASE}${item.url}`}
                  alt={item.filename}
                  loading="lazy"
                />
              )}
              <div className="media-overlay">
                <div>
                  <div className="media-filename">{item.filename}</div>
                  <div className="media-size">{formatSize(item.size)}</div>
                </div>
                <button 
                  className="download-btn"
                  onClick={(e) => handleDownload(item, e)}
                  title="Download"
                >
                  <Download size={16} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Lightbox Modal */}
      {selectedItem && (
        <div className="lightbox-overlay" onClick={() => setSelectedIndex(null)}>
          <button className="lightbox-close" onClick={() => setSelectedIndex(null)}>
            <X size={24} />
          </button>
          
          {/* Navigation */}
          <button 
            className="lightbox-nav" 
            style={{ left: '20px' }}
            onClick={(e) => {
              e.stopPropagation()
              setSelectedIndex(prev => prev > 0 ? prev - 1 : mediaList.length - 1)
            }}
          >
            <ChevronLeft size={28} />
          </button>
          
          <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
            {selectedItem.type === 'video' ? (
              <video
                src={`${BACKEND_BASE}${selectedItem.url}`}
                autoPlay
                loop
                controls
                style={{ borderRadius: '12px' }}
              />
            ) : (
              <img
                src={`${BACKEND_BASE}${selectedItem.url}`}
                alt={selectedItem.filename}
                style={{ borderRadius: '12px' }}
              />
            )}
          </div>
          
          <button 
            className="lightbox-nav" 
            style={{ right: '20px' }}
            onClick={(e) => {
              e.stopPropagation()
              setSelectedIndex(prev => prev < mediaList.length - 1 ? prev + 1 : 0)
            }}
          >
            <ChevronRight size={28} />
          </button>

          {/* Info bar */}
          <div className="lightbox-info">
            <span style={{ color: '#fff', fontWeight: 500 }}>{selectedItem.filename}</span>
            <span style={{ color: 'rgba(255,255,255,0.6)' }}>{formatSize(selectedItem.size)}</span>
            <span style={{ color: 'rgba(255,255,255,0.5)' }}>{selectedIndex + 1} / {mediaList.length}</span>
            <button 
              className="download-btn"
              onClick={(e) => handleDownload(selectedItem, e)}
              title="Download"
            >
              <Download size={16} />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
