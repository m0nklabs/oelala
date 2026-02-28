import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { Filter, Image as ImageIcon, Video, RefreshCw, Eye, Heart, Check, Download, CheckSquare, Square, X as XIcon } from 'lucide-react'
import { apiFetch } from '../api'
import { BACKEND_BASE } from '../config'
import { useAuth } from '../contexts/AuthContext'
import { useNSFW } from '../contexts/NSFWContext'
import MediaDetailModal from '../components/MediaDetailModal'
import { TOOL_IDS } from '../dashboard/nav'

// Lazy loaded media item - only loads src when in viewport
const LazyMediaItem = React.memo(({ item, getMediaUrl, getPreviewUrl, onClick, selected = false, selectMode = false }) => {
  const ref = useRef(null)
  const [isVisible, setIsVisible] = useState(false)
  const [hasLoaded, setHasLoaded] = useState(false)

  useEffect(() => {
    const node = ref.current
    if (!node) return

    // Create observer for THIS item only
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true)
          // Once visible, stop observing - media will load once and stay
          observer.disconnect()
        }
      },
      {
        rootMargin: '100px', // Start loading 100px before entering viewport
        threshold: 0,
      }
    )

    observer.observe(node)
    return () => observer.disconnect()
  }, [])

  const previewUrl = getPreviewUrl(item)
  const mediaUrl = getMediaUrl(item)

  return (
    <div
      ref={ref}
      onClick={onClick}
      style={{
        background: '#2a2a2a',
        borderRadius: '12px',
        overflow: 'hidden',
        cursor: 'pointer',
        transition: 'transform 0.2s, box-shadow 0.2s',
        border: selected ? '2px solid #3b82f6' : '1px solid #333',
        boxShadow: selected ? '0 0 0 2px rgba(59,130,246,0.3)' : 'none',
        position: 'relative',
      }}
      onMouseEnter={(e) => {
        if (!selectMode) {
          e.currentTarget.style.transform = 'translateY(-4px)'
          e.currentTarget.style.boxShadow = '0 8px 24px rgba(0,0,0,0.4)'
        }
      }}
      onMouseLeave={(e) => {
        if (!selectMode) {
          e.currentTarget.style.transform = 'translateY(0)'
          e.currentTarget.style.boxShadow = selected ? '0 0 0 2px rgba(59,130,246,0.3)' : 'none'
        }
      }}
    >
      {/* Selection indicator */}
      {selectMode && (
        <div style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          zIndex: 10,
          background: selected ? '#3b82f6' : 'rgba(0,0,0,0.6)',
          border: `2px solid ${selected ? '#3b82f6' : '#aaa'}`,
          borderRadius: '6px',
          width: '24px',
          height: '24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          {selected && <Check size={14} color="#fff" />}
        </div>
      )}
      {/* Media Preview */}
      <div style={{
        aspectRatio: '9/16',
        background: '#000',
        position: 'relative',
        overflow: 'hidden'
      }}>
        {item.media_type === 'video' ? (
          isVisible ? (
            <video
              src={mediaUrl}
              poster={previewUrl || undefined}
              preload="metadata"
              muted
              playsInline
              onLoadedData={() => setHasLoaded(true)}
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'cover',
                opacity: hasLoaded ? 1 : 0,
                transition: 'opacity 0.3s',
              }}
            />
          ) : (
            <div style={{
              width: '100%',
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: '#111',
              color: '#444',
            }}>
              <Video size={32} />
            </div>
          )
        ) : item.media_type === 'audio' ? (
          <div style={{
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: '#111',
            color: '#aaa',
            fontSize: '12px'
          }}>
            🎵 Audio
          </div>
        ) : isVisible ? (
          <>
            <img
              src={mediaUrl}
              alt={item.title}
              loading="lazy"
              decoding="async"
              onLoad={() => setHasLoaded(true)}
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'cover',
                opacity: hasLoaded ? 1 : 0,
                transition: 'opacity 0.3s',
              }}
            />
            {!hasLoaded && (
              <div style={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: '#111',
                color: '#444',
              }}>
                <ImageIcon size={32} />
              </div>
            )}
          </>
        ) : (
          <div style={{
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: '#111',
            color: '#444',
          }}>
            <ImageIcon size={32} />
          </div>
        )}

        {/* NSFW Badge */}
        {item.is_nsfw && (
          <div style={{
            position: 'absolute',
            top: '8px',
            right: '8px',
            background: 'rgba(239, 68, 68, 0.9)',
            color: '#fff',
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '11px',
            fontWeight: 600,
          }}>
            🔞 NSFW
          </div>
        )}

        {/* Stats Overlay */}
        <div style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          background: 'linear-gradient(to top, rgba(0,0,0,0.8), transparent)',
          padding: '8px 12px',
          display: 'flex',
          gap: '12px',
          fontSize: '12px',
          color: '#fff'
        }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Eye size={14} />
            {item.view_count}
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Heart size={14} />
            {item.like_count}
          </span>
        </div>
      </div>

      {/* Info */}
      <div style={{ padding: '12px' }}>
        <h3 style={{
          margin: '0 0 6px',
          fontSize: '14px',
          fontWeight: 600,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap'
        }}>
          {item.title}
        </h3>

        {item.description && (
          <p style={{
            margin: '0 0 8px',
            fontSize: '12px',
            color: '#888',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
            lineHeight: 1.4
          }}>
            {item.description}
          </p>
        )}

        {/* Tags */}
        {item.tags && item.tags.length > 0 && (
          <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '4px',
            marginTop: '8px'
          }}>
            {item.tags.slice(0, 3).map((tag, idx) => (
              <span
                key={idx}
                style={{
                  fontSize: '11px',
                  padding: '2px 8px',
                  background: '#3a3a3a',
                  borderRadius: '4px',
                  color: '#aaa'
                }}
              >
                {tag}
              </span>
            ))}
            {item.tags.length > 3 && (
              <span style={{ fontSize: '11px', color: '#666' }}>
                +{item.tags.length - 3}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  )
})

LazyMediaItem.displayName = 'LazyMediaItem'

export default function Gallery({ onRemix = null }) {
  const DEBUG = true // Enable debug for troubleshooting
  const debugLog = (message, data = null) => {
    if (!DEBUG) return
    if (data) {
      console.log('🐛 [Gallery]', message, data)
    } else {
      console.log('🐛 [Gallery]', message)
    }
  }

  const { user } = useAuth()
  const { nsfwEnabled } = useNSFW()
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [mediaType, setMediaType] = useState('all') // all, video, image
  const [sortBy, setSortBy] = useState('created_at') // created_at, like_count, view_count
  const [page, setPage] = useState(1)
  const [hasMore, setHasMore] = useState(true)
  const [total, setTotal] = useState(0)
  const containerRef = useRef(null)
  const [selectedItem, setSelectedItem] = useState(null)
  const PAGE_SIZE = 12 // Items per page

  // Deep-link: ?openItem=<media_id> — fetch + auto-open that item
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const itemId = params.get('openItem')
    if (!itemId) return
    apiFetch(`/api/gallery/${itemId}`)
      .then((res) => res.ok ? res.json() : null)
      .then((data) => { if (data) setSelectedItem(data) })
      .catch((err) => console.warn('⚠️ [Gallery] deep-link fetch failed:', err))
    // Remove the param so refresh doesn't re-open
    const url = new URL(window.location.href)
    url.searchParams.delete('openItem')
    window.history.replaceState({}, '', url.toString())
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Multi-select state
  const [selectMode, setSelectMode] = useState(false)
  const [selectedIds, setSelectedIds] = useState(new Set())
  const [batchDownloading, setBatchDownloading] = useState(false)

  const toggleSelectMode = () => {
    setSelectMode(prev => !prev)
    setSelectedIds(new Set()) // clear selection when toggling
  }

  const toggleSelectItem = (itemId) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(itemId)) next.delete(itemId)
      else next.add(itemId)
      return next
    })
  }

  const selectAll = () => setSelectedIds(new Set(items.map(i => i.id)))
  const clearSelection = () => setSelectedIds(new Set())

  const handleBatchDownload = async () => {
    if (selectedIds.size === 0) return
    setBatchDownloading(true)
    try {
      const batchItems = items
        .filter(i => selectedIds.has(i.id))
        .map(i => ({
          url: `/api/gallery/${i.id}/file`,
          filename: `${i.title?.replace(/[^a-z0-9]/gi, '_') || i.id}.${i.media_type === 'video' ? 'mp4' : i.media_type === 'audio' ? 'mp3' : 'png'}`,
        }))

      const response = await apiFetch('/api/media/batch-download-zip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ items: batchItems }),
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Download failed' }))
        setError(`Batch download failed: ${err.detail || 'Unknown error'}`)
        return
      }

      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `oelala_gallery_${new Date().toISOString().slice(0, 10)}.zip`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error('❌ Gallery batch download error:', err)
      setError(`Batch download failed: ${err.message}`)
    } finally {
      setBatchDownloading(false)
    }
  }

  // Fetch gallery items
  const fetchGallery = useCallback(async (resetPage = false) => {
    setLoading(true)
    setError('')

    const currentPage = resetPage ? 1 : page
    if (resetPage) {
      setPage(1)
      setItems([])
    }

    try {
      // Build query params
      const params = new URLSearchParams()
      if (mediaType && mediaType !== 'all') {
        params.append('media_type', mediaType)
      }
      // If user is not logged in or NSFW is disabled, force SFW
      if (!user || !nsfwEnabled) {
        params.append('is_nsfw', 'false')
      }
      params.append('sort_by', sortBy)
      params.append('order', 'desc')
      params.append('page', currentPage.toString())
      params.append('per_page', PAGE_SIZE.toString())

      debugLog('🔍 Fetching gallery', { currentPage, mediaType, sortBy })
      const response = await apiFetch(`/api/gallery?${params.toString()}`)

      if (!response.ok) {
        throw new Error('Failed to fetch gallery')
      }

      const data = await response.json()
      debugLog('📸 Gallery data received', { total: data.total, itemCount: data.items?.length })

      if (resetPage) {
        setItems(data.items)
      } else {
        setItems(prev => [...prev, ...data.items])
      }

      setTotal(data.total)
      setHasMore(data.has_more)

    } catch (err) {
      console.error('❌ Gallery error:', err)
      setError(err.message || 'Failed to load gallery')
    } finally {
      setLoading(false)
    }
  }, [mediaType, sortBy, page, user, nsfwEnabled])

  // Fetch on mount and when filters change
  useEffect(() => {
    fetchGallery(true)
  }, [mediaType, sortBy, user, nsfwEnabled])

  const loadNextPage = useCallback(() => {
    if (loading || !hasMore) return
    debugLog('📄 Loading next page')
    setPage(prev => prev + 1)
  }, [loading, hasMore])

  const handleScroll = useCallback((e) => {
    const { scrollTop, clientHeight, scrollHeight } = e.target
    if (scrollTop <= 0) return
    if (scrollHeight - scrollTop - clientHeight < 300) {
      loadNextPage()
    }
  }, [loadNextPage])

  useEffect(() => {
    if (page > 1) {
      fetchGallery(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page])

  // Get media URL for display - use the PUBLIC gallery file endpoint
  // This streams media from the owner's storage without requiring auth
  const getMediaUrl = useCallback((item) => {
    return `${BACKEND_BASE}/api/gallery/${item.id}/file`
  }, [])

  const getPreviewUrl = useCallback((item) => {
    if (!item.thumbnail_url) {
      return null
    }
    if (item.thumbnail_url.startsWith('http://') || item.thumbnail_url.startsWith('https://')) {
      return item.thumbnail_url
    }
    return `${BACKEND_BASE}${item.thumbnail_url}`
  }, [])

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: 'var(--bg-primary, #1a1a1a)',
      color: 'var(--text-primary, #fff)',
    }}>
      {/* Header */}
      <div style={{
        padding: '16px 24px',
        borderBottom: '1px solid #333',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '12px'
      }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 600 }}>
            🖼️ Community Gallery
          </h2>
          <p style={{ margin: '4px 0 0', fontSize: '0.9rem', color: '#888' }}>
            Discover amazing AI creations from the community
            {total > 0 && ` · ${total} items`}
          </p>
        </div>

        {/* Filters */}
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          {/* Media Type Filter */}
          <div style={{ display: 'flex', gap: '6px' }}>
            <button
              onClick={() => setMediaType('all')}
              style={{
                padding: '8px 12px',
                background: mediaType === 'all' ? '#3b82f6' : '#2a2a2a',
                border: '1px solid #444',
                borderRadius: '6px',
                color: '#fff',
                fontSize: '13px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <Filter size={14} />
              All
            </button>
            <button
              onClick={() => setMediaType('video')}
              style={{
                padding: '8px 12px',
                background: mediaType === 'video' ? '#3b82f6' : '#2a2a2a',
                border: '1px solid #444',
                borderRadius: '6px',
                color: '#fff',
                fontSize: '13px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <Video size={14} />
              Videos
            </button>
            <button
              onClick={() => setMediaType('image')}
              style={{
                padding: '8px 12px',
                background: mediaType === 'image' ? '#3b82f6' : '#2a2a2a',
                border: '1px solid #444',
                borderRadius: '6px',
                color: '#fff',
                fontSize: '13px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <ImageIcon size={14} />
              Images
            </button>
          </div>

          {/* Sort */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            style={{
              padding: '8px 12px',
              background: '#2a2a2a',
              border: '1px solid #444',
              borderRadius: '6px',
              color: '#fff',
              fontSize: '13px',
              cursor: 'pointer',
            }}
          >
            <option value="created_at">Newest</option>
            <option value="like_count">Most Liked</option>
            <option value="view_count">Most Viewed</option>
          </select>

          {/* Refresh */}
          <button
            onClick={() => fetchGallery(true)}
            disabled={loading}
            style={{
              padding: '8px 12px',
              background: '#2a2a2a',
              border: '1px solid #444',
              borderRadius: '6px',
              color: '#fff',
              cursor: loading ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              opacity: loading ? 0.5 : 1,
            }}
          >
            <RefreshCw size={16} className={loading ? 'spinning' : ''} />
          </button>

          {/* Select mode toggle (logged-in users only) */}
          {user && (
            <button
              onClick={toggleSelectMode}
              style={{
                padding: '8px 12px',
                background: selectMode ? '#3b82f6' : '#2a2a2a',
                border: `1px solid ${selectMode ? '#3b82f6' : '#444'}`,
                borderRadius: '6px',
                color: '#fff',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontSize: '13px',
              }}
            >
              {selectMode ? <CheckSquare size={14} /> : <Square size={14} />}
              {selectMode ? 'Cancel' : 'Select'}
            </button>
          )}
        </div>
      </div>

      {/* NSFW Warning for anonymous users */}
      {!user && (
        <div style={{
          margin: '16px 24px',
          padding: '12px 16px',
          background: 'rgba(59, 130, 246, 0.1)',
          border: '1px solid rgba(59, 130, 246, 0.3)',
          borderRadius: '8px',
          fontSize: '14px',
          color: '#60a5fa'
        }}>
          🔒 Log in to view all content. Gallery is filtered to SFW content for anonymous users.
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{
          margin: '16px 24px',
          padding: '12px 16px',
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: '8px',
          fontSize: '14px',
          color: '#ef4444'
        }}>
          ❌ {error}
        </div>
      )}

      {/* Batch action bar (visible in select mode) */}
      {selectMode && (
        <div style={{
          margin: '0 24px 8px',
          padding: '10px 16px',
          background: '#1e3a5f',
          border: '1px solid #3b82f6',
          borderRadius: '8px',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          flexWrap: 'wrap',
        }}>
          <span style={{ color: '#93c5fd', fontSize: '14px', fontWeight: 500 }}>
            {selectedIds.size} selected
          </span>

          <button
            onClick={selectAll}
            style={{
              padding: '6px 12px',
              background: 'transparent',
              border: '1px solid #3b82f6',
              borderRadius: '6px',
              color: '#93c5fd',
              cursor: 'pointer',
              fontSize: '13px',
            }}
          >
            Select all ({items.length})
          </button>

          {selectedIds.size > 0 && (
            <button
              onClick={clearSelection}
              style={{
                padding: '6px 12px',
                background: 'transparent',
                border: '1px solid #666',
                borderRadius: '6px',
                color: '#aaa',
                cursor: 'pointer',
                fontSize: '13px',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
              }}
            >
              <XIcon size={12} /> Clear
            </button>
          )}

          {selectedIds.size > 0 && (
            <button
              onClick={handleBatchDownload}
              disabled={batchDownloading}
              style={{
                padding: '6px 14px',
                background: batchDownloading ? '#1e3a5f' : '#3b82f6',
                border: '1px solid #3b82f6',
                borderRadius: '6px',
                color: '#fff',
                cursor: batchDownloading ? 'not-allowed' : 'pointer',
                fontSize: '13px',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontWeight: 500,
              }}
            >
              <Download size={14} />
              {batchDownloading ? 'Preparing ZIP…' : `Download ZIP (${selectedIds.size})`}
            </button>
          )}
        </div>
      )}

      {/* Gallery Grid */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '24px',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
          gap: '20px',
          alignContent: 'start'
        }}
      >
        {items.map((item) => (
          <LazyMediaItem
            key={item.id}
            item={item}
            getMediaUrl={getMediaUrl}
            getPreviewUrl={getPreviewUrl}
            selected={selectedIds.has(item.id)}
            selectMode={selectMode}
            onClick={() => {
              if (selectMode) {
                toggleSelectItem(item.id)
              } else {
                setSelectedItem(item)
              }
            }}
          />
        ))}

        {/* Loading indicator */}
        {loading && (
          <div style={{
            gridColumn: '1 / -1',
            textAlign: 'center',
            padding: '20px',
            color: '#888'
          }}>
            Loading more...
          </div>
        )}

        {/* No more items */}
        {!loading && !hasMore && items.length > 0 && (
          <div style={{
            gridColumn: '1 / -1',
            textAlign: 'center',
            padding: '20px',
            color: '#666'
          }}>
            No more items
          </div>
        )}


        {/* Empty state */}
        {!loading && items.length === 0 && (
          <div style={{
            gridColumn: '1 / -1',
            textAlign: 'center',
            padding: '60px 20px',
            color: '#666'
          }}>
            <ImageIcon size={48} style={{ marginBottom: '16px', opacity: 0.3 }} />
            <p style={{ fontSize: '16px' }}>No items in the gallery yet</p>
            <p style={{ fontSize: '14px', marginTop: '8px' }}>
              Be the first to publish your creations!
            </p>
          </div>
        )}
      </div>

      {selectedItem && (
        <MediaDetailModal
          item={selectedItem}
          onClose={() => setSelectedItem(null)}
          onRemix={onRemix ? (settings) => {
            const toolId = selectedItem.media_type === 'video'
              ? TOOL_IDS.TEXT_TO_VIDEO
              : TOOL_IDS.TEXT_TO_IMAGE
            onRemix(toolId, settings)
            setSelectedItem(null)
          } : null}
        />
      )}
    </div>
  )
}
