import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Filter, Image as ImageIcon, Video, RefreshCw, Eye, Heart } from 'lucide-react'
import { apiFetch } from '../api'
import { BACKEND_BASE } from '../config'
import { useAuth } from '../contexts/AuthContext'
import { useNSFW } from '../contexts/NSFWContext'
import MediaDetailModal from '../components/MediaDetailModal'

export default function Gallery() {
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
      params.append('per_page', '30')

      const response = await apiFetch(`/api/gallery?${params.toString()}`)

      if (!response.ok) {
        throw new Error('Failed to fetch gallery')
      }

      const data = await response.json()
      console.log('📸 Gallery data:', data)

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

  // Infinite scroll
  const handleScroll = useCallback((e) => {
    const { scrollTop, clientHeight, scrollHeight } = e.target
    if (scrollHeight - scrollTop - clientHeight < 500 && !loading && hasMore) {
      setPage(prev => prev + 1)
    }
  }, [loading, hasMore])

  useEffect(() => {
    if (page > 1) {
      fetchGallery(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page])

  // Get media URL for display
  const getMediaUrl = (item) => {
    // Assuming storage_path is like "video/filename.mp4" or "image/filename.png"
    return `${BACKEND_BASE}/user/media/${item.storage_path}`
  }

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
          <div
            key={item.id}
            onClick={() => setSelectedItem(item)}
            style={{
              background: '#2a2a2a',
              borderRadius: '12px',
              overflow: 'hidden',
              cursor: 'pointer',
              transition: 'transform 0.2s, box-shadow 0.2s',
              border: '1px solid #333',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-4px)'
              e.currentTarget.style.boxShadow = '0 8px 24px rgba(0,0,0,0.4)'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)'
              e.currentTarget.style.boxShadow = 'none'
            }}
          >
            {/* Media Preview */}
            <div style={{
              aspectRatio: '9/16',
              background: '#000',
              position: 'relative',
              overflow: 'hidden'
            }}>
              {item.media_type === 'video' ? (
                <video
                  src={getMediaUrl(item)}
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover'
                  }}
                />
              ) : (
                <img
                  src={getMediaUrl(item)}
                  alt={item.title}
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover'
                  }}
                />
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
        />
      )}
    </div>
  )
}
