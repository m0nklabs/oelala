import React, { useState } from 'react'
import { X, Heart, Eye, Share2, Copy, Check } from 'lucide-react'
import { BACKEND_BASE } from '../config'
import { apiFetch } from '../api'
import { useAuth } from '../contexts/AuthContext'

export default function MediaDetailModal({ item, onClose }) {
  const { user } = useAuth()
  const [liked, setLiked] = useState(item.user_liked || false)
  const [likeCount, setLikeCount] = useState(item.like_count || 0)
  const [copying, setCopying] = useState(false)
  const [copied, setCopied] = useState(false)

  // Get media URL
  const getMediaUrl = () => {
    return `${BACKEND_BASE}/user/media/${item.storage_path}`
  }

  // Toggle like
  const handleLike = async () => {
    if (!user) {
      alert('Please log in to like items')
      return
    }

    try {
      const response = await apiFetch(`/api/gallery/${item.id}/like`, {
        method: 'POST',
      })

      if (!response.ok) {
        throw new Error('Failed to toggle like')
      }

      const data = await response.json()
      setLiked(data.liked)
      setLikeCount(data.like_count)
    } catch (err) {
      console.error('❌ Like error:', err)
      alert('Failed to update like status')
    }
  }

  // Copy share link
  const handleShare = async () => {
    const url = `${window.location.origin}/gallery/${item.id}`
    try {
      await navigator.clipboard.writeText(url)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  // Copy prompt to clipboard
  const handleCopyPrompt = async () => {
    const prompt = item.metadata?.positive_prompt || item.metadata?.prompt
    if (!prompt) return

    try {
      await navigator.clipboard.writeText(prompt)
      setCopying(true)
      setTimeout(() => setCopying(false), 2000)
    } catch (err) {
      console.error('Failed to copy prompt:', err)
    }
  }

  const mediaUrl = getMediaUrl()

  return (
    <div 
      className="modal-overlay" 
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.9)',
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '20px',
      }}
    >
      <div 
        className="modal-content"
        onClick={(e) => e.stopPropagation()}
        style={{
          maxWidth: '1200px',
          width: '100%',
          maxHeight: '90vh',
          background: '#1a1a1a',
          borderRadius: '12px',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'row',
          border: '1px solid #333',
        }}
      >
        {/* Media Display */}
        <div style={{ 
          flex: '0 0 60%',
          background: '#000',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative'
        }}>
          {item.media_type === 'video' ? (
            <video
              src={mediaUrl}
              controls
              autoPlay
              loop
              style={{ 
                maxWidth: '100%', 
                maxHeight: '100%',
                objectFit: 'contain'
              }}
            />
          ) : (
            <img
              src={mediaUrl}
              alt={item.title}
              style={{ 
                maxWidth: '100%', 
                maxHeight: '100%',
                objectFit: 'contain'
              }}
            />
          )}
        </div>

        {/* Info Panel */}
        <div style={{ 
          flex: '0 0 40%',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden'
        }}>
          {/* Header */}
          <div style={{ 
            padding: '20px',
            borderBottom: '1px solid #333',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start'
          }}>
            <div style={{ flex: 1 }}>
              <h2 style={{ 
                margin: '0 0 8px',
                fontSize: '1.25rem',
                fontWeight: 600,
                color: '#fff'
              }}>
                {item.title}
              </h2>
              
              {/* Stats */}
              <div style={{ 
                display: 'flex', 
                gap: '16px',
                fontSize: '14px',
                color: '#888'
              }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Eye size={16} />
                  {item.view_count} views
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Heart size={16} />
                  {likeCount} likes
                </span>
              </div>
            </div>

            <button
              onClick={onClose}
              style={{
                background: 'none',
                border: 'none',
                color: '#ccc',
                cursor: 'pointer',
                padding: '4px',
                display: 'flex',
                marginLeft: '12px'
              }}
            >
              <X size={24} />
            </button>
          </div>

          {/* Scrollable Content */}
          <div style={{ 
            flex: 1,
            overflowY: 'auto',
            padding: '20px'
          }}>
            {/* Description */}
            {item.description && (
              <div style={{ marginBottom: '20px' }}>
                <p style={{ 
                  margin: 0,
                  fontSize: '14px',
                  lineHeight: 1.6,
                  color: '#ccc'
                }}>
                  {item.description}
                </p>
              </div>
            )}

            {/* Tags */}
            {item.tags && item.tags.length > 0 && (
              <div style={{ marginBottom: '20px' }}>
                <div style={{ 
                  fontSize: '13px',
                  color: '#888',
                  marginBottom: '8px',
                  fontWeight: 500
                }}>
                  Tags
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                  {item.tags.map((tag, idx) => (
                    <span
                      key={idx}
                      style={{
                        fontSize: '12px',
                        padding: '4px 10px',
                        background: '#2a2a2a',
                        borderRadius: '6px',
                        color: '#aaa',
                        border: '1px solid #333'
                      }}
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Prompt */}
            {(item.metadata?.positive_prompt || item.metadata?.prompt) && (
              <div style={{ marginBottom: '20px' }}>
                <div style={{ 
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '8px'
                }}>
                  <div style={{ 
                    fontSize: '13px',
                    color: '#888',
                    fontWeight: 500
                  }}>
                    Prompt
                  </div>
                  <button
                    onClick={handleCopyPrompt}
                    style={{
                      padding: '4px 8px',
                      background: copying ? '#10b981' : '#2a2a2a',
                      border: '1px solid #444',
                      borderRadius: '4px',
                      color: '#fff',
                      fontSize: '12px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px'
                    }}
                  >
                    {copying ? (
                      <>
                        <Check size={12} />
                        Copied
                      </>
                    ) : (
                      <>
                        <Copy size={12} />
                        Copy
                      </>
                    )}
                  </button>
                </div>
                <div style={{ 
                  padding: '12px',
                  background: '#2a2a2a',
                  borderRadius: '8px',
                  fontSize: '13px',
                  lineHeight: 1.5,
                  color: '#ccc',
                  border: '1px solid #333',
                  fontFamily: 'monospace',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word'
                }}>
                  {item.metadata?.positive_prompt || item.metadata?.prompt}
                </div>
              </div>
            )}

            {/* Negative Prompt */}
            {item.metadata?.negative_prompt && (
              <div style={{ marginBottom: '20px' }}>
                <div style={{ 
                  fontSize: '13px',
                  color: '#888',
                  marginBottom: '8px',
                  fontWeight: 500
                }}>
                  Negative Prompt
                </div>
                <div style={{ 
                  padding: '12px',
                  background: '#2a2a2a',
                  borderRadius: '8px',
                  fontSize: '13px',
                  lineHeight: 1.5,
                  color: '#ccc',
                  border: '1px solid #333',
                  fontFamily: 'monospace',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word'
                }}>
                  {item.metadata.negative_prompt}
                </div>
              </div>
            )}

            {/* Generation Settings */}
            {item.metadata && Object.keys(item.metadata).length > 0 && (
              <div style={{ marginBottom: '20px' }}>
                <div style={{ 
                  fontSize: '13px',
                  color: '#888',
                  marginBottom: '8px',
                  fontWeight: 500
                }}>
                  Settings
                </div>
                <div style={{ 
                  padding: '12px',
                  background: '#2a2a2a',
                  borderRadius: '8px',
                  fontSize: '12px',
                  color: '#ccc',
                  border: '1px solid #333'
                }}>
                  {item.metadata.model && (
                    <div style={{ marginBottom: '6px' }}>
                      <span style={{ color: '#888' }}>Model:</span>{' '}
                      <span style={{ color: '#aaa' }}>{item.metadata.model}</span>
                    </div>
                  )}
                  {item.metadata.steps && (
                    <div style={{ marginBottom: '6px' }}>
                      <span style={{ color: '#888' }}>Steps:</span>{' '}
                      <span style={{ color: '#aaa' }}>{item.metadata.steps}</span>
                    </div>
                  )}
                  {item.metadata.cfg_scale && (
                    <div style={{ marginBottom: '6px' }}>
                      <span style={{ color: '#888' }}>CFG Scale:</span>{' '}
                      <span style={{ color: '#aaa' }}>{item.metadata.cfg_scale}</span>
                    </div>
                  )}
                  {item.metadata.seed && (
                    <div>
                      <span style={{ color: '#888' }}>Seed:</span>{' '}
                      <span style={{ color: '#aaa' }}>{item.metadata.seed}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Actions */}
          <div style={{ 
            padding: '16px 20px',
            borderTop: '1px solid #333',
            display: 'flex',
            gap: '12px'
          }}>
            <button
              onClick={handleLike}
              disabled={!user}
              style={{
                flex: 1,
                padding: '10px 16px',
                background: liked ? '#ef4444' : '#2a2a2a',
                border: '1px solid #444',
                borderRadius: '8px',
                color: '#fff',
                fontSize: '14px',
                fontWeight: 500,
                cursor: user ? 'pointer' : 'not-allowed',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                opacity: user ? 1 : 0.5
              }}
            >
              <Heart size={16} fill={liked ? '#fff' : 'none'} />
              {liked ? 'Unlike' : 'Like'}
            </button>

            <button
              onClick={handleShare}
              style={{
                flex: 1,
                padding: '10px 16px',
                background: copied ? '#10b981' : '#2a2a2a',
                border: '1px solid #444',
                borderRadius: '8px',
                color: '#fff',
                fontSize: '14px',
                fontWeight: 500,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px'
              }}
            >
              {copied ? (
                <>
                  <Check size={16} />
                  Copied
                </>
              ) : (
                <>
                  <Share2 size={16} />
                  Share
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
