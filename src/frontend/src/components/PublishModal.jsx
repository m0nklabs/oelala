import React, { useState } from 'react'
import { X, Upload, Tag, AlertCircle } from 'lucide-react'
import { apiFetch } from '../api'
import { BACKEND_BASE } from '../config'

export default function PublishModal({ mediaItem, onClose, onPublished }) {
  const [title, setTitle] = useState(mediaItem.metadata?.positive_prompt?.slice(0, 100) || '')
  const [description, setDescription] = useState('')
  const [tags, setTags] = useState('')
  const [isNsfw, setIsNsfw] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Determine media type from filename
  const getMediaType = (filename) => {
    const ext = filename.toLowerCase().split('.').pop()
    if (['mp4', 'webm', 'mov', 'avi'].includes(ext)) return 'video'
    if (['jpg', 'jpeg', 'png', 'gif', 'webp'].includes(ext)) return 'image'
    if (['mp3', 'wav', 'ogg', 'flac'].includes(ext)) return 'audio'
    return 'image'
  }

  // Build media URL for preview
  const getMediaUrl = () => {
    if (mediaItem.source === 'storage') {
      // User storage media
      const mediaType = getMediaType(mediaItem.filename)
      return `${BACKEND_BASE}/user/media/${mediaType}/${mediaItem.filename}`
    } else {
      // ComfyUI output media
      return `${BACKEND_BASE}/comfyui/output/${mediaItem.filename}`
    }
  }

  const handlePublish = async () => {
    if (!title.trim()) {
      setError('Title is required')
      return
    }

    if (title.length > 100) {
      setError('Title must be 100 characters or less')
      return
    }

    if (description.length > 500) {
      setError('Description must be 500 characters or less')
      return
    }

    setLoading(true)
    setError('')

    try {
      // Parse tags
      const tagList = tags
        .split(',')
        .map(t => t.trim())
        .filter(t => t.length > 0)
        .slice(0, 10) // Max 10 tags

      // Build storage path
      const mediaType = getMediaType(mediaItem.filename)
      const storagePath = mediaItem.source === 'storage' 
        ? `${mediaType}/${mediaItem.filename}`
        : mediaItem.filename

      const payload = {
        storage_path: storagePath,
        title: title.trim(),
        description: description.trim() || null,
        tags: tagList,
        is_nsfw: isNsfw,
        media_type: mediaType,
        thumbnail_url: null, // Could add thumbnail generation later
        metadata: mediaItem.metadata || {},
      }

      const response = await apiFetch('/api/gallery/publish', {
        method: 'POST',
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to publish')
      }

      const published = await response.json()
      console.log('✅ Published successfully:', published)

      if (onPublished) {
        onPublished(published)
      }

      onClose()
    } catch (err) {
      console.error('❌ Publish error:', err)
      setError(err.message || 'Failed to publish media')
    } finally {
      setLoading(false)
    }
  }

  const mediaUrl = getMediaUrl()
  const mediaType = getMediaType(mediaItem.filename)

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div 
        className="modal-content" 
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: '600px', maxHeight: '90vh', overflowY: 'auto' }}
      >
        {/* Header */}
        <div className="modal-header" style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          marginBottom: '16px',
          paddingBottom: '12px',
          borderBottom: '1px solid #333'
        }}>
          <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Upload size={20} />
            Publish to Gallery
          </h3>
          <button 
            onClick={onClose}
            style={{ 
              background: 'none', 
              border: 'none', 
              color: '#ccc', 
              cursor: 'pointer',
              padding: '4px',
              display: 'flex'
            }}
          >
            <X size={20} />
          </button>
        </div>

        {/* Preview */}
        <div style={{ marginBottom: '16px', borderRadius: '8px', overflow: 'hidden', background: '#000' }}>
          {mediaType === 'video' ? (
            <video 
              src={mediaUrl} 
              controls 
              style={{ width: '100%', maxHeight: '300px', objectFit: 'contain' }}
            />
          ) : (
            <img 
              src={mediaUrl} 
              alt="Preview" 
              style={{ width: '100%', maxHeight: '300px', objectFit: 'contain' }}
            />
          )}
        </div>

        {/* Error message */}
        {error && (
          <div style={{
            marginBottom: '16px',
            padding: '12px',
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: '6px',
            color: '#ef4444',
            display: 'flex',
            alignItems: 'flex-start',
            gap: '8px'
          }}>
            <AlertCircle size={18} style={{ marginTop: '2px', flexShrink: 0 }} />
            <span>{error}</span>
          </div>
        )}

        {/* Form */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {/* Title */}
          <div>
            <label style={{ display: 'block', marginBottom: '6px', fontSize: '13px', color: '#ccc' }}>
              Title <span style={{ color: '#ef4444' }}>*</span>
            </label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Give your creation a catchy title..."
              maxLength={100}
              style={{
                width: '100%',
                padding: '8px 12px',
                background: '#1a1a1a',
                border: '1px solid #333',
                borderRadius: '6px',
                color: '#fff',
                fontSize: '14px',
              }}
            />
            <div style={{ fontSize: '11px', color: '#666', marginTop: '4px', textAlign: 'right' }}>
              {title.length}/100
            </div>
          </div>

          {/* Description */}
          <div>
            <label style={{ display: 'block', marginBottom: '6px', fontSize: '13px', color: '#ccc' }}>
              Description (optional)
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Add a description to help people understand your creation..."
              maxLength={500}
              rows={4}
              style={{
                width: '100%',
                padding: '8px 12px',
                background: '#1a1a1a',
                border: '1px solid #333',
                borderRadius: '6px',
                color: '#fff',
                fontSize: '14px',
                resize: 'vertical',
              }}
            />
            <div style={{ fontSize: '11px', color: '#666', marginTop: '4px', textAlign: 'right' }}>
              {description.length}/500
            </div>
          </div>

          {/* Tags */}
          <div>
            <label style={{ display: 'block', marginBottom: '6px', fontSize: '13px', color: '#ccc', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Tag size={14} />
              Tags (comma-separated, max 10)
            </label>
            <input
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="e.g., anime, portrait, fantasy"
              style={{
                width: '100%',
                padding: '8px 12px',
                background: '#1a1a1a',
                border: '1px solid #333',
                borderRadius: '6px',
                color: '#fff',
                fontSize: '14px',
              }}
            />
          </div>

          {/* NSFW Toggle */}
          <div style={{
            padding: '12px',
            background: '#1a1a1a',
            border: '1px solid #333',
            borderRadius: '6px',
          }}>
            <label style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '12px',
              cursor: 'pointer',
              fontSize: '14px',
            }}>
              <input
                type="checkbox"
                checked={isNsfw}
                onChange={(e) => setIsNsfw(e.target.checked)}
                style={{ 
                  width: '18px', 
                  height: '18px',
                  cursor: 'pointer',
                }}
              />
              <span>
                <span style={{ fontWeight: '500' }}>Mark as NSFW</span>
                <span style={{ display: 'block', fontSize: '12px', color: '#888', marginTop: '2px' }}>
                  Content will only be visible to logged-in users
                </span>
              </span>
            </label>
          </div>
        </div>

        {/* Actions */}
        <div style={{ 
          display: 'flex', 
          gap: '12px', 
          marginTop: '24px',
          paddingTop: '16px',
          borderTop: '1px solid #333'
        }}>
          <button
            onClick={onClose}
            disabled={loading}
            style={{
              flex: 1,
              padding: '10px 20px',
              background: '#2a2a2a',
              border: '1px solid #444',
              borderRadius: '6px',
              color: '#ccc',
              fontSize: '14px',
              fontWeight: '500',
              cursor: loading ? 'not-allowed' : 'pointer',
              opacity: loading ? 0.5 : 1,
            }}
          >
            Cancel
          </button>
          <button
            onClick={handlePublish}
            disabled={loading || !title.trim()}
            style={{
              flex: 1,
              padding: '10px 20px',
              background: loading || !title.trim() ? '#444' : '#3b82f6',
              border: 'none',
              borderRadius: '6px',
              color: '#fff',
              fontSize: '14px',
              fontWeight: '500',
              cursor: loading || !title.trim() ? 'not-allowed' : 'pointer',
              opacity: loading ? 0.7 : 1,
            }}
          >
            {loading ? 'Publishing...' : 'Publish'}
          </button>
        </div>
      </div>
    </div>
  )
}
