import React, { useState, useEffect } from 'react'
import { X, Heart, Eye, Share2, Copy, Check, AlertCircle, Download, FileJson, Shuffle, User, Flag } from 'lucide-react'
import { BACKEND_BASE } from '../config'
import { apiFetch } from '../api'
import { useAuth } from '../contexts/AuthContext'

export default function MediaDetailModal({ item, onClose, onRemix = null, onViewProfile = null }) {
  const { user } = useAuth()
  const [liked, setLiked] = useState(item.user_liked || false)
  const [likeCount, setLikeCount] = useState(item.like_count || 0)
  const [viewCount, setViewCount] = useState(item.view_count || 0)
  const [copying, setCopying] = useState(false)
  const [copied, setCopied] = useState(false)
  const [likeError, setLikeError] = useState('')
  const [downloadingWorkflow, setDownloadingWorkflow] = useState(false)
  const [workflowError, setWorkflowError] = useState('')
  const [showReportModal, setShowReportModal] = useState(false)
  const [reportReason, setReportReason] = useState('')
  const [reportDescription, setReportDescription] = useState('')
  const [reportSubmitting, setReportSubmitting] = useState(false)
  const [reportSuccess, setReportSuccess] = useState(false)
  const [reportError, setReportError] = useState('')

  // Fetch fresh item data on open — gets accurate user_liked + triggers view increment
  useEffect(() => {
    const fetchFresh = async () => {
      try {
        const response = await apiFetch(`/api/gallery/${item.id}`)
        if (response.ok) {
          const data = await response.json()
          if (data.user_liked !== undefined && data.user_liked !== null) {
            setLiked(data.user_liked)
          }
          if (data.like_count !== undefined) setLikeCount(data.like_count)
          if (data.view_count !== undefined) setViewCount(data.view_count)
        }
      } catch (err) {
        console.warn('⚠️ Failed to refresh media data:', err)
      }
    }
    fetchFresh()
  }, [item.id]) // eslint-disable-line react-hooks/exhaustive-deps

  // Download workflow JSON from media file via backend API
  const handleDownloadWorkflow = async () => {
    console.log('🔧 handleDownloadWorkflow called, item.id:', item.id)
    setDownloadingWorkflow(true)
    setWorkflowError('')
    try {
      // Call gallery API to extract workflow from published media
      const url = `/api/gallery/${item.id}/workflow`
      console.log('🔧 Fetching workflow from:', url)
      const response = await apiFetch(url)
      console.log('🔧 Response status:', response.status)

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Failed to extract workflow')
      }

      const data = await response.json()
      console.log('🔧 Workflow data keys:', Object.keys(data))
      const workflow = data.workflow

      if (workflow) {
        const workflowJson = JSON.stringify(workflow, null, 2)
        const workflowBlob = new Blob([workflowJson], { type: 'application/json' })
        const url = URL.createObjectURL(workflowBlob)
        const a = document.createElement('a')
        a.href = url
        a.download = `${item.title?.replace(/[^a-z0-9]/gi, '_') || 'workflow'}_workflow.json`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(url)
      } else {
        setWorkflowError('No workflow found')
      }
    } catch (err) {
      console.error('Failed to download workflow:', err)
      setWorkflowError(err.message || 'Download failed')
      setTimeout(() => setWorkflowError(''), 3000)
    } finally {
      setDownloadingWorkflow(false)
    }
  }

  // Get media URL
  const getMediaUrl = () => {
    return `${BACKEND_BASE}/user/media/${item.storage_path}`
  }

  // Toggle like
  const handleLike = async () => {
    if (!user) {
      setLikeError('Please log in to like items')
      setTimeout(() => setLikeError(''), 3000)
      return
    }

    try {
      setLikeError('')
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
      setLikeError('Failed to update like status')
      setTimeout(() => setLikeError(''), 3000)
    }
  }

  // Copy share link  — points to /share/{id} for social preview support
  const handleShare = async () => {
    const shareUrl = `${window.location.origin}/share/${item.id}`
    try {
      if (navigator.share) {
        await navigator.share({ title: item.title, url: shareUrl })
        return
      }
      await navigator.clipboard.writeText(shareUrl)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      if (err.name !== 'AbortError') console.error('Failed to share:', err)
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

  const handleReport = async () => {
    if (!reportReason) return
    setReportSubmitting(true)
    setReportError('')
    try {
      const resp = await apiFetch('/api/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          media_id: item.id,
          reason: reportReason,
          description: reportDescription || null,
        }),
      })
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: 'Report failed' }))
        throw new Error(err.detail || 'Report failed')
      }
      setReportSuccess(true)
      setTimeout(() => {
        setShowReportModal(false)
        setReportSuccess(false)
        setReportReason('')
        setReportDescription('')
      }, 2000)
    } catch (err) {
      setReportError(err.message)
    } finally {
      setReportSubmitting(false)
    }
  }

  const REPORT_REASONS = [
    { value: 'inappropriate', label: 'Inappropriate content' },
    { value: 'copyright', label: 'Copyright violation' },
    { value: 'spam', label: 'Spam or misleading' },
    { value: 'harassment', label: 'Harassment or bullying' },
    { value: 'underage', label: 'Underage content' },
    { value: 'other', label: 'Other' },
  ]

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

              {/* Creator Info */}
              {(item.creator_username || item.creator_display_name) && (
                <button
                  onClick={() => onViewProfile && onViewProfile(item.user_id)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    marginBottom: '10px',
                    background: 'none',
                    border: 'none',
                    padding: 0,
                    cursor: onViewProfile ? 'pointer' : 'default',
                    color: '#ccc'
                  }}
                >
                  {item.creator_avatar_url ? (
                    <img
                      src={item.creator_avatar_url}
                      alt=""
                      style={{
                        width: 28,
                        height: 28,
                        borderRadius: '50%',
                        objectFit: 'cover'
                      }}
                    />
                  ) : (
                    <div style={{
                      width: 28,
                      height: 28,
                      borderRadius: '50%',
                      background: 'linear-gradient(135deg, #7c3aed, #a855f7)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}>
                      <User size={14} color="#fff" />
                    </div>
                  )}
                  <span style={{ fontSize: '14px', fontWeight: 500 }}>
                    {item.creator_display_name || `@${item.creator_username}`}
                  </span>
                  {item.creator_display_name && item.creator_username && (
                    <span style={{ fontSize: '13px', color: '#666' }}>
                      @{item.creator_username}
                    </span>
                  )}
                </button>
              )}

              {/* Stats */}
              <div style={{
                display: 'flex',
                gap: '16px',
                fontSize: '14px',
                color: '#888'
              }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Eye size={16} />
                  {viewCount} views
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
            flexDirection: 'column',
            gap: '12px'
          }}>
            {/* Error message */}
            {(likeError || workflowError) && (
              <div style={{
                padding: '10px 12px',
                background: 'rgba(239, 68, 68, 0.1)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                borderRadius: '6px',
                color: '#ef4444',
                fontSize: '13px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                <AlertCircle size={16} />
                {likeError || workflowError}
              </div>
            )}

            {/* Action buttons */}
            <div style={{ display: 'flex', gap: '12px' }}>
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

              {/* Remix Button — only shown when prompt metadata is available */}
              {onRemix && (item.metadata?.positive_prompt || item.metadata?.prompt) && (
                <button
                  onClick={() => {
                    onRemix({
                      positive: item.metadata?.positive_prompt || item.metadata?.prompt,
                      negative: item.metadata?.negative_prompt,
                      steps: item.metadata?.steps,
                      cfg: item.metadata?.cfg_scale,
                      seed: item.metadata?.seed,
                    })
                    onClose()
                  }}
                  title="Open in generator with these settings"
                  style={{
                    padding: '10px 16px',
                    background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                    border: 'none',
                    borderRadius: '8px',
                    color: '#fff',
                    fontSize: '14px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '8px',
                  }}
                >
                  <Shuffle size={16} />
                  Remix
                </button>
              )}

              {/* Workflow Download Button */}
              {item.metadata && (
                <button
                  onClick={handleDownloadWorkflow}
                  disabled={downloadingWorkflow}
                  title="Download ComfyUI workflow JSON"
                  style={{
                    padding: '10px 16px',
                    background: '#2a2a2a',
                    border: '1px solid #444',
                    borderRadius: '8px',
                    color: '#fff',
                    fontSize: '14px',
                    fontWeight: 500,
                    cursor: downloadingWorkflow ? 'wait' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '8px',
                    opacity: downloadingWorkflow ? 0.7 : 1
                  }}
                >
                  <FileJson size={16} />
                  WF
                </button>
              )}

              {/* Report Button — not shown for own content */}
              {user && item.user_id !== user.id && (
                <button
                  onClick={() => setShowReportModal(true)}
                  title="Report this content"
                  style={{
                    padding: '10px 16px',
                    background: '#2a2a2a',
                    border: '1px solid #444',
                    borderRadius: '8px',
                    color: '#ef4444',
                    fontSize: '14px',
                    fontWeight: 500,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '8px',
                  }}
                >
                  <Flag size={16} />
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Report Modal */}
      {showReportModal && (
        <div
          onClick={(e) => { e.stopPropagation(); setShowReportModal(false) }}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.7)',
            zIndex: 1100,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '20px',
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              background: '#1a1a1a',
              border: '1px solid #333',
              borderRadius: '12px',
              width: '100%',
              maxWidth: '420px',
              overflow: 'hidden',
            }}
          >
            <div style={{
              padding: '16px 20px',
              borderBottom: '1px solid #333',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}>
              <h3 style={{ margin: 0, fontSize: '16px', color: '#fff' }}>
                <Flag size={16} style={{ marginRight: '8px', verticalAlign: 'middle', color: '#ef4444' }} />
                Report Content
              </h3>
              <button
                onClick={() => setShowReportModal(false)}
                style={{ background: 'none', border: 'none', color: '#888', cursor: 'pointer' }}
              >
                <X size={18} />
              </button>
            </div>

            <div style={{ padding: '16px 20px' }}>
              {reportSuccess ? (
                <div style={{
                  textAlign: 'center',
                  padding: '24px 0',
                  color: '#10b981',
                  fontSize: '15px',
                }}>
                  <Check size={32} style={{ marginBottom: '8px' }} />
                  <p>Report submitted. Thank you.</p>
                </div>
              ) : (
                <>
                  <p style={{ color: '#aaa', fontSize: '13px', margin: '0 0 12px' }}>
                    Why are you reporting this content?
                  </p>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginBottom: '12px' }}>
                    {REPORT_REASONS.map(r => (
                      <label
                        key={r.value}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '10px',
                          padding: '8px 12px',
                          background: reportReason === r.value ? '#2a2a2a' : 'transparent',
                          border: `1px solid ${reportReason === r.value ? '#6366f1' : '#333'}`,
                          borderRadius: '8px',
                          cursor: 'pointer',
                          color: '#ddd',
                          fontSize: '14px',
                        }}
                      >
                        <input
                          type="radio"
                          name="report_reason"
                          value={r.value}
                          checked={reportReason === r.value}
                          onChange={(e) => setReportReason(e.target.value)}
                          style={{ accentColor: '#6366f1' }}
                        />
                        {r.label}
                      </label>
                    ))}
                  </div>

                  <textarea
                    placeholder="Additional details (optional)"
                    value={reportDescription}
                    onChange={(e) => setReportDescription(e.target.value)}
                    maxLength={500}
                    rows={3}
                    style={{
                      width: '100%',
                      background: '#2a2a2a',
                      border: '1px solid #444',
                      borderRadius: '8px',
                      padding: '10px 12px',
                      color: '#fff',
                      fontSize: '14px',
                      resize: 'vertical',
                      marginBottom: '12px',
                      boxSizing: 'border-box',
                    }}
                  />

                  {reportError && (
                    <div style={{
                      padding: '8px 12px',
                      background: 'rgba(239, 68, 68, 0.1)',
                      border: '1px solid rgba(239, 68, 68, 0.3)',
                      borderRadius: '6px',
                      color: '#ef4444',
                      fontSize: '13px',
                      marginBottom: '12px',
                    }}>
                      {reportError}
                    </div>
                  )}

                  <button
                    onClick={handleReport}
                    disabled={!reportReason || reportSubmitting}
                    style={{
                      width: '100%',
                      padding: '10px',
                      background: reportReason ? '#ef4444' : '#333',
                      border: 'none',
                      borderRadius: '8px',
                      color: '#fff',
                      fontSize: '14px',
                      fontWeight: 600,
                      cursor: reportReason && !reportSubmitting ? 'pointer' : 'not-allowed',
                      opacity: !reportReason || reportSubmitting ? 0.5 : 1,
                    }}
                  >
                    {reportSubmitting ? 'Submitting...' : 'Submit Report'}
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
