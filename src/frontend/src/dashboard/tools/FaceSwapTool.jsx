import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import {
  User, Users, Upload, Loader2, Download, AlertCircle,
  Smile, RefreshCw, Trash2, Plus, Check, Image as ImageIcon,
  Cpu, Zap, Clock, Copy, CheckCheck
} from 'lucide-react'
import { BACKEND_BASE, DEBUG, getMediaUrl } from '../../config'
import { apiFetch } from '../../api'
import { useAuth } from '../../contexts/AuthContext'
import MediaImportModal from '../../components/MediaImportModal'
import CreationsPickerModal from '../../components/CreationsPickerModal'
import { useToolSettings } from '../../hooks/useToolSettings'
import ResetDefaultsButton from '../../components/ResetDefaultsButton'

// ─────────────────────────────────────────────────────────────────────────────
// FaceSwapTool
// ─────────────────────────────────────────────────────────────────────────────

export default function FaceSwapTool({ onJobSubmitted, pendingImport, onImportConsumed }) {
  const { user, requestLogin } = useAuth()
  const [tab, setTab] = useState('swap') // 'swap' | 'profiles' | 'train'

  return (
    <div className="tool-container">
      {/* Tab pills */}
      <div className="grok-toggle-group">
        <button
          onClick={() => setTab('swap')}
          className={`grok-toggle-btn ${tab === 'swap' ? 'active' : ''}`}
          style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}
        >
          <RefreshCw size={14} />
          Swap
        </button>
        <button
          onClick={() => setTab('profiles')}
          className={`grok-toggle-btn ${tab === 'profiles' ? 'active' : ''}`}
          style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}
        >
          <Users size={14} />
          Profiles
        </button>
        <button
          onClick={() => setTab('train')}
          className={`grok-toggle-btn ${tab === 'train' ? 'active' : ''}`}
          style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}
        >
          <Cpu size={14} />
          Train LoRA
        </button>
      </div>

      {tab === 'swap' && (
        <SwapPanel user={user} requestLogin={requestLogin} onJobSubmitted={onJobSubmitted} pendingImport={pendingImport} onImportConsumed={onImportConsumed} />
      )}
      {tab === 'profiles' && (
        <ProfilesPanel user={user} requestLogin={requestLogin} />
      )}
      {tab === 'train' && (
        <TrainLoraPanel user={user} requestLogin={requestLogin} />
      )}

      {/* Ethical use notice */}
      <div className="status-banner warning">
        <AlertCircle size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
        <div>
          <strong>Ethical Use:</strong> Only use face swap with consent of all parties involved.
          Creating non-consensual deepfakes is illegal in many jurisdictions.
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// SwapPanel
// ─────────────────────────────────────────────────────────────────────────────

const FACESWAP_DEFAULTS = {
  sourceMode: 'upload', swapAllFaces: false, faceIndex: 0,
}

function SwapPanel({ user, requestLogin, onJobSubmitted, pendingImport, onImportConsumed }) {
  const { initial, save: saveSettings, resetDefaults } = useToolSettings('face_swap', FACESWAP_DEFAULTS)
  const [sourceMode, setSourceMode] = useState(initial.sourceMode) // 'upload' | 'profile'
  const [profiles, setProfiles] = useState([])
  const [selectedProfileId, setSelectedProfileId] = useState(null)

  const [targetFile, setTargetFile] = useState(null)
  const [targetPreview, setTargetPreview] = useState(null)
  const [sourceFile, setSourceFile] = useState(null)
  const [sourcePreview, setSourcePreview] = useState(null)
  const [importModal, setImportModal] = useState(null)
  const [showCreationsPicker, setShowCreationsPicker] = useState(false)

  const [swapAllFaces, setSwapAllFaces] = useState(initial.swapAllFaces)
  const [detectedFaces, setDetectedFaces] = useState(null)
  const [faceIndex, setFaceIndex] = useState(initial.faceIndex)

  // ── Auto-save settings ──────────────────────────────────────────
  const settingsSnapshot = useMemo(() => ({ sourceMode, swapAllFaces, faceIndex }), [sourceMode, swapAllFaces, faceIndex])
  useEffect(() => { saveSettings(settingsSnapshot) }, [settingsSnapshot, saveSettings])

  const handleResetDefaults = useCallback(() => {
    const d = resetDefaults()
    setSourceMode(d.sourceMode); setSwapAllFaces(d.swapAllFaces); setFaceIndex(d.faceIndex)
  }, [resetDefaults])

  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState(null) // { objectUrl }
  const [error, setError] = useState(null)

  const targetInputRef = useRef(null)
  const sourceInputRef = useRef(null)

  // Load profiles so user can pick one as source
  useEffect(() => {
    apiFetch('/api/face-profiles')
      .then(r => r.json())
      .then(d => setProfiles(d.profiles || []))
      .catch(() => {})
  }, [])

  // Auto-open import modal when Dashboard sends a pendingImport
  useEffect(() => {
    if (!pendingImport) return
    setImportModal(pendingImport)
    if (onImportConsumed) onImportConsumed()
  }, [pendingImport])

  const handleApplyImport = async (selected) => {
    if (selected.image && importModal?.item) {
      const item = importModal.item

      // If item is a video, use the companion .png (first frame) instead
      let imageUrl
      if (item.type === 'video' && item.filename?.match(/\.(mp4|webm|mov)$/i)) {
        imageUrl = item.url?.replace(/\.(mp4|webm|mov)$/i, '.png')
        console.debug('🎬 FaceSwap: video detected, using companion image')
      } else {
        imageUrl = getMediaUrl(item.url, item.signed_url)
      }

      try {
        const response = await apiFetch(imageUrl)
        if (!response.ok) throw new Error(`Failed to fetch image: ${response.status}`)
        const blob = await response.blob()
        const filename = imageUrl.split('/').pop() || 'image.png'
        const fileObj = new File([blob], filename, { type: blob.type || 'image/png' })
        setTargetFile(fileObj)
        setTargetPreview(URL.createObjectURL(fileObj))
        setResult(null)
        setError(null)
        setDetectedFaces(null)
        if (DEBUG) console.log('👤 FaceSwap imported target:', filename)
      } catch (e) {
        console.error('Failed to load image from import:', e)
        setError('⚠️ Failed to load image from import')
      }
    }
    setImportModal(null)
  }

  const handleCreationsSelect = useCallback(async (item) => {
    try {
      let mediaUrl
      if (item.type === 'video' && item.filename?.match(/\.(mp4|webm|mov)$/i)) {
        mediaUrl = getMediaUrl(item.url, item.signed_url)
      } else {
        mediaUrl = getMediaUrl(item.url, item.signed_url)
      }
      const response = await apiFetch(mediaUrl)
      if (!response.ok) throw new Error(`Failed to fetch: ${response.status}`)
      const blob = await response.blob()
      const filename = item.filename || mediaUrl.split('/').pop()
      const fileObj = new File([blob], filename, { type: blob.type || 'image/png' })
      setTargetFile(fileObj)
      setTargetPreview(URL.createObjectURL(fileObj))
      setResult(null)
      setError(null)
      setDetectedFaces(null)
      if (DEBUG) console.log('\ud83d\udcc1 FaceSwap: loaded target from creations:', filename)
    } catch (e) {
      console.error('Failed to load from creations:', e)
      setError('\u26a0\ufe0f Failed to load from My Creations')
    }
  }, [])

  const handleDragOver = (e) => e.preventDefault()

  const handleTargetDrop = useCallback((e) => {
    e.preventDefault()
    const f = e.dataTransfer?.files?.[0] || e.target?.files?.[0]
    if (!f) return
    if (!f.type.startsWith('image/') && !f.type.startsWith('video/')) return
    setTargetFile(f)
    setResult(null)
    setError(null)
    setDetectedFaces(null)
    setTargetPreview(URL.createObjectURL(f))
  }, [])

  const handleSourceDrop = useCallback((e) => {
    e.preventDefault()
    const f = e.dataTransfer?.files?.[0] || e.target?.files?.[0]
    if (!f || !f.type.startsWith('image/')) return
    setSourceFile(f)
    setResult(null)
    setError(null)
    setSourcePreview(URL.createObjectURL(f))
  }, [])

  const detectFaces = async () => {
    if (!targetFile || targetFile.type.startsWith('video/')) return
    setIsLoading(true)
    setError(null)
    try {
      const fd = new FormData()
      fd.append('image', targetFile)
      const res = await apiFetch('/detect-faces', { method: 'POST', body: fd })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Detection failed')
      setDetectedFaces(data.faces || [])
      if (DEBUG) console.log('👤 Detected:', data.faces?.length)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const isVideoTarget = targetFile?.type?.startsWith('video/')

  const handleSwap = async () => {
    if (!user) { requestLogin('Log in om face swap te gebruiken'); return }
    if (!targetFile) { setError('Upload a target image or video first'); return }
    if (sourceMode === 'upload' && !sourceFile) { setError('Upload a source face image'); return }
    if (sourceMode === 'profile' && !selectedProfileId) { setError('Select a face profile'); return }

    setIsLoading(true)
    setError(null)
    setResult(null)

    try {
      const idx = swapAllFaces ? '-1' : String(faceIndex)
      const fd = new FormData()
      fd.append('face_indices', idx)

      let endpoint
      if (isVideoTarget) {
        // Video endpoints use field name 'video' instead of 'target'
        fd.append('video', targetFile)
        if (sourceMode === 'profile') {
          fd.append('profile_id', selectedProfileId)
          endpoint = `${BACKEND_BASE}/face-swap-video/profile`
        } else {
          fd.append('source', sourceFile)
          endpoint = `${BACKEND_BASE}/face-swap-video`
        }
      } else {
        fd.append('target', targetFile)
        if (sourceMode === 'profile') {
          fd.append('profile_id', selectedProfileId)
          endpoint = `${BACKEND_BASE}/face-swap/profile`
        } else {
          fd.append('source', sourceFile)
          endpoint = `${BACKEND_BASE}/face-swap`
        }
      }

      if (DEBUG) console.log('👤 FaceSwap →', endpoint, 'indices:', idx, 'video:', isVideoTarget)

      const res = await apiFetch(endpoint, { method: 'POST', body: fd })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
        throw new Error(err.detail || 'Face swap failed')
      }

      const blob = await res.blob()
      const objectUrl = URL.createObjectURL(blob)
      setResult({ objectUrl, isVideo: isVideoTarget })

    } catch (err) {
      console.error('❌ FaceSwap error:', err)
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleDownload = () => {
    if (!result?.objectUrl) return
    const a = document.createElement('a')
    a.href = result.objectUrl
    a.download = result.isVideo ? `faceswap_${Date.now()}.mp4` : `faceswap_${Date.now()}.png`
    a.click()
  }

  const swapInputsAction = () => {
    const tf = targetFile, tp = targetPreview
    setTargetFile(sourceFile); setTargetPreview(sourcePreview)
    setSourceFile(tf); setSourcePreview(tp)
    setResult(null); setDetectedFaces(null)
  }

  const indicesLabel = swapAllFaces ? 'all faces' : `face #${faceIndex}`

  return (
    <div className="tool-container" style={{ gap: '12px' }}>
      {importModal && (
        <MediaImportModal
          item={importModal.item}
          parsedData={importModal.workflow || {}}
          availableFields={['image']}
          onApply={handleApplyImport}
          onClose={() => setImportModal(null)}
        />
      )}

      {/* Source mode toggle */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Source Face</div>
          <ResetDefaultsButton onReset={handleResetDefaults} />
        </div>
        <div className="grok-toggle-group">
          <button
            onClick={() => setSourceMode('upload')}
            className={`grok-toggle-btn ${sourceMode === 'upload' ? 'active' : ''}`}
          >
            Upload photo
          </button>
          <button
            onClick={() => setSourceMode('profile')}
            className={`grok-toggle-btn ${sourceMode === 'profile' ? 'active' : ''}`}
          >
            Saved profile {profiles.length > 0 && `(${profiles.length})`}
          </button>
        </div>
      </div>

      {/* Upload grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        {/* Target */}
        <DropZone
          label="Target (face to replace)"
          preview={targetPreview}
          isVideo={targetFile?.type.startsWith('video/')}
          badge={detectedFaces ? `${detectedFaces.length} face${detectedFaces.length !== 1 ? 's' : ''}` : null}
          onDrop={handleTargetDrop}
          onDragOver={handleDragOver}
          onClick={() => targetInputRef.current?.click()}
          accept="image/*,video/*"
          inputRef={targetInputRef}
          inputOnChange={handleTargetDrop}
          placeholder="Target image / video"
          Icon={Upload}
        />

        {/* Source: upload or profile picker */}
        {sourceMode === 'upload' ? (
          <DropZone
            label="Source (your face)"
            preview={sourcePreview}
            onDrop={handleSourceDrop}
            onDragOver={handleDragOver}
            onClick={() => sourceInputRef.current?.click()}
            accept="image/*"
            inputRef={sourceInputRef}
            inputOnChange={handleSourceDrop}
            placeholder="Source face photo"
            Icon={Smile}
          />
        ) : (
          <ProfilePicker
            profiles={profiles}
            selectedId={selectedProfileId}
            onSelect={setSelectedProfileId}
          />
        )}
      </div>

      <button
        onClick={() => setShowCreationsPicker(true)}
        className="btn-creations-picker"
      >
        📁 Target from My Creations
      </button>

      <CreationsPickerModal
        show={showCreationsPicker}
        onClose={() => setShowCreationsPicker(false)}
        onSelect={handleCreationsSelect}
        title="Select Target for Face Swap"
      />

      {/* Swap target↔source */}
      {sourceMode === 'upload' && (targetFile || sourceFile) && (
        <button
          onClick={swapInputsAction}
          className="btn-secondary"
        >
          <RefreshCw size={14} />
          Swap Target ↔ Source
        </button>
      )}

      {/* Detect faces in target */}
      {targetFile && !targetFile.type.startsWith('video/') && (
        <button
          onClick={detectFaces}
          disabled={isLoading}
          className="btn-secondary"
          style={{ opacity: isLoading ? 0.5 : 1 }}
        >
          <User size={14} />
          Detect Faces in Target
        </button>
      )}

      {/* Face selector (appears after detection) */}
      {detectedFaces && (
        <div className="grok-card">
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={swapAllFaces}
              onChange={e => setSwapAllFaces(e.target.checked)}
              style={{ borderRadius: '4px' }}
            />
            Swap all {detectedFaces.length} detected face{detectedFaces.length !== 1 ? 's' : ''}
          </label>
          {!swapAllFaces && detectedFaces.length > 1 && (
            <div className="grok-toggle-group" style={{ flexWrap: 'wrap', gap: '4px', marginTop: '8px' }}>
              {detectedFaces.map((face, idx) => (
                <button
                  key={idx}
                  onClick={() => setFaceIndex(idx)}
                  className={`grok-toggle-btn ${faceIndex === idx ? 'active' : ''}`}
                  style={{ fontSize: '0.8rem', padding: '6px 12px' }}
                >
                  Face {idx + 1}
                  {face?.confidence && (
                    <span style={{ marginLeft: '4px', opacity: 0.6, fontSize: '0.7rem' }}>{Math.round(face.confidence * 100)}%</span>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="status-banner error">{error}</div>
      )}

      {/* Generate button */}
      <button
        className="primary-btn"
        onClick={handleSwap}
        disabled={
          isLoading ||
          !targetFile ||
          (sourceMode === 'upload' && !sourceFile) ||
          (sourceMode === 'profile' && !selectedProfileId)
        }
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', height: '48px', fontSize: '1rem' }}
      >
        {isLoading ? (
          <><Loader2 size={18} className="animate-spin" /> {isVideoTarget ? 'Processing video...' : 'Swapping...'}</>
        ) : (
          <><User size={18} /> {isVideoTarget ? 'Swap Face in Video' : `Swap Face${detectedFaces ? ` (${indicesLabel})` : ''}`}</>
        )}
      </button>

      {/* Result */}
      {result && (
        <div className="grok-card">
          <div className="grok-card-header">
            <div className="grok-card-title">Result</div>
          </div>
          <div style={{ borderRadius: '8px', overflow: 'hidden', background: '#000' }}>
            {result.isVideo ? (
              <video
                src={result.objectUrl}
                controls
                autoPlay
                loop
                muted
                playsInline
                style={{ width: '100%', maxHeight: '480px', display: 'block' }}
              />
            ) : (
              <img src={result.objectUrl} alt="Face swap result" style={{ width: '100%', display: 'block' }} />
            )}
          </div>
          <button
            className="primary-btn"
            onClick={handleDownload}
            style={{ marginTop: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', height: '40px' }}
          >
            <Download size={16} />
            Download {result.isVideo ? 'Video' : 'Image'}
          </button>
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// ProfilePicker (inside SwapPanel, compact version)
// ─────────────────────────────────────────────────────────────────────────────

function ProfilePicker({ profiles, selectedId, onSelect }) {
  return (
    <div>
      <label className="grok-section-label">Source (saved profile)</label>
      <div className="upload-box" style={{ minHeight: '12rem', maxHeight: '16rem', overflowY: 'auto', padding: '8px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
        {profiles.length === 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '8px', color: 'var(--text-muted)', padding: '16px 0' }}>
            <Users size={24} />
            <span style={{ fontSize: '0.75rem', textAlign: 'center' }}>No profiles yet.<br/>Create one in the Profiles tab.</span>
          </div>
        ) : profiles.map(p => (
          <button
            key={p.id}
            onClick={() => onSelect(p.id)}
            className={`grok-toggle-btn ${selectedId === p.id ? 'active' : ''}`}
            style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 10px', textAlign: 'left', width: '100%' }}
          >
            <User size={14} style={{ flexShrink: 0 }} />
            <div style={{ flex: 1, minWidth: 0, overflow: 'hidden' }}>
              <div style={{ fontWeight: 500, fontSize: '0.75rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.name}</div>
              <div style={{ fontSize: '0.7rem', opacity: 0.6 }}>{p.image_count} photo{p.image_count !== 1 ? 's' : ''}</div>
            </div>
            {selectedId === p.id && <Check size={14} style={{ flexShrink: 0 }} />}
          </button>
        ))}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// ProfilesPanel
// ─────────────────────────────────────────────────────────────────────────────

function ProfilesPanel({ user, requestLogin }) {
  const [profiles, setProfiles] = useState([])
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState(null)

  // Create form state
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [newName, setNewName] = useState('')
  const [newDesc, setNewDesc] = useState('')
  const [newImages, setNewImages] = useState([])
  const [newPreviews, setNewPreviews] = useState([])

  const fileInputRef = useRef(null)

  const loadProfiles = async () => {
    setLoading(true)
    try {
      const res = await apiFetch('/api/face-profiles')
      const data = await res.json()
      setProfiles(data.profiles || [])
    } catch {
      setProfiles([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { loadProfiles() }, [])

  const handleImagePick = (e) => {
    const files = Array.from(e.target.files || [])
    const imgs = files.filter(f => f.type.startsWith('image/'))
    setNewImages(prev => [...prev, ...imgs])
    setNewPreviews(prev => [...prev, ...imgs.map(f => URL.createObjectURL(f))])
    // reset input so same file can be re-added if needed
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const removeImage = (idx) => {
    setNewImages(prev => prev.filter((_, i) => i !== idx))
    setNewPreviews(prev => prev.filter((_, i) => i !== idx))
  }

  const resetForm = () => {
    setNewName('')
    setNewDesc('')
    setNewImages([])
    setNewPreviews([])
    setError(null)
    setShowCreateForm(false)
  }

  const handleCreate = async () => {
    if (!user) { requestLogin('Log in om een face profile te maken'); return }
    if (!newName.trim()) { setError('Profile name is required'); return }
    if (newImages.length === 0) { setError('Upload at least one reference photo'); return }

    setCreating(true)
    setError(null)

    try {
      const fd = new FormData()
      fd.append('name', newName.trim())
      fd.append('description', newDesc.trim())
      newImages.forEach(img => fd.append('images', img))

      const res = await apiFetch('/api/face-profiles', { method: 'POST', body: fd })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Create failed')

      resetForm()
      loadProfiles()

    } catch (err) {
      setError(err.message)
    } finally {
      setCreating(false)
    }
  }

  const handleDelete = async (profileId, profileName) => {
    if (!window.confirm(`Delete face profile "${profileName}"?`)) return
    try {
      const res = await apiFetch(`/api/face-profiles/${profileId}`, { method: 'DELETE' })
      if (!res.ok) throw new Error('Delete failed')
      setProfiles(prev => prev.filter(p => p.id !== profileId))
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div className="tool-container" style={{ gap: '12px' }}>
      {/* Header */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title">Face Profiles</div>
          <button
            onClick={() => setShowCreateForm(v => !v)}
            className="primary-btn"
            style={{ height: '32px', fontSize: '0.8rem', padding: '0 12px', display: 'flex', alignItems: 'center', gap: '6px' }}
          >
            <Plus size={14} />
            New Profile
          </button>
        </div>
        <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Save faces for quick reuse across all generation tools</p>
      </div>

      {/* Create form */}
      {showCreateForm && (
        <div className="grok-card" style={{ borderColor: 'rgba(168, 85, 247, 0.3)' }}>
          <div className="grok-card-header">
            <div className="grok-card-title" style={{ color: '#c084fc' }}>Create New Profile</div>
          </div>

          <div className="form-group">
            <label className="grok-section-label">Name *</label>
            <input
              type="text"
              className="form-input"
              value={newName}
              onChange={e => setNewName(e.target.value)}
              placeholder="e.g. John Doe"
            />
          </div>

          <div className="form-group">
            <label className="grok-section-label">Description (optional)</label>
            <input
              type="text"
              className="form-input"
              value={newDesc}
              onChange={e => setNewDesc(e.target.value)}
              placeholder="e.g. Actor, friend, client…"
            />
          </div>

          {/* Photo upload */}
          <div className="form-group">
            <label className="grok-section-label">Reference Photos *</label>
            <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
              Multiple angles &amp; lighting conditions = better identity accuracy. Embeddings are averaged.
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '8px' }}>
              {newPreviews.map((url, idx) => (
                <div key={idx} style={{ position: 'relative' }}>
                  <img src={url} alt="" style={{ width: '64px', height: '64px', objectFit: 'cover', borderRadius: '6px', border: '1px solid var(--border-color)' }} />
                  <button
                    onClick={() => removeImage(idx)}
                    style={{
                      position: 'absolute', top: '-4px', right: '-4px', width: '20px', height: '20px',
                      background: '#dc2626', borderRadius: '50%', border: 'none',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      color: '#fff', cursor: 'pointer',
                    }}
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              ))}
              <button
                onClick={() => fileInputRef.current?.click()}
                style={{
                  width: '64px', height: '64px', border: '2px dashed var(--border-color)',
                  borderRadius: '6px', background: 'transparent',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  cursor: 'pointer', color: 'var(--text-muted)',
                }}
                title="Add photos"
              >
                <Plus size={20} />
              </button>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              onChange={handleImagePick}
              style={{ display: 'none' }}
            />
            <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{newImages.length} photo{newImages.length !== 1 ? 's' : ''} selected</p>
          </div>

          {error && (
            <div className="status-banner error">{error}</div>
          )}

          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              className="primary-btn"
              onClick={handleCreate}
              disabled={creating || !newName.trim() || newImages.length === 0}
              style={{ flex: 1, height: '40px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}
            >
              {creating
                ? <><Loader2 size={16} className="animate-spin" /> Creating...</>
                : 'Create Profile'
              }
            </button>
            <button
              onClick={resetForm}
              className="btn-secondary"
              style={{ width: 'auto', padding: '0 16px' }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Profiles list */}
      {loading ? (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '32px', color: 'var(--text-muted)', gap: '8px' }}>
          <Loader2 size={20} className="animate-spin" />
          <span style={{ fontSize: '0.85rem' }}>Loading profiles…</span>
        </div>
      ) : profiles.length === 0 ? (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '32px', color: 'var(--text-muted)', gap: '12px' }}>
          <Users size={40} style={{ opacity: 0.3 }} />
          <div style={{ textAlign: 'center' }}>
            <p style={{ fontSize: '0.85rem', fontWeight: 500 }}>No face profiles yet</p>
            <p style={{ fontSize: '0.75rem', marginTop: '4px', color: 'var(--text-muted)' }}>Create a profile to quickly reuse faces across all tools</p>
          </div>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {profiles.map(profile => (
            <ProfileCard
              key={profile.id}
              profile={profile}
              onDelete={() => handleDelete(profile.id, profile.name)}
            />
          ))}
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// ProfileCard
// ─────────────────────────────────────────────────────────────────────────────

function ProfileCard({ profile, onDelete }) {
  return (
    <div className="grok-card" style={{ padding: '12px 16px', display: 'flex', alignItems: 'center', gap: '12px' }}>
      <div style={{
        width: '40px', height: '40px', borderRadius: '50%', background: 'var(--bg-secondary)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
      }}>
        <User size={20} style={{ color: 'var(--text-muted)' }} />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontWeight: 500, fontSize: '0.85rem', color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{profile.name}</div>
        {profile.description && (
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{profile.description}</div>
        )}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: '4px' }}>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <ImageIcon size={12} />
            {profile.image_count} photo{profile.image_count !== 1 ? 's' : ''}
          </span>
          <span className="nav-badge" style={{ fontSize: '0.7rem' }}>
            {profile.embedding_count} embedding{profile.embedding_count !== 1 ? 's' : ''}
          </span>
        </div>
      </div>
      <button
        onClick={onDelete}
        style={{ padding: '8px', color: 'var(--text-muted)', background: 'transparent', border: 'none', cursor: 'pointer', flexShrink: 0 }}
        title="Delete profile"
      >
        <Trash2 size={16} />
      </button>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// TrainLoraPanel
// ─────────────────────────────────────────────────────────────────────────────

const STATUS_COLORS = {
  pending: 'text-yellow-400',
  running: 'text-blue-400',
  done: 'text-green-400',
  failed: 'text-red-400',
  cancelled: 'text-gray-400',
}

const STATUS_ICONS = {
  pending: Clock,
  running: Loader2,
  done: Check,
  failed: AlertCircle,
  cancelled: AlertCircle,
}

function TrainLoraPanel({ user, requestLogin }) {
  const [images, setImages] = useState([])
  const [previews, setPreviews] = useState([])
  const [name, setName] = useState('')
  const [steps, setSteps] = useState(1000)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)
  const [jobs, setJobs] = useState([])
  const [loras, setLoras] = useState([])
  const [copiedTrigger, setCopiedTrigger] = useState(null)

  const fileInputRef = useRef(null)

  const loadStatus = useCallback(async () => {
    try {
      const [jobsRes, lorasRes] = await Promise.all([
        apiFetch('/api/face-train'),
        apiFetch('/api/face-train/loras'),
      ])
      const [jobsData, lorasData] = await Promise.all([jobsRes.json(), lorasRes.json()])
      setJobs((jobsData.jobs || []).slice().reverse()) // newest first
      setLoras(lorasData.loras || [])
    } catch {}
  }, [])

  useEffect(() => {
    loadStatus()
    // Poll while any job is running or pending
    const interval = setInterval(() => {
      loadStatus()
    }, 5000)
    return () => clearInterval(interval)
  }, [loadStatus])

  const handleImagePick = (e) => {
    const files = Array.from(e.target.files || []).filter(f => f.type.startsWith('image/'))
    setImages(prev => [...prev, ...files])
    setPreviews(prev => [...prev, ...files.map(f => URL.createObjectURL(f))])
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const removeImage = (idx) => {
    setImages(prev => prev.filter((_, i) => i !== idx))
    setPreviews(prev => prev.filter((_, i) => i !== idx))
  }

  const handleSubmit = async () => {
    if (!user) { requestLogin('Log in om een face LoRA te trainen'); return }
    if (!name.trim()) { setError('Name is required'); return }
    if (images.length < 2) { setError('Upload at least 2 reference photos'); return }

    setSubmitting(true)
    setError(null)
    try {
      const fd = new FormData()
      fd.append('name', name.trim())
      fd.append('steps', String(steps))
      images.forEach(img => fd.append('images', img))

      const res = await apiFetch('/api/face-train', { method: 'POST', body: fd })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Start failed')

      setName('')
      setImages([])
      setPreviews([])
      loadStatus()
    } catch (err) {
      setError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  const handleCancel = async (jobId) => {
    await apiFetch(`/api/face-train/${jobId}`, { method: 'DELETE' })
    loadStatus()
  }

  const copyTrigger = (trigger) => {
    navigator.clipboard.writeText(trigger)
    setCopiedTrigger(trigger)
    setTimeout(() => setCopiedTrigger(null), 2000)
  }

  const triggerPreview = name.trim()
    ? `ohwx_${name.trim().toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '')}`
    : null

  return (
    <div className="tool-container" style={{ gap: '12px' }}>
      {/* Train new LoRA form */}
      <div className="grok-card" style={{ borderColor: 'rgba(168, 85, 247, 0.3)' }}>
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Cpu size={16} style={{ color: '#c084fc' }} />
            Train New Face LoRA
          </div>
        </div>
        <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '12px' }}>
          Trains a Dreambooth-style SDXL LoRA from your reference photos.
          Use the trigger word in any SDXL prompt to generate images with this person's face.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <div className="form-group">
            <label className="grok-section-label">Person Name *</label>
            <input
              type="text"
              className="form-input"
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="e.g. John Doe"
            />
            {triggerPreview && (
              <p style={{ fontSize: '0.7rem', color: '#c084fc', marginTop: '4px' }}>Trigger: <code>{triggerPreview}</code></p>
            )}
          </div>
          <div className="form-group">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <label className="grok-section-label">Training Steps</label>
              <span className="nav-badge">{steps}</span>
            </div>
            <input
              type="range"
              className="form-range"
              min={500}
              max={2000}
              step={100}
              value={steps}
              onChange={e => setSteps(Number(e.target.value))}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '2px' }}>
              <span>500 (fast)</span>
              <span>2000 (detailed)</span>
            </div>
          </div>
        </div>

        {/* Photo upload */}
        <div className="form-group">
          <label className="grok-section-label">Reference Photos * ({images.length} selected)</label>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '8px' }}>
            {previews.map((url, idx) => (
              <div key={idx} style={{ position: 'relative' }}>
                <img src={url} alt="" style={{ width: '56px', height: '56px', objectFit: 'cover', borderRadius: '6px', border: '1px solid var(--border-color)' }} />
                <button
                  onClick={() => removeImage(idx)}
                  style={{
                    position: 'absolute', top: '-4px', right: '-4px', width: '16px', height: '16px',
                    background: '#dc2626', borderRadius: '50%', border: 'none',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: '#fff', cursor: 'pointer',
                  }}
                >
                  <Trash2 size={10} />
                </button>
              </div>
            ))}
            <button
              onClick={() => fileInputRef.current?.click()}
              style={{
                width: '56px', height: '56px', border: '2px dashed var(--border-color)',
                borderRadius: '6px', background: 'transparent',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                cursor: 'pointer', color: 'var(--text-muted)',
              }}
            >
              <Plus size={16} />
            </button>
          </div>
          <input ref={fileInputRef} type="file" accept="image/*" multiple onChange={handleImagePick} style={{ display: 'none' }} />
          <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Tip: 10–20 varied photos (angles, lighting, expressions) give the best results.</p>
        </div>

        {error && (
          <div className="status-banner error">{error}</div>
        )}

        <button
          className="primary-btn"
          onClick={handleSubmit}
          disabled={submitting || !name.trim() || images.length < 2}
          style={{ height: '42px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}
        >
          {submitting
            ? <><Loader2 size={16} className="animate-spin" /> Starting...</>
            : <><Zap size={16} /> Start Training (~{Math.ceil(steps / 60)} min)</>}
        </button>
      </div>

      {/* Training jobs */}
      {jobs.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <label className="grok-section-label">Training Jobs</label>
          {jobs.map(job => {
            const progress = job.steps_total > 0 ? Math.round((job.steps_done / job.steps_total) * 100) : 0
            const StatusIcon = STATUS_ICONS[job.status] || Clock
            return (
              <div key={job.id} className="grok-card" style={{ padding: '12px 16px' }}>
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '8px' }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <StatusIcon
                        size={14}
                        className={`${STATUS_COLORS[job.status]} ${job.status === 'running' ? 'animate-spin' : ''}`}
                        style={{ flexShrink: 0 }}
                      />
                      <span style={{ fontSize: '0.85rem', fontWeight: 500, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{job.name}</span>
                      <span className={`nav-badge ${STATUS_COLORS[job.status]}`} style={{ fontSize: '0.7rem' }}>{job.status}</span>
                    </div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                      trigger: <code style={{ color: '#c084fc' }}>{job.trigger}</code>
                      {' · '}{job.images_count} photos · {job.steps_total} steps
                    </div>
                    {job.status === 'running' && (
                      <div style={{ marginTop: '8px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
                          <span>Step {job.steps_done} / {job.steps_total}</span>
                          <span>{progress}%</span>
                        </div>
                        <div style={{ width: '100%', background: 'var(--bg-secondary)', borderRadius: '4px', height: '6px' }}>
                          <div style={{ width: `${progress}%`, background: '#a855f7', height: '6px', borderRadius: '4px', transition: 'width 0.5s' }} />
                        </div>
                      </div>
                    )}
                    {job.error && (
                      <div style={{ fontSize: '0.7rem', color: '#fca5a5', marginTop: '4px' }}>{job.error}</div>
                    )}
                  </div>
                  {(job.status === 'pending' || job.status === 'running') && (
                    <button
                      onClick={() => handleCancel(job.id)}
                      style={{ padding: '4px', color: 'var(--text-muted)', background: 'transparent', border: 'none', cursor: 'pointer', flexShrink: 0 }}
                      title="Cancel"
                    >
                      <Trash2 size={14} />
                    </button>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Ready LoRAs */}
      {loras.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <label className="grok-section-label">Trained LoRAs — Ready to Use</label>
          {loras.map(lora => (
            <div key={lora.filename} className="grok-card" style={{ padding: '12px 16px', display: 'flex', alignItems: 'center', gap: '12px', borderColor: 'rgba(34, 197, 94, 0.3)' }}>
              <Check size={16} style={{ color: '#22c55e', flexShrink: 0 }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: '0.85rem', fontWeight: 500, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{lora.filename}</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{lora.size_mb} MB</div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <code className="nav-badge" style={{ fontSize: '0.7rem', fontFamily: 'monospace' }}>{lora.trigger}</code>
                <button
                  onClick={() => copyTrigger(lora.trigger)}
                  style={{ padding: '6px', color: 'var(--text-muted)', background: 'transparent', border: 'none', cursor: 'pointer' }}
                  title="Copy trigger word"
                >
                  {copiedTrigger === lora.trigger
                    ? <CheckCheck size={14} style={{ color: '#22c55e' }} />
                    : <Copy size={14} />}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {jobs.length === 0 && loras.length === 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '32px', color: 'var(--text-muted)', gap: '12px' }}>
          <Cpu size={40} style={{ opacity: 0.3 }} />
          <div style={{ textAlign: 'center' }}>
            <p style={{ fontSize: '0.85rem', fontWeight: 500 }}>No trained LoRAs yet</p>
            <p style={{ fontSize: '0.75rem', marginTop: '4px' }}>Fill in the form above to train your first face LoRA</p>
          </div>
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// DropZone
// ─────────────────────────────────────────────────────────────────────────────

function DropZone({
  label, preview, isVideo, badge, onDrop, onDragOver, onClick,
  accept, inputRef, inputOnChange, placeholder, Icon,
}) {
  return (
    <div>
      <label className="grok-section-label">{label}</label>
      <div
        className="upload-box"
        onClick={onClick}
        onDrop={onDrop}
        onDragOver={onDragOver}
        style={{ minHeight: '10rem', aspectRatio: '1 / 1', cursor: 'pointer', padding: '16px' }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          onChange={inputOnChange}
          style={{ display: 'none' }}
        />
        {preview ? (
          <div style={{ position: 'relative', width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {isVideo ? (
              <video src={preview} style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'cover', borderRadius: '6px' }} muted />
            ) : (
              <img src={preview} alt="Preview" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'cover', borderRadius: '6px' }} />
            )}
            {badge && (
              <div style={{
                position: 'absolute', bottom: '4px', right: '4px',
                background: 'rgba(0,0,0,0.7)', padding: '2px 8px',
                borderRadius: '4px', fontSize: '0.7rem', color: '#fff',
              }}>
                {badge}
              </div>
            )}
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px', color: 'var(--text-muted)' }}>
            <Icon size={24} />
            <span style={{ fontSize: '0.75rem' }}>{placeholder}</span>
          </div>
        )}
      </div>
    </div>
  )
}
