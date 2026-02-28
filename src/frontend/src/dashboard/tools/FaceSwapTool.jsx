import React, { useState, useCallback, useRef, useEffect } from 'react'
import {
  User, Users, Upload, Loader2, Download, AlertCircle,
  Smile, RefreshCw, Trash2, Plus, Check, Image as ImageIcon
} from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { useAuth } from '../../contexts/AuthContext'

// ─────────────────────────────────────────────────────────────────────────────
// FaceSwapTool
// ─────────────────────────────────────────────────────────────────────────────

export default function FaceSwapTool({ onJobSubmitted }) {
  const { user, requestLogin } = useAuth()
  const [tab, setTab] = useState('swap') // 'swap' | 'profiles'

  return (
    <div className="space-y-4">
      {/* Tab pills */}
      <div className="flex gap-2 p-1 bg-gray-800 rounded-lg">
        <button
          onClick={() => setTab('swap')}
          className={`flex-1 py-2 text-sm font-medium rounded transition-colors flex items-center justify-center gap-2 ${
            tab === 'swap'
              ? 'bg-purple-600 text-white'
              : 'text-gray-400 hover:text-gray-200'
          }`}
        >
          <RefreshCw className="w-4 h-4" />
          Face Swap
        </button>
        <button
          onClick={() => setTab('profiles')}
          className={`flex-1 py-2 text-sm font-medium rounded transition-colors flex items-center justify-center gap-2 ${
            tab === 'profiles'
              ? 'bg-purple-600 text-white'
              : 'text-gray-400 hover:text-gray-200'
          }`}
        >
          <Users className="w-4 h-4" />
          Face Profiles
        </button>
      </div>

      {tab === 'swap' && (
        <SwapPanel user={user} requestLogin={requestLogin} onJobSubmitted={onJobSubmitted} />
      )}
      {tab === 'profiles' && (
        <ProfilesPanel user={user} requestLogin={requestLogin} />
      )}

      {/* Ethical use notice */}
      <div className="flex items-start gap-2 p-3 bg-yellow-900/30 border border-yellow-700/50 rounded-lg">
        <AlertCircle className="w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5" />
        <div className="text-sm text-yellow-200">
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

function SwapPanel({ user, requestLogin, onJobSubmitted }) {
  const [sourceMode, setSourceMode] = useState('upload') // 'upload' | 'profile'
  const [profiles, setProfiles] = useState([])
  const [selectedProfileId, setSelectedProfileId] = useState(null)

  const [targetFile, setTargetFile] = useState(null)
  const [targetPreview, setTargetPreview] = useState(null)
  const [sourceFile, setSourceFile] = useState(null)
  const [sourcePreview, setSourcePreview] = useState(null)

  const [swapAllFaces, setSwapAllFaces] = useState(false)
  const [detectedFaces, setDetectedFaces] = useState(null)
  const [faceIndex, setFaceIndex] = useState(0)

  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState(null) // { objectUrl }
  const [error, setError] = useState(null)

  const targetInputRef = useRef(null)
  const sourceInputRef = useRef(null)

  // Load profiles so user can pick one as source
  useEffect(() => {
    fetch(`${BACKEND_BASE}/api/face-profiles`)
      .then(r => r.json())
      .then(d => setProfiles(d.profiles || []))
      .catch(() => {})
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
      const res = await fetch(`${BACKEND_BASE}/detect-faces`, { method: 'POST', body: fd })
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
      fd.append('target', targetFile)
      fd.append('face_indices', idx)

      let endpoint = `${BACKEND_BASE}/face-swap`
      if (sourceMode === 'profile') {
        fd.append('profile_id', selectedProfileId)
        endpoint = `${BACKEND_BASE}/face-swap/profile`
      } else {
        fd.append('source', sourceFile)
      }

      if (DEBUG) console.log('👤 FaceSwap →', endpoint, 'indices:', idx)

      const res = await fetch(endpoint, { method: 'POST', body: fd })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
        throw new Error(err.detail || 'Face swap failed')
      }

      // Backend returns PNG bytes directly via StreamingResponse
      const blob = await res.blob()
      const objectUrl = URL.createObjectURL(blob)
      setResult({ objectUrl })

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
    a.download = `faceswap_${Date.now()}.png`
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
    <div className="space-y-4">
      {/* Source mode toggle */}
      <div>
        <label className="block text-xs font-medium text-gray-400 mb-1 uppercase tracking-wide">
          Source face
        </label>
        <div className="flex gap-2 p-1 bg-gray-800 rounded-lg">
          <button
            onClick={() => setSourceMode('upload')}
            className={`flex-1 py-1.5 text-sm rounded transition-colors ${sourceMode === 'upload' ? 'bg-gray-600 text-white' : 'text-gray-400 hover:text-gray-200'}`}
          >
            Upload photo
          </button>
          <button
            onClick={() => setSourceMode('profile')}
            className={`flex-1 py-1.5 text-sm rounded transition-colors ${sourceMode === 'profile' ? 'bg-gray-600 text-white' : 'text-gray-400 hover:text-gray-200'}`}
          >
            Saved profile {profiles.length > 0 && `(${profiles.length})`}
          </button>
        </div>
      </div>

      {/* Upload grid */}
      <div className="grid grid-cols-2 gap-4">
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
            borderColor="border-blue-600"
          />
        ) : (
          <ProfilePicker
            profiles={profiles}
            selectedId={selectedProfileId}
            onSelect={setSelectedProfileId}
          />
        )}
      </div>

      {/* Swap target↔source */}
      {sourceMode === 'upload' && (targetFile || sourceFile) && (
        <button
          onClick={swapInputsAction}
          className="w-full py-2 bg-gray-700 hover:bg-gray-600 rounded-lg flex items-center justify-center gap-2 text-sm"
        >
          <RefreshCw className="w-4 h-4" />
          Swap Target ↔ Source
        </button>
      )}

      {/* Detect faces in target */}
      {targetFile && !targetFile.type.startsWith('video/') && (
        <button
          onClick={detectFaces}
          disabled={isLoading}
          className="w-full py-2 bg-gray-700 hover:bg-gray-600 rounded-lg flex items-center justify-center gap-2 text-sm disabled:opacity-50"
        >
          <User className="w-4 h-4" />
          Detect Faces in Target
        </button>
      )}

      {/* Face selector (appears after detection) */}
      {detectedFaces && (
        <div className="bg-gray-800 rounded-lg p-3 space-y-2">
          <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
            <input
              type="checkbox"
              checked={swapAllFaces}
              onChange={e => setSwapAllFaces(e.target.checked)}
              className="rounded"
            />
            Swap all {detectedFaces.length} detected face{detectedFaces.length !== 1 ? 's' : ''}
          </label>
          {!swapAllFaces && detectedFaces.length > 1 && (
            <div className="flex gap-2 flex-wrap mt-1">
              {detectedFaces.map((face, idx) => (
                <button
                  key={idx}
                  onClick={() => setFaceIndex(idx)}
                  className={`px-3 py-1 text-sm rounded ${
                    faceIndex === idx ? 'bg-purple-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  Face {idx + 1}
                  {face?.confidence && (
                    <span className="ml-1 opacity-60 text-xs">{Math.round(face.confidence * 100)}%</span>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Generate button */}
      <button
        onClick={handleSwap}
        disabled={
          isLoading ||
          !targetFile ||
          (sourceMode === 'upload' && !sourceFile) ||
          (sourceMode === 'profile' && !selectedProfileId)
        }
        className="w-full py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg font-semibold flex items-center justify-center gap-2 transition-colors"
      >
        {isLoading ? (
          <><Loader2 className="w-5 h-5 animate-spin" /> Swapping...</>
        ) : (
          <><User className="w-5 h-5" /> Swap Face {detectedFaces ? `(${indicesLabel})` : ''}</>
        )}
      </button>

      {/* Error */}
      {error && (
        <div className="p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-200 text-sm">
          {error}
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="space-y-3">
          <div className="rounded-lg overflow-hidden border border-gray-700">
            <img src={result.objectUrl} alt="Face swap result" className="w-full" />
          </div>
          <button
            onClick={handleDownload}
            className="w-full py-2 bg-green-600 hover:bg-green-700 rounded-lg flex items-center justify-center gap-2"
          >
            <Download className="w-4 h-4" />
            Download Result
          </button>
        </div>
      )}

      <div className="text-xs text-gray-500 space-y-1">
        <p>📸 Best results: clear frontal face photo, good lighting, no obstructions.</p>
        <p>⚡ Runs directly — no queue, result appears in seconds.</p>
        <p>💾 Reuse faces often? Save them as a <strong>Face Profile</strong> in the Profiles tab.</p>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// ProfilePicker (inside SwapPanel, compact version)
// ─────────────────────────────────────────────────────────────────────────────

function ProfilePicker({ profiles, selectedId, onSelect }) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-300 mb-2">Source (saved profile)</label>
      <div className="border-2 border-dashed border-gray-600 rounded-lg p-2 space-y-1.5 overflow-y-auto"
        style={{ minHeight: '12rem', maxHeight: '16rem' }}
      >
        {profiles.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-gray-500 py-4">
            <Users className="w-6 h-6" />
            <span className="text-xs text-center">No profiles yet.<br/>Create one in the Profiles tab.</span>
          </div>
        ) : profiles.map(p => (
          <button
            key={p.id}
            onClick={() => onSelect(p.id)}
            className={`w-full flex items-center gap-2 px-2.5 py-2 rounded text-sm text-left transition-colors ${
              selectedId === p.id
                ? 'bg-purple-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            <User className="w-3.5 h-3.5 flex-shrink-0" />
            <div className="truncate flex-1">
              <div className="font-medium truncate text-xs">{p.name}</div>
              <div className="text-xs opacity-60">{p.image_count} photo{p.image_count !== 1 ? 's' : ''}</div>
            </div>
            {selectedId === p.id && <Check className="w-3.5 h-3.5 flex-shrink-0" />}
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
      const res = await fetch(`${BACKEND_BASE}/api/face-profiles`)
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

      const res = await fetch(`${BACKEND_BASE}/api/face-profiles`, { method: 'POST', body: fd })
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
      const res = await fetch(`${BACKEND_BASE}/api/face-profiles/${profileId}`, { method: 'DELETE' })
      if (!res.ok) throw new Error('Delete failed')
      setProfiles(prev => prev.filter(p => p.id !== profileId))
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-gray-200">Face Profiles</h3>
          <p className="text-xs text-gray-400">Save faces for quick reuse across all generation tools</p>
        </div>
        <button
          onClick={() => setShowCreateForm(v => !v)}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-600 hover:bg-purple-700 rounded-lg text-sm font-medium transition-colors"
        >
          <Plus className="w-4 h-4" />
          New Profile
        </button>
      </div>

      {/* Create form */}
      {showCreateForm && (
        <div className="border border-purple-700/50 bg-gray-800/50 rounded-lg p-4 space-y-3">
          <h4 className="text-sm font-semibold text-purple-300">Create New Profile</h4>

          <div>
            <label className="block text-xs text-gray-400 mb-1">Name *</label>
            <input
              type="text"
              value={newName}
              onChange={e => setNewName(e.target.value)}
              placeholder="e.g. John Doe"
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-purple-500"
            />
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1">Description (optional)</label>
            <input
              type="text"
              value={newDesc}
              onChange={e => setNewDesc(e.target.value)}
              placeholder="e.g. Actor, friend, client…"
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-purple-500"
            />
          </div>

          {/* Photo upload */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">Reference Photos *</label>
            <p className="text-xs text-gray-500 mb-2">
              Multiple angles &amp; lighting conditions = better identity accuracy. Embeddings are averaged.
            </p>
            <div className="flex flex-wrap gap-2 mb-2">
              {newPreviews.map((url, idx) => (
                <div key={idx} className="relative">
                  <img src={url} alt="" className="w-16 h-16 object-cover rounded border border-gray-600" />
                  <button
                    onClick={() => removeImage(idx)}
                    className="absolute -top-1 -right-1 w-5 h-5 bg-red-600 rounded-full flex items-center justify-center text-white hover:bg-red-700"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              ))}
              <button
                onClick={() => fileInputRef.current?.click()}
                className="w-16 h-16 border-2 border-dashed border-gray-600 rounded flex items-center justify-center hover:border-purple-500 transition-colors"
                title="Add photos"
              >
                <Plus className="w-5 h-5 text-gray-400" />
              </button>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              onChange={handleImagePick}
              className="hidden"
            />
            <p className="text-xs text-gray-500">{newImages.length} photo{newImages.length !== 1 ? 's' : ''} selected</p>
          </div>

          {error && (
            <div className="p-2 bg-red-900/50 border border-red-700 rounded text-red-200 text-xs">{error}</div>
          )}

          <div className="flex gap-2">
            <button
              onClick={handleCreate}
              disabled={creating || !newName.trim() || newImages.length === 0}
              className="flex-1 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded font-medium text-sm flex items-center justify-center gap-2"
            >
              {creating
                ? <><Loader2 className="w-4 h-4 animate-spin" /> Creating...</>
                : 'Create Profile'
              }
            </button>
            <button
              onClick={resetForm}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Profiles list */}
      {loading ? (
        <div className="flex items-center justify-center p-8 text-gray-400 gap-2">
          <Loader2 className="w-5 h-5 animate-spin" />
          <span className="text-sm">Loading profiles…</span>
        </div>
      ) : profiles.length === 0 ? (
        <div className="flex flex-col items-center justify-center p-8 text-gray-500 gap-3">
          <Users className="w-10 h-10 opacity-30" />
          <div className="text-center">
            <p className="text-sm font-medium">No face profiles yet</p>
            <p className="text-xs mt-1 text-gray-500">Create a profile to quickly reuse faces across all tools</p>
          </div>
        </div>
      ) : (
        <div className="space-y-2">
          {profiles.map(profile => (
            <ProfileCard
              key={profile.id}
              profile={profile}
              onDelete={() => handleDelete(profile.id, profile.name)}
            />
          ))}
        </div>
      )}

      {!loading && profiles.length > 0 && (
        <div className="text-xs text-gray-500 space-y-1">
          <p>🧠 Averaged embeddings across all photos = better identity stability.</p>
          <p>📸 5–15 photos from varied angles &amp; expressions gives the best results.</p>
          <p>🎬 Profiles work in Face Swap and (soon) Lynx video identity injection.</p>
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
    <div className="flex items-center gap-3 p-3 bg-gray-800 rounded-lg border border-gray-700 hover:border-gray-600 transition-colors">
      <div className="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center flex-shrink-0">
        <User className="w-5 h-5 text-gray-400" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="font-medium text-sm text-gray-200 truncate">{profile.name}</div>
        {profile.description && (
          <div className="text-xs text-gray-400 truncate">{profile.description}</div>
        )}
        <div className="flex items-center gap-3 mt-1">
          <span className="text-xs text-gray-500 flex items-center gap-1">
            <ImageIcon className="w-3 h-3" />
            {profile.image_count} photo{profile.image_count !== 1 ? 's' : ''}
          </span>
          <span className="text-xs text-purple-400">
            {profile.embedding_count} embedding{profile.embedding_count !== 1 ? 's' : ''}
          </span>
        </div>
      </div>
      <button
        onClick={onDelete}
        className="p-2 text-gray-500 hover:text-red-400 transition-colors flex-shrink-0"
        title="Delete profile"
      >
        <Trash2 className="w-4 h-4" />
      </button>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// DropZone
// ─────────────────────────────────────────────────────────────────────────────

function DropZone({
  label, preview, isVideo, badge, onDrop, onDragOver, onClick,
  accept, inputRef, inputOnChange, placeholder, Icon, borderColor = 'border-gray-600',
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-300 mb-2">{label}</label>
      <div
        onClick={onClick}
        onDrop={onDrop}
        onDragOver={onDragOver}
        className={`border-2 border-dashed ${borderColor} rounded-lg p-4 text-center cursor-pointer hover:border-purple-500 transition-colors flex items-center justify-center`}
        style={{ minHeight: '10rem', aspectRatio: '1 / 1' }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          onChange={inputOnChange}
          className="hidden"
        />
        {preview ? (
          <div className="relative w-full h-full flex items-center justify-center">
            {isVideo ? (
              <video src={preview} className="max-w-full max-h-full object-cover rounded" muted />
            ) : (
              <img src={preview} alt="Preview" className="max-w-full max-h-full object-cover rounded" />
            )}
            {badge && (
              <div className="absolute bottom-1 right-1 bg-black/70 px-2 py-1 rounded text-xs text-white">
                {badge}
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2 text-gray-400">
            <Icon className="w-6 h-6" />
            <span className="text-xs">{placeholder}</span>
          </div>
        )}
      </div>
    </div>
  )
}
