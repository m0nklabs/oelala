import React, { useState, useCallback, useEffect, useMemo } from 'react'
import { Upload, Wand2, Copy, Send, Loader2, Image as ImageIcon, Pencil, ChevronDown, RotateCcw, Search, Volume2, MessageCircle, Video, Plus, X, Sparkles } from 'lucide-react'
import { BACKEND_BASE, DEBUG, getMediaUrl } from '../../config'
import { apiFetch } from '../../api'
import { extractVideoFirstFrame } from '../../utils/mediaUtils'
import { storeImage, getImage, removeImage } from '../../utils/imageStore'
import MediaImportModal from '../../components/MediaImportModal'
import CreationsPickerModal from '../../components/CreationsPickerModal'

import { VISION_MODELS, DEFAULT_VISION_MODEL } from '../../constants/llmModels'
import { useAuth } from '../../contexts/AuthContext'
import { useToolSettings } from '../../hooks/useToolSettings'
import ResetDefaultsButton from '../../components/ResetDefaultsButton'
import CameraMotionSelector, { getCameraMotionPrefix } from '../../components/CameraMotionSelector'
import { TOOL_IDS } from '../nav'

const CAPTION_MODES = [
  { id: 'brief', label: 'Brief', description: '1-line summary', group: 'caption' },
  { id: 'detailed', label: 'Detailed', description: 'Full paragraph', group: 'caption' },
  { id: 'tags', label: 'Tags', description: 'Comma-separated keywords', group: 'caption' },
  { id: 'structured', label: 'Structured', description: 'Subject, style, mood', group: 'caption' },
  { id: 'prompt_i2v', label: '🎬 I2V Prompt', description: 'Motion & activity for video gen', group: 'prompt' },
  { id: 'prompt_t2i', label: '🖼️ T2I Prompt', description: 'Tag-style for image gen', group: 'prompt' },
  { id: 'prompt_nsfw', label: '🔞 NSFW Prompt', description: 'Explicit & uncensored', group: 'prompt' },
]

const NSFW_LEVELS = [
  { value: 1, label: 'Sensual', emoji: '💋', description: 'Suggestive poses, lingerie, soft lighting' },
  { value: 2, label: 'Softcore', emoji: '🔥', description: 'Partial nudity, erotic posing, teasing' },
  { value: 3, label: 'Nude', emoji: '🍑', description: 'Full nudity, explicit body details' },
  { value: 4, label: 'Hardcore', emoji: '🍆', description: 'Explicit sexual acts, positions' },
  { value: 5, label: 'Extreme', emoji: '⛓️', description: 'Rough, aggressive, extreme acts' },
]

const isPromptMode = (m) => m.startsWith('prompt_')

// VISION_MODELS imported from shared constants/llmModels.js
const MODELS = VISION_MODELS

const I2T_DEFAULTS = {
  model: DEFAULT_VISION_MODEL, mode: 'detailed', nsfwIntensity: 3,
  cameraMotion: '', detailLevel: 3, includeMotion: true, motionHint: '',
}

const SEND_TO_TOOLS = [
  { id: TOOL_IDS.IMAGE_TO_VIDEO, label: '🎬 Image to Video', icon: '🎬' },
  { id: TOOL_IDS.TEXT_TO_VIDEO, label: '🎥 Text to Video', icon: '🎥' },
  { id: TOOL_IDS.TEXT_TO_IMAGE, label: '🖼️ Text to Image', icon: '🖼️' },
  { id: TOOL_IDS.IMAGE_TO_IMAGE, label: '🎨 Image to Image', icon: '🎨' },
]

export default function ImageToTextTool({ onSendToTool, pendingImport = null, onImportConsumed = null }) {
  const { user, requestLogin } = useAuth()
  const { initial, save: saveSettings, resetDefaults } = useToolSettings('image_to_text', I2T_DEFAULTS)

  // ── Restore session state (for back-navigation) ────────────────
  const savedSession = useMemo(() => {
    try {
      const raw = sessionStorage.getItem('i2t_session')
      return raw ? JSON.parse(raw) : null
    } catch { return null }
  }, [])

  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(
    savedSession?.preview && savedSession.preview !== '__idb__' && !savedSession.preview.startsWith('blob:')
      ? savedSession.preview : null
  )
  const [model, setModel] = useState(initial.model)
  const [mode, setMode] = useState(initial.mode)
  const [caption, setCaption] = useState(savedSession?.caption || '')
  const [negativePrompt, setNegativePrompt] = useState(savedSession?.negativePrompt || '')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [isRefining, setIsRefining] = useState(false)
  const [showRefineInput, setShowRefineInput] = useState(false)
  const [refineInstruction, setRefineInstruction] = useState('')
  const [importModal, setImportModal] = useState(null)  // { item, workflow }
  const [showCreationsPicker, setShowCreationsPicker] = useState(false)
  const [nsfwIntensity, setNsfwIntensity] = useState(initial.nsfwIntensity)  // 1-5 NSFW intensity scale
  const [cameraMotion, setCameraMotion] = useState(initial.cameraMotion || '')
  const [detailLevel, setDetailLevel] = useState(initial.detailLevel ?? 3)
  const [includeMotion, setIncludeMotion] = useState(initial.includeMotion ?? true)
  const [motionHint, setMotionHint] = useState(initial.motionHint || '')

  // ── Concept Studio state ────────────────────────────────────────
  const [concept, setConcept] = useState(savedSession?.concept || null)           // structured analysis from Analyze step
  const [analyzing, setAnalyzing] = useState(false)      // loading state for Analyze
  const [directorAudio, setDirectorAudio] = useState(savedSession?.directorAudio || '') // ambient sound notes
  const [dialogueLines, setDialogueLines] = useState(savedSession?.dialogueLines || []) // [{subject, line}]
  const [audioPrompt, setAudioPrompt] = useState(savedSession?.audioPrompt || '')     // generated audio prompt output
  const [cameraDirection, setCameraDirection] = useState(savedSession?.cameraDirection || '') // LLM-driven camera direction text
  const [conceptRefineText, setConceptRefineText] = useState('') // refinement prompt for concept
  const [notesRefineText, setNotesRefineText] = useState('')     // refinement prompt for notes
  const [refiningConcept, setRefiningConcept] = useState(false)  // loading state for concept refine
  const [refiningNotes, setRefiningNotes] = useState(false)      // loading state for notes refine
  const [restoringFile, setRestoringFile] = useState(!!savedSession?.preview) // true while restoring file from session

  // ── Restore file from session (IndexedDB for blobs, server URL fallback) ──
  useEffect(() => {
    if (!savedSession?.preview) { setRestoringFile(false); return }
    const restoreFile = async () => {
      try {
        let blob = null
        if (savedSession.preview === '__idb__') {
          // Restore from IndexedDB (reliable, no size limit)
          blob = await getImage('i2t_preview')
        } else if (savedSession.preview.startsWith('data:')) {
          // Legacy: data URL in sessionStorage (kept for backward compat)
          const resp = await fetch(savedSession.preview)
          blob = await resp.blob()
        } else if (savedSession.preview.startsWith('/')) {
          // Server URL: fetch via API
          const resp = await apiFetch(savedSession.preview)
          if (resp.ok) blob = await resp.blob()
        }
        if (blob) {
          const f = blob instanceof File ? blob : new File([blob], savedSession.previewFilename || 'restored.png', { type: blob.type || 'image/png' })
          setFile(f)
          setPreview(URL.createObjectURL(f))
        }
      } catch (e) {
        console.warn('Failed to restore I2T file from session:', e)
      } finally {
        setRestoringFile(false)
        removeImage('i2t_preview').catch(() => {})
      }
    }
    restoreFile()
    sessionStorage.removeItem('i2t_session')
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Auto-save settings ──────────────────────────────────────────
  const settingsSnapshot = useMemo(() => ({
    model, mode, nsfwIntensity, cameraMotion, detailLevel, includeMotion, motionHint,
  }), [model, mode, nsfwIntensity, cameraMotion, detailLevel, includeMotion, motionHint])
  useEffect(() => { saveSettings(settingsSnapshot) }, [settingsSnapshot, saveSettings])

  const handleResetDefaults = useCallback(() => {
    const d = resetDefaults()
    setModel(d.model); setMode(d.mode); setNsfwIntensity(d.nsfwIntensity)
    setCameraMotion(d.cameraMotion || ''); setDetailLevel(d.detailLevel ?? 3)
    setIncludeMotion(d.includeMotion ?? true); setMotionHint(d.motionHint || '')
  }, [resetDefaults])

  // Auto-open import modal when Dashboard sends a pendingImport
  useEffect(() => {
    if (!pendingImport) return
    setImportModal(pendingImport)
    if (onImportConsumed) onImportConsumed()
  }, [pendingImport])

  const handleApplyImport = async (selected) => {
    if (selected.image && importModal?.item) {
      const item = importModal.item

      // If item is a video, extract first frame client-side
      if (item.type === 'video' && item.filename?.match(/\.(mp4|webm|mov)$/i)) {
        try {
          const fetchUrl = item.signed_url || (item.url?.startsWith('/') ? item.url : `/${item.url}`)
          console.debug('🎬 I2T: video detected, extracting first frame from:', item.filename)
          const { file: fileObj, previewUrl } = await extractVideoFirstFrame(apiFetch, fetchUrl, item.filename)
          setFile(fileObj)
          setPreview(previewUrl)
          setCaption('')
          setError(null)
          if (DEBUG) console.log('🖼️ Extracted first frame from video:', fileObj.name)
        } catch (e) {
          console.error('Failed to extract frame from video:', e)
          setError('⚠️ Failed to extract first frame from video')
        }
      } else {
        const imageUrl = getMediaUrl(item.url, item.signed_url)
        const fetchUrl = item.signed_url || (item.url?.startsWith('/') ? item.url : `/${item.url}`)
        const imageFilename = item.filename || item.url?.split('/').pop() || 'image.png'
        try {
          const response = await apiFetch(fetchUrl)
          const blob = await response.blob()
          const fileObj = new File([blob], imageFilename, { type: blob.type || 'image/png' })
          setFile(fileObj)
          setPreview(imageUrl)
          setCaption('')
          setError(null)
          if (DEBUG) console.log('🖼️ Imported image from creations:', imageFilename)
        } catch (e) {
          console.error('Failed to load image from import:', e)
          setError('⚠️ Failed to load image from import')
        }
      }
    }
    setImportModal(null)
  }

  const handleCreationsSelect = useCallback(async (item) => {
    try {
      if (item.type === 'video' && item.filename?.match(/\.(mp4|webm|mov)$/i)) {
        // Extract first frame from video client-side
        const fetchUrl = item.signed_url || (item.url?.startsWith('/') ? item.url : `/${item.url}`)
        console.debug('🎬 I2T creations: extracting first frame from video:', item.filename)
        const { file: fileObj, previewUrl } = await extractVideoFirstFrame(apiFetch, fetchUrl, item.filename)
        setFile(fileObj)
        setPreview(previewUrl)
        setCaption('')
        setError(null)
        if (DEBUG) console.log('📁 I2T: extracted frame from video:', fileObj.name)
      } else {
        const imageUrl = getMediaUrl(item.url, item.signed_url)
        const response = await apiFetch(imageUrl)
        if (!response.ok) throw new Error(`Failed to fetch: ${response.status}`)
        const blob = await response.blob()
        const filename = imageUrl.split('/').pop() || 'image.png'
        const fileObj = new File([blob], filename, { type: blob.type || 'image/png' })
        setFile(fileObj)
        setPreview(imageUrl)
        setCaption('')
        setError(null)
        if (DEBUG) console.log('📁 I2T: loaded from creations:', filename)
      }
    } catch (e) {
      console.error('Failed to load from creations:', e)
      setError('\u26a0\ufe0f Failed to load image from My Creations')
    }
  }, [])

  const handleFileChange = useCallback((e) => {
    const f = e.target.files?.[0]
    if (f) {
      setFile(f)
      setPreview(URL.createObjectURL(f))
      setCaption('')
      setConcept(null)
      setError(null)
    }
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0]
    if (f && f.type.startsWith('image/')) {
      setFile(f)
      setPreview(URL.createObjectURL(f))
      setCaption('')
      setConcept(null)
      setError(null)
    }
  }, [])

  // ── Step 1: Analyze image → structured concept ──────────────────
  const handleAnalyze = async () => {
    if (!user) { requestLogin('Log in om afbeeldingen te analyseren'); return }
    if (!file) return

    setAnalyzing(true)
    setError(null)
    setConcept(null)

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('model', model)
      formData.append('mode', 'concept')
      formData.append('detail_level', detailLevel.toString())

      const res = await apiFetch(`${BACKEND_BASE}/caption-image`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Analysis failed')
      }

      const data = await res.json()
      const c = data.concept || {}
      setConcept(c)

      // Pre-fill director's notes from suggestions
      if (c.suggested_motion) setMotionHint(c.suggested_motion)
      if (c.suggested_audio) setDirectorAudio(c.suggested_audio)
      if (c.suggested_dialogue?.length) setDialogueLines(c.suggested_dialogue)
      if (c.suggested_camera) setCameraDirection(c.suggested_camera)

      if (DEBUG) console.log('🔍 Concept analysis:', c)
    } catch (err) {
      console.error('Analyze error:', err)
      setError(err.message)
    } finally {
      setAnalyzing(false)
    }
  }

  // ── Refine concept or director's notes via LLM ─────────────────
  const handleRefine = async (target) => {
    if (!user || !file || !concept) return

    const prompt = target === 'notes' ? notesRefineText : conceptRefineText
    if (!prompt.trim()) return

    const setRefining = target === 'notes' ? setRefiningNotes : setRefiningConcept
    setRefining(true)
    setError(null)

    try {
      // Build current concept with live director's notes merged in
      const currentConcept = {
        ...concept,
        suggested_motion: motionHint,
        suggested_audio: directorAudio,
        suggested_dialogue: dialogueLines,
        suggested_camera: cameraDirection,
      }

      const formData = new FormData()
      formData.append('file', file)
      formData.append('model', model)
      formData.append('mode', 'concept')
      formData.append('detail_level', detailLevel.toString())
      formData.append('concept_context', JSON.stringify(currentConcept))
      formData.append('refinement_prompt', prompt.trim())
      formData.append('refinement_target', target)

      const res = await apiFetch(`${BACKEND_BASE}/caption-image`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Refinement failed')
      }

      const data = await res.json()
      const c = data.concept || {}
      setConcept(c)

      // Update director's notes from refined suggestions
      if (c.suggested_motion) setMotionHint(c.suggested_motion)
      if (c.suggested_audio) setDirectorAudio(c.suggested_audio)
      if (c.suggested_dialogue?.length) setDialogueLines(c.suggested_dialogue)
      if (c.suggested_camera) setCameraDirection(c.suggested_camera)

      // Clear the refinement input after success
      if (target === 'notes') setNotesRefineText('')
      else setConceptRefineText('')

      if (DEBUG) console.log(`✨ Concept refined (${target}):`, c)
    } catch (err) {
      console.error('Refine error:', err)
      setError(err.message)
    } finally {
      setRefining(false)
    }
  }

  // ── Step 2: Generate production prompt ──────────────────────────
  const handleCaption = async () => {
    if (!user) { requestLogin('Log in om afbeeldingen te analyseren'); return }
    if (!file) return

    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('model', model)
      formData.append('mode', mode)
      if (mode === 'prompt_nsfw') {
        formData.append('nsfw_intensity', nsfwIntensity.toString())
      }
      formData.append('detail_level', detailLevel.toString())
      if (isPromptMode(mode)) formData.append('include_negative', 'true')
      if (isPromptMode(mode)) formData.append('include_motion', includeMotion.toString())
      if (isPromptMode(mode) && includeMotion && motionHint.trim()) {
        formData.append('motion_hint', motionHint.trim())
      }

      // Pass concept context if we did an Analyze step
      // Merge in live director's notes (user may have edited cameraDirection etc.)
      if (concept && isPromptMode(mode)) {
        const liveContext = {
          ...concept,
          suggested_camera: cameraDirection || concept.suggested_camera,
          suggested_motion: motionHint || concept.suggested_motion,
          suggested_audio: directorAudio || concept.suggested_audio,
          suggested_dialogue: dialogueLines?.length ? dialogueLines : concept.suggested_dialogue,
        }
        formData.append('concept_context', JSON.stringify(liveContext))
      }

      // Pass audio context if director's notes have audio/dialogue
      const hasAudioNotes = directorAudio.trim() || dialogueLines.some(d => d.line?.trim())
      if (hasAudioNotes && isPromptMode(mode)) {
        formData.append('audio_context', JSON.stringify({
          ambient: directorAudio.trim(),
          dialogue: dialogueLines.filter(d => d.line?.trim()),
        }))
      }

      const res = await apiFetch(`${BACKEND_BASE}/caption-image`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Caption failed')
      }

      const data = await res.json()
      let captionText = data.caption || ''

      // Prepend camera motion prefix when in prompt mode and motion is selected
      if (isPromptMode(mode) && cameraMotion) {
        const motionPrefix = getCameraMotionPrefix(cameraMotion)
        if (motionPrefix) captionText = motionPrefix + captionText
      }
      // Prepend LLM-driven camera direction if present and not already in caption
      const camDir = cameraDirection?.trim()
      if (isPromptMode(mode) && camDir && !captionText.toLowerCase().includes(camDir.toLowerCase().slice(0, 20))) {
        captionText = camDir + ', ' + captionText
      }
      setCaption(captionText)
      setNegativePrompt(data.negative_prompt || '')
      if (data.audio_prompt) setAudioPrompt(data.audio_prompt)

      if (DEBUG) console.log('🖼️ Caption result:', data)
    } catch (err) {
      console.error('Caption error:', err)
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Refine/tweak all outputs with user suggestion via dedicated endpoint
  const handleRefineCaption = async () => {
    if (!caption.trim() || isRefining || !refineInstruction.trim()) return
    setIsRefining(true)
    setError(null)

    try {
      const res = await apiFetch(`${BACKEND_BASE}/refine-caption`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          positive: caption.trim(),
          negative: negativePrompt || null,
          suggestion: refineInstruction.trim(),
        }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Refine failed')
      }

      const data = await res.json()
      if (data.positive) setCaption(data.positive)
      if (data.negative !== undefined) setNegativePrompt(data.negative || '')
      setRefineInstruction('')
      setShowRefineInput(false)
    } catch (err) {
      console.error('Refine error:', err)
      setError(`Refine failed: ${err.message}`)
    } finally {
      setIsRefining(false)
    }
  }

  const handleCopyPositive = () => { if (caption) navigator.clipboard.writeText(caption) }
  const handleCopyNegative = () => { if (negativePrompt) navigator.clipboard.writeText(negativePrompt) }

  const [showSendMenu, setShowSendMenu] = useState(false)

  // ── Save session state & send to tool ───────────────────────────
  const handleSendToTool = (toolId) => {
    if (caption && onSendToTool) {
      // Save I2T state to sessionStorage so user can navigate back
      try {
        // Store image blob in IndexedDB (no 5MB sessionStorage limit)
        if (file) {
          storeImage('i2t_preview', file).catch(() => {})
        }
        sessionStorage.setItem('i2t_session', JSON.stringify({
          caption,
          negativePrompt,
          preview: preview && !preview.startsWith('blob:') ? preview : (file ? '__idb__' : null),
          previewFilename: file?.name || null,
          audioPrompt: audioPrompt || null,
          concept: concept || null,
          directorAudio: directorAudio || null,
          dialogueLines: dialogueLines?.length ? dialogueLines : null,
          cameraDirection: cameraDirection || null,
        }))
      } catch { /* quota exceeded — best effort */ }

      // Build final positive prompt — ensure camera direction is included
      let finalPositive = caption
      const camDir = cameraDirection?.trim()
      if (camDir && !finalPositive.toLowerCase().includes(camDir.toLowerCase().slice(0, 30))) {
        finalPositive = camDir + ', ' + finalPositive
      }

      onSendToTool(toolId, {
        item: {
          filename: file?.name || 'i2t-image.png',
          type: 'image',
          url: preview,
          _file: file,  // direct File blob for tools that support it
        },
        workflow: {
          positive: finalPositive,
          negative: negativePrompt || undefined,
          audio: audioPrompt || undefined,
        },
      })
      setShowSendMenu(false)
    }
  }

  // ── Clear all state ─────────────────────────────────────────────
  const handleClearAll = useCallback(() => {
    setFile(null)
    setPreview(null)
    setCaption('')
    setNegativePrompt('')
    setAudioPrompt('')
    setConcept(null)
    setDirectorAudio('')
    setDialogueLines([])
    setConceptRefineText('')
    setNotesRefineText('')
    setError(null)
    setShowRefineInput(false)
    setRefineInstruction('')
    setShowSendMenu(false)
    sessionStorage.removeItem('i2t_session')
  }, [])

  return (
    <div className="tool-container">

      {/* Import from previous generation modal */}
      {importModal && (
        <MediaImportModal
          item={importModal.item}
          parsedData={importModal.workflow}
          availableFields={['image']}
          onApply={handleApplyImport}
          onClose={() => setImportModal(null)}
        />
      )}

      {/* Upload Image Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <ImageIcon size={16} />
            Upload Image
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            {(file || caption) && (
              <button
                className="icon-btn"
                onClick={handleClearAll}
                title="Clear all & start fresh"
                style={{
                  display: 'flex', alignItems: 'center', gap: '4px',
                  padding: '4px 10px', fontSize: '0.72rem', fontWeight: 600,
                  color: 'var(--text-muted, #888)', background: 'transparent',
                  border: '1px solid var(--border-color, #444)', borderRadius: '6px',
                  cursor: 'pointer', transition: 'all 0.15s',
                }}
              >
                <RotateCcw size={12} />
                Clear
              </button>
            )}
            <ResetDefaultsButton onReset={handleResetDefaults} />
          </div>
        </div>

        <div
          className={`upload-dropzone ${preview ? 'has-preview' : ''}`}
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => document.getElementById('i2t-file-input').click()}
        >
          {preview ? (
            <img src={preview} alt="Preview" className="upload-preview" />
          ) : (
            <div className="upload-placeholder">
              <Upload size={32} />
              <p>Drop image here or click to upload</p>
            </div>
          )}
          <input
            id="i2t-file-input"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
        </div>

        <button
          onClick={() => setShowCreationsPicker(true)}
          className="btn-creations-picker"
        >
          {'📁'} From My Creations
        </button>

        <CreationsPickerModal
          show={showCreationsPicker}
          onClose={() => setShowCreationsPicker(false)}
          onSelect={handleCreationsSelect}
          filter="image"
          title="Select Image for Captioning"
        />
      </div>

      {/* ── Step 1: Analyze Button ────────────────────────────────── */}
      {file && !concept && (
        <button
          className="primary-btn"
          onClick={handleAnalyze}
          disabled={analyzing}
          style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
        >
          {analyzing ? (
            <>
              <Loader2 size={18} className="spin" />
              Analyzing image...
            </>
          ) : (
            <>
              <Search size={18} />
              🔍 Analyze Image
            </>
          )}
        </button>
      )}

      {/* ── Concept Card — shows after analysis ───────────────────── */}
      {concept && (
        <div className="grok-card">
          <div className="grok-card-header">
            <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Search size={16} />
              Concept Analysis
            </div>
            <button
              className="icon-btn"
              onClick={handleAnalyze}
              disabled={analyzing}
              title="Re-analyze"
              style={{
                display: 'flex', alignItems: 'center', gap: '4px',
                padding: '4px 10px', fontSize: '0.72rem', fontWeight: 600,
                color: 'var(--text-muted, #888)', background: 'transparent',
                border: '1px solid var(--border-color, #444)', borderRadius: '6px',
                cursor: 'pointer',
              }}
            >
              {analyzing ? <Loader2 size={12} className="spin" /> : <RotateCcw size={12} />}
              Re-analyze
            </button>
          </div>

          {/* Scene */}
          {concept.scene && (
            <div className="i2t-concept-field">
              <span className="i2t-concept-label">🎬 Scene</span>
              <p className="i2t-concept-text">{concept.scene}</p>
            </div>
          )}

          {/* Subjects */}
          {concept.subjects?.length > 0 && (
            <div className="i2t-concept-field">
              <span className="i2t-concept-label">👤 Subjects</span>
              {concept.subjects.map((s, i) => (
                <div key={i} className="i2t-concept-subject">
                  <strong>{s.label}</strong>
                  <span style={{ color: 'var(--text-muted, #888)', fontSize: '0.78rem' }}>
                    {s.position && ` (${s.position})`}
                  </span>
                  <p style={{ margin: '2px 0 0', fontSize: '0.82rem', color: 'var(--text-secondary, #ccc)' }}>
                    {s.description}
                  </p>
                </div>
              ))}
            </div>
          )}

          {/* Mood */}
          {concept.mood && (
            <div className="i2t-concept-field">
              <span className="i2t-concept-label">🎭 Mood</span>
              <p className="i2t-concept-text">{concept.mood}</p>
            </div>
          )}

          {/* ── Concept Refinement Input ──────────────────────────── */}
          <div className="i2t-refine-row">
            <input
              type="text"
              value={conceptRefineText}
              onChange={(e) => setConceptRefineText(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !refiningConcept) handleRefine('concept') }}
              placeholder="Refine concept… e.g. 'make it more dramatic' or 'focus on background details'"
              className="i2t-refine-input"
              disabled={refiningConcept}
            />
            <button
              className="i2t-refine-btn"
              onClick={() => handleRefine('concept')}
              disabled={refiningConcept || !conceptRefineText.trim()}
              title="Refine concept with AI"
            >
              {refiningConcept ? <Loader2 size={14} className="spin" /> : <Sparkles size={14} />}
              Refine
            </button>
          </div>
        </div>
      )}

      {/* ── Director's Notes — editable suggestions ───────────────── */}
      {concept && (
        <div className="grok-card">
          <div className="grok-card-header">
            <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Video size={16} />
              Director's Notes
            </div>
          </div>

          {/* Motion */}
          <div className="form-group">
            <label className="grok-section-label">🎬 Motion</label>
            <textarea
              value={motionHint}
              onChange={(e) => setMotionHint(e.target.value)}
              rows={2}
              placeholder="How subjects and scene elements move..."
              className="i2t-director-textarea"
            />
          </div>

          {/* Audio / Ambient */}
          <div className="form-group">
            <label className="grok-section-label">🔊 Audio & Ambient</label>
            <textarea
              value={directorAudio}
              onChange={(e) => setDirectorAudio(e.target.value)}
              rows={2}
              placeholder="Environmental sounds, music mood, ambience..."
              className="i2t-director-textarea"
            />
          </div>

          {/* Dialogue per subject */}
          <div className="form-group">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <label className="grok-section-label">💬 Dialogue</label>
              <button
                className="icon-btn"
                onClick={() => setDialogueLines(prev => [...prev, { subject: '', line: '' }])}
                title="Add dialogue line"
                style={{
                  display: 'flex', alignItems: 'center', gap: '3px',
                  padding: '3px 8px', fontSize: '0.7rem',
                  color: 'var(--accent-color, #8b5cf6)', background: 'transparent',
                  border: '1px solid var(--accent-color, #8b5cf6)', borderRadius: '5px',
                  cursor: 'pointer',
                }}
              >
                <Plus size={11} /> Add
              </button>
            </div>
            {dialogueLines.length === 0 && (
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted, #666)', margin: '4px 0', fontStyle: 'italic' }}>
                No dialogue — add lines for characters to speak in the video.
              </p>
            )}
            {dialogueLines.map((dl, i) => (
              <div key={i} className="i2t-dialogue-row">
                <input
                  type="text"
                  value={dl.subject}
                  onChange={(e) => {
                    const updated = [...dialogueLines]
                    updated[i] = { ...updated[i], subject: e.target.value }
                    setDialogueLines(updated)
                  }}
                  placeholder="Who"
                  className="i2t-dialogue-who"
                />
                <input
                  type="text"
                  value={dl.line}
                  onChange={(e) => {
                    const updated = [...dialogueLines]
                    updated[i] = { ...updated[i], line: e.target.value }
                    setDialogueLines(updated)
                  }}
                  placeholder="What they say..."
                  className="i2t-dialogue-line"
                />
                <button
                  className="icon-btn"
                  onClick={() => setDialogueLines(prev => prev.filter((_, j) => j !== i))}
                  title="Remove"
                  style={{
                    width: '24px', height: '24px', padding: '4px',
                    color: 'var(--text-muted, #666)', background: 'transparent',
                    border: 'none', cursor: 'pointer', flexShrink: 0,
                  }}
                >
                  <X size={13} />
                </button>
              </div>
            ))}
          </div>

          {/* Camera Direction — LLM-driven, editable text */}
          <div className="form-group">
            <label className="grok-section-label">🎥 Camera Direction</label>
            <textarea
              value={cameraDirection}
              onChange={(e) => setCameraDirection(e.target.value)}
              rows={2}
              placeholder="Camera movement, shot type, composition... e.g. 'slow dolly-in from wide to medium close-up on subject'"
              className="i2t-director-textarea"
            />
          </div>

          {/* ── Notes Refinement Input ────────────────────────────── */}
          <div className="i2t-refine-row">
            <input
              type="text"
              value={notesRefineText}
              onChange={(e) => setNotesRefineText(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !refiningNotes) handleRefine('notes') }}
              placeholder="Refine notes… e.g. 'more action' or 'add suspenseful music'"
              className="i2t-refine-input"
              disabled={refiningNotes}
            />
            <button
              className="i2t-refine-btn"
              onClick={() => handleRefine('notes')}
              disabled={refiningNotes || !notesRefineText.trim()}
              title="Refine director's notes with AI"
            >
              {refiningNotes ? <Loader2 size={14} className="spin" /> : <Sparkles size={14} />}
              Refine
            </button>
          </div>
        </div>
      )}

      {/* Settings Card */}
      <div className="grok-card">
        <div className="grok-card-header">
          <div className="grok-card-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Wand2 size={16} />
            Settings
          </div>
        </div>

        {/* Vision Model */}
        <div className="form-group">
          <label className="grok-section-label">Vision Model</label>
          <select className="form-select" value={model} onChange={(e) => setModel(e.target.value)}>
            {MODELS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label} - {m.description}
              </option>
            ))}
          </select>
        </div>

        {/* Prompt Generator — dropdown */}
        <div className="form-group">
          <label className="grok-section-label">Prompt Generator</label>
          <select
            className="form-select"
            value={isPromptMode(mode) ? mode : ''}
            onChange={(e) => setMode(e.target.value || 'detailed')}
          >
            <option value="">None (caption only)</option>
            {CAPTION_MODES.filter(m => m.group === 'prompt').map((m) => (
              <option key={m.id} value={m.id}>{m.label} — {m.description}</option>
            ))}
          </select>
        </div>

        {/* Detail Level */}
        <div className="form-group">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <label style={{ margin: 0 }}>Detail Level</label>
            <span style={{
              fontSize: '0.75rem',
              padding: '2px 8px',
              borderRadius: '4px',
              background: detailLevel >= 4 ? 'rgba(139, 92, 246, 0.15)' : 'rgba(255,255,255,0.06)',
              color: detailLevel >= 4 ? '#a78bfa' : 'var(--text-muted, #888)',
            }}>
              {['', 'Brief', 'Concise', 'Default', 'Thorough', 'Exhaustive'][detailLevel]}
            </span>
          </div>
          <input
            type="range"
            min={1}
            max={5}
            step={1}
            value={detailLevel}
            onChange={(e) => setDetailLevel(Number(e.target.value))}
            style={{ width: '100%', marginTop: '4px' }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-muted, #666)', marginTop: '2px' }}>
            <span>Brief</span>
            <span>Exhaustive</span>
          </div>
        </div>

        {/* Camera Motion — prompt modes only, when no concept analysis done */}
        {isPromptMode(mode) && !concept && (
          <div className="form-group">
            <CameraMotionSelector
              value={cameraMotion}
              onChange={setCameraMotion}
            />
            <p style={{ margin: '6px 0 0', fontSize: '11px', color: 'var(--text-muted, #666)' }}>
              Prepended to prompt — ready for T2V / I2V.
            </p>
          </div>
        )}

        {/* Include Motion — prompt modes only, when no concept analysis done */}
        {isPromptMode(mode) && !concept && (
          <div className="form-group" style={{ marginTop: '4px' }}>
            <label className="i2t-checkbox">
              <input
                type="checkbox"
                checked={includeMotion}
                onChange={(e) => setIncludeMotion(e.target.checked)}
              />
              <span className="i2t-checkbox-label">🎬 Include Motion in Prompt</span>
              <span className="i2t-checkbox-desc">Subject animation, movement, dynamics</span>
            </label>
            {includeMotion && (
              <input
                type="text"
                value={motionHint}
                onChange={(e) => setMotionHint(e.target.value)}
                placeholder="e.g., walking towards camera, hair blowing in wind..."
                style={{
                  marginTop: '6px',
                  width: '100%',
                  background: 'var(--bg-input, #1a1a1a)',
                  border: '1px solid var(--border-color, #444)',
                  borderRadius: '6px',
                  padding: '6px 10px',
                  fontSize: '0.8rem',
                  color: 'var(--text-primary, #eee)',
                  outline: 'none',
                }}
              />
            )}
          </div>
        )}

        {/* NSFW Intensity — NSFW mode only */}
        {mode === 'prompt_nsfw' && (
          <div className="form-group">
            <div className="nsfw-intensity-section">
              <div className="nsfw-intensity-header">
                <label>Intensity</label>
                <span className="nsfw-intensity-badge">
                  {NSFW_LEVELS.find(l => l.value === nsfwIntensity)?.emoji}{' '}
                  {NSFW_LEVELS.find(l => l.value === nsfwIntensity)?.label}
                </span>
              </div>
              <input
                type="range"
                min={1}
                max={5}
                step={1}
                value={nsfwIntensity}
                onChange={(e) => setNsfwIntensity(Number(e.target.value))}
                className="nsfw-slider"
              />
              <div className="nsfw-intensity-labels">
                {NSFW_LEVELS.map((level) => (
                  <span
                    key={level.value}
                    className={`nsfw-label ${nsfwIntensity === level.value ? 'active' : ''}`}
                    onClick={() => setNsfwIntensity(level.value)}
                  >
                    {level.emoji}
                  </span>
                ))}
              </div>
              <p className="nsfw-intensity-desc">
                {NSFW_LEVELS.find(l => l.value === nsfwIntensity)?.description}
              </p>
            </div>
          </div>
        )}
      </div>

      <button
        className="primary-btn"
        onClick={handleCaption}
        disabled={!file || loading || restoringFile}
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
      >
        {restoringFile ? (
          <>
            <Loader2 size={18} className="spin" />
            Restoring image...
          </>
        ) : loading ? (
          <>
            <Loader2 size={18} className="spin" />
            {isPromptMode(mode) ? 'Generating prompt...' : 'Generating caption...'}
          </>
        ) : (
          <>
            <Wand2 size={18} />
            {isPromptMode(mode) ? 'Generate Prompt' : 'Generate Caption'}
          </>
        )}
      </button>

      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}

      {caption && (
        <div className="tool-section result-section">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <h3>{isPromptMode(mode) ? '🎯 Generated Prompt' : 'Generated Caption'}</h3>
            <button
              className="icon-btn"
              style={{
                width: '28px', height: '28px', padding: '6px',
                background: showRefineInput ? 'var(--accent-color, #8b5cf6)' : undefined,
                color: showRefineInput ? 'white' : undefined,
                borderRadius: '6px',
                border: '1px solid var(--border-color, #444)',
                cursor: 'pointer',
              }}
              onClick={() => setShowRefineInput(!showRefineInput)}
              disabled={!caption.trim()}
              title="Refine all outputs with AI"
            >
              <Pencil size={14} />
            </button>
          </div>

          {/* Refine instruction input — refines all outputs at once */}
          {showRefineInput && (
            <div style={{
              marginTop: '8px',
              padding: '8px 12px',
              backgroundColor: 'rgba(139, 92, 246, 0.08)',
              border: '1px solid rgba(139, 92, 246, 0.25)',
              borderRadius: '8px',
              display: 'flex',
              gap: '8px',
              alignItems: 'center',
            }}>
              <Pencil size={14} style={{ color: '#a78bfa', flexShrink: 0 }} />
              <input
                type="text"
                value={refineInstruction}
                onChange={(e) => setRefineInstruction(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter' && refineInstruction.trim()) handleRefineCaption() }}
                placeholder="How to improve? (e.g., more cinematic, add rain, darker mood...)"
                style={{
                  flex: 1,
                  background: 'var(--bg-input, #1a1a1a)',
                  border: '1px solid var(--border-color, #444)',
                  borderRadius: '6px',
                  padding: '6px 10px',
                  fontSize: '0.8rem',
                  color: 'var(--text-primary, #eee)',
                  outline: 'none',
                }}
              />
              <button
                className="icon-btn"
                style={{
                  height: '28px',
                  padding: '4px 12px',
                  fontSize: '0.75rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  background: isRefining ? 'var(--bg-input)' : 'linear-gradient(135deg, #8b5cf6, #6d28d9)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  whiteSpace: 'nowrap',
                  cursor: 'pointer',
                }}
                onClick={handleRefineCaption}
                disabled={isRefining || !refineInstruction.trim()}
                title="Refine all outputs with AI suggestion"
              >
                {isRefining ? <Loader2 size={12} className="spin" /> : <Pencil size={12} />}
                <span>{isRefining ? 'Refining...' : 'Refine'}</span>
              </button>
            </div>
          )}

          {/* ✨ Positive Prompt — always shown */}
          <div className="caption-result" style={{ marginTop: '8px', position: 'relative' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <label className="i2t-output-label">✨ Positive Prompt</label>
              <button
                className="icon-btn"
                style={{ width: '24px', height: '24px', padding: '4px', border: 'none', background: 'transparent', cursor: 'pointer', color: 'var(--text-muted, #888)', borderRadius: '4px' }}
                onClick={handleCopyPositive}
                title="Copy positive prompt"
              >
                <Copy size={13} />
              </button>
            </div>
            <textarea
              value={caption}
              onChange={(e) => setCaption(e.target.value)}
              rows={isPromptMode(mode) ? 5 : 4}
            />
          </div>

          {/* 🚫 Negative Prompt — always shown in prompt mode */}
          {isPromptMode(mode) && (
            <div className="caption-result" style={{ marginTop: '10px', position: 'relative' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <label className="i2t-output-label">🚫 Negative Prompt</label>
                <button
                  className="icon-btn"
                  style={{ width: '24px', height: '24px', padding: '4px', border: 'none', background: 'transparent', cursor: 'pointer', color: 'var(--text-muted, #888)', borderRadius: '4px' }}
                  onClick={handleCopyNegative}
                  title="Copy negative prompt"
                >
                  <Copy size={13} />
                </button>
              </div>
              <textarea
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
                rows={3}
                placeholder="Negative prompt will appear here..."
              />
            </div>
          )}

          {/* 🔊 Audio Prompt — shown when audio context was used */}
          {audioPrompt && (
            <div className="caption-result" style={{ marginTop: '10px', position: 'relative' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <label className="i2t-output-label">🔊 Audio Prompt</label>
                <button
                  className="icon-btn"
                  style={{ width: '24px', height: '24px', padding: '4px', border: 'none', background: 'transparent', cursor: 'pointer', color: 'var(--text-muted, #888)', borderRadius: '4px' }}
                  onClick={() => navigator.clipboard.writeText(audioPrompt)}
                  title="Copy audio prompt"
                >
                  <Copy size={13} />
                </button>
              </div>
              <textarea
                value={audioPrompt}
                onChange={(e) => setAudioPrompt(e.target.value)}
                rows={3}
                placeholder="Audio description for LTX video with audio..."
              />
            </div>
          )}

          {/* Send to Tool */}
          {onSendToTool && (
            <div style={{ marginTop: '12px', position: 'relative' }}>
              <button
                className={isPromptMode(mode) ? 'btn-primary btn-large btn-glow' : 'btn-primary btn-large'}
                onClick={() => setShowSendMenu(!showSendMenu)}
                style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}
              >
                <Send size={16} />
                Use in Tool
                <ChevronDown size={14} style={{ marginLeft: '4px', transform: showSendMenu ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
              </button>
              {showSendMenu && (
                <div className="i2t-send-menu">
                  {SEND_TO_TOOLS.map((tool) => (
                    <button
                      key={tool.id}
                      className="i2t-send-menu-item"
                      onClick={() => handleSendToTool(tool.id)}
                    >
                      {tool.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

        </div>
      )}

      <style>{`
        .i2t-send-menu {
          position: absolute;
          bottom: calc(100% + 6px);
          left: 0;
          right: 0;
          background: var(--bg-card, #1e1e1e);
          border: 1px solid var(--border-color, #444);
          border-radius: 10px;
          overflow: hidden;
          z-index: 50;
          box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        .i2t-send-menu-item {
          display: block;
          width: 100%;
          padding: 10px 14px;
          background: none;
          border: none;
          color: var(--text-primary, #eee);
          font-size: 0.85rem;
          text-align: left;
          cursor: pointer;
          transition: background 0.15s;
        }
        .i2t-send-menu-item:hover {
          background: var(--bg-hover, rgba(139, 92, 246, 0.15));
        }
        .i2t-send-menu-item + .i2t-send-menu-item {
          border-top: 1px solid var(--border-color, #333);
        }
        .upload-dropzone {
          border: 2px dashed var(--border-color, #444);
          border-radius: 12px;
          padding: 40px;
          text-align: center;
          cursor: pointer;
          transition: all 0.2s;
          min-height: 200px;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .upload-dropzone:hover {
          border-color: var(--accent-color, #7c3aed);
          background: rgba(124, 58, 237, 0.05);
        }
        .upload-dropzone.has-preview {
          padding: 8px;
        }
        .upload-preview {
          max-width: 100%;
          max-height: 300px;
          border-radius: 8px;
          object-fit: contain;
        }
        .upload-placeholder {
          color: var(--text-muted, #888);
        }
        .upload-placeholder p {
          margin-top: 12px;
        }
        .button-group {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
        }
        .btn-option {
          padding: 8px 16px;
          border: 1px solid var(--border-color, #444);
          border-radius: 8px;
          background: transparent;
          color: var(--text-color, #fff);
          cursor: pointer;
          transition: all 0.2s;
        }
        .btn-option:hover {
          border-color: var(--accent-color, #7c3aed);
        }
        .btn-option.active {
          background: var(--accent-color, #7c3aed);
          border-color: var(--accent-color, #7c3aed);
        }
        .caption-result textarea {
          width: 100%;
          padding: 12px;
          border-radius: 8px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          font-family: inherit;
          resize: vertical;
        }
        .caption-actions {
          display: flex;
          gap: 8px;
          margin-top: 12px;
        }
        .error-message {
          padding: 12px;
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 8px;
          color: #ef4444;
          margin-top: 12px;
        }
        .btn-option--nsfw.active {
          background: #dc2626;
          border-color: #dc2626;
        }
        .nsfw-intensity-section {
          margin-top: 12px;
          padding: 12px 16px;
          background: rgba(220, 38, 38, 0.06);
          border: 1px solid rgba(220, 38, 38, 0.2);
          border-radius: 10px;
        }
        .nsfw-intensity-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 8px;
        }
        .nsfw-intensity-header label {
          font-size: 0.85rem;
          font-weight: 600;
          color: var(--text-primary, #eee);
        }
        .nsfw-intensity-badge {
          font-size: 0.8rem;
          padding: 2px 10px;
          border-radius: 12px;
          background: rgba(220, 38, 38, 0.15);
          color: #f87171;
          font-weight: 600;
        }
        .nsfw-slider {
          width: 100%;
          height: 6px;
          -webkit-appearance: none;
          appearance: none;
          background: linear-gradient(90deg, #f97316 0%, #dc2626 50%, #7f1d1d 100%);
          border-radius: 3px;
          outline: none;
          cursor: pointer;
        }
        .nsfw-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: #ef4444;
          border: 2px solid #fff;
          box-shadow: 0 0 8px rgba(239, 68, 68, 0.5);
          cursor: pointer;
        }
        .nsfw-slider::-moz-range-thumb {
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: #ef4444;
          border: 2px solid #fff;
          box-shadow: 0 0 8px rgba(239, 68, 68, 0.5);
          cursor: pointer;
        }
        .nsfw-intensity-labels {
          display: flex;
          justify-content: space-between;
          margin-top: 6px;
          padding: 0 2px;
        }
        .nsfw-label {
          cursor: pointer;
          font-size: 1rem;
          opacity: 0.4;
          transition: all 0.2s;
          user-select: none;
        }
        .nsfw-label:hover {
          opacity: 0.8;
          transform: scale(1.2);
        }
        .nsfw-label.active {
          opacity: 1;
          transform: scale(1.3);
        }
        .nsfw-intensity-desc {
          margin-top: 6px;
          font-size: 0.78rem;
          color: var(--text-muted, #888);
          text-align: center;
          font-style: italic;
        }
        .btn-glow {
          box-shadow: 0 0 12px rgba(124, 58, 237, 0.4);
        }
        .prompt-hint {
          margin-top: 8px;
          font-size: 0.85em;
          color: var(--text-muted, #888);
        }
        .i2t-checkbox {
          display: flex;
          align-items: center;
          gap: 8px;
          cursor: pointer;
          padding: 6px 8px;
          border-radius: 6px;
          transition: background 0.15s;
        }
        .i2t-checkbox:hover {
          background: rgba(255,255,255,0.04);
        }
        .i2t-checkbox input[type="checkbox"] {
          width: 16px;
          height: 16px;
          accent-color: var(--accent-color, #7c3aed);
          cursor: pointer;
          flex-shrink: 0;
        }
        .i2t-checkbox-label {
          font-size: 0.82rem;
          font-weight: 600;
          color: var(--text-primary, #eee);
          white-space: nowrap;
        }
        .i2t-checkbox-desc {
          font-size: 0.72rem;
          color: var(--text-muted, #888);
          margin-left: auto;
        }
        .i2t-output-label {
          display: block;
          font-size: 0.78rem;
          font-weight: 600;
          color: var(--text-muted, #aaa);
          margin-bottom: 4px;
        }
        /* ── Concept Card styles ── */
        .i2t-concept-field {
          padding: 8px 0;
          border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        .i2t-concept-field:last-child {
          border-bottom: none;
        }
        .i2t-concept-label {
          display: block;
          font-size: 0.75rem;
          font-weight: 700;
          color: var(--accent-color, #8b5cf6);
          margin-bottom: 4px;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        .i2t-concept-text {
          margin: 0;
          font-size: 0.85rem;
          color: var(--text-primary, #ddd);
          line-height: 1.4;
        }
        .i2t-concept-subject {
          padding: 6px 10px;
          margin: 4px 0;
          background: rgba(255,255,255,0.03);
          border-radius: 6px;
          border-left: 2px solid var(--accent-color, #8b5cf6);
        }
        /* ── Director's Notes styles ── */
        .i2t-director-textarea {
          width: 100%;
          padding: 8px 10px;
          border-radius: 6px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-input, #1a1a1a);
          color: var(--text-primary, #eee);
          font-family: inherit;
          font-size: 0.82rem;
          resize: vertical;
          outline: none;
          transition: border-color 0.2s;
        }
        .i2t-director-textarea:focus {
          border-color: var(--accent-color, #7c3aed);
        }
        .i2t-dialogue-row {
          display: flex;
          gap: 6px;
          align-items: center;
          margin-top: 6px;
        }
        .i2t-dialogue-who {
          width: 100px;
          flex-shrink: 0;
          padding: 6px 8px;
          border-radius: 6px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-input, #1a1a1a);
          color: var(--accent-color, #a78bfa);
          font-size: 0.8rem;
          font-weight: 600;
          outline: none;
        }
        .i2t-dialogue-line {
          flex: 1;
          padding: 6px 8px;
          border-radius: 6px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-input, #1a1a1a);
          color: var(--text-primary, #eee);
          font-size: 0.8rem;
          outline: none;
        }
        .i2t-dialogue-who:focus, .i2t-dialogue-line:focus {
          border-color: var(--accent-color, #7c3aed);
        }
        /* ── Refinement input row ── */
        .i2t-refine-row {
          display: flex;
          gap: 6px;
          align-items: center;
          margin-top: 10px;
          padding-top: 10px;
          border-top: 1px solid rgba(255,255,255,0.06);
        }
        .i2t-refine-input {
          flex: 1;
          padding: 7px 10px;
          border-radius: 6px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-input, #1a1a1a);
          color: var(--text-primary, #eee);
          font-size: 0.8rem;
          outline: none;
          transition: border-color 0.2s;
        }
        .i2t-refine-input:focus {
          border-color: var(--accent-color, #7c3aed);
        }
        .i2t-refine-input::placeholder {
          color: var(--text-muted, #666);
          font-style: italic;
        }
        .i2t-refine-btn {
          display: flex;
          align-items: center;
          gap: 4px;
          padding: 7px 12px;
          border-radius: 6px;
          border: 1px solid var(--accent-color, #8b5cf6);
          background: rgba(139, 92, 246, 0.1);
          color: var(--accent-color, #a78bfa);
          font-size: 0.78rem;
          font-weight: 600;
          cursor: pointer;
          white-space: nowrap;
          transition: all 0.15s;
        }
        .i2t-refine-btn:hover:not(:disabled) {
          background: rgba(139, 92, 246, 0.2);
        }
        .i2t-refine-btn:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }
        .spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}
