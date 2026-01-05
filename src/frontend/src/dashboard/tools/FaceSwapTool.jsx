import React, { useState, useCallback, useRef } from 'react'
import { User, Upload, Loader2, Download, AlertCircle, ChevronDown, Smile, RefreshCw } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { postForm, getJson } from '../../api'

const FACE_MODELS = [
  { id: 'inswapper', label: 'InSwapper (Best Quality)', description: 'High quality, slower' },
  { id: 'simswap', label: 'SimSwap (Fast)', description: 'Faster, good quality' },
]

const ENHANCE_OPTIONS = [
  { id: 'none', label: 'None' },
  { id: 'gfpgan', label: 'GFPGAN (Faces)' },
  { id: 'codeformer', label: 'CodeFormer (Natural)' },
  { id: 'both', label: 'Both (Best)' },
]

export default function FaceSwapTool({ onJobSubmitted }) {
  const [targetFile, setTargetFile] = useState(null)
  const [targetPreview, setTargetPreview] = useState(null)
  const [sourceFile, setSourceFile] = useState(null)
  const [sourcePreview, setSourcePreview] = useState(null)
  const [model, setModel] = useState(FACE_MODELS[0])
  const [enhance, setEnhance] = useState('gfpgan')
  const [strength, setStrength] = useState(1.0)
  const [blendAmount, setBlendAmount] = useState(0.8)
  const [faceIndex, setFaceIndex] = useState(0)
  const [swapAllFaces, setSwapAllFaces] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [detectedFaces, setDetectedFaces] = useState(null)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [lastQueued, setLastQueued] = useState(null)

  const targetInputRef = useRef(null)
  const sourceInputRef = useRef(null)

  const handleTargetDrop = useCallback((e) => {
    e.preventDefault()
    const dropped = e.dataTransfer?.files?.[0] || e.target?.files?.[0]
    if (dropped && (dropped.type.startsWith('image/') || dropped.type.startsWith('video/'))) {
      setTargetFile(dropped)
      setResult(null)
      setError(null)
      setDetectedFaces(null)
      setLastQueued(null)
      const url = URL.createObjectURL(dropped)
      setTargetPreview(url)
    }
  }, [])

  const handleSourceDrop = useCallback((e) => {
    e.preventDefault()
    const dropped = e.dataTransfer?.files?.[0] || e.target?.files?.[0]
    if (dropped && dropped.type.startsWith('image/')) {
      setSourceFile(dropped)
      setResult(null)
      setError(null)
      setLastQueued(null)
      const url = URL.createObjectURL(dropped)
      setSourcePreview(url)
    }
  }, [])

  const handleDragOver = (e) => e.preventDefault()

  const detectFaces = async () => {
    if (!targetFile) return

    setIsLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('image', targetFile)

      const res = await postForm(`${BACKEND_BASE}/detect-faces`, formData)

      if (res.ok && res.data?.faces) {
        setDetectedFaces(res.data.faces)
        if (DEBUG) console.log('👤 Detected faces:', res.data.faces.length)
      } else {
        throw new Error(res.data?.detail || 'Face detection failed')
      }
    } catch (err) {
      console.error('❌ Face detection error:', err)
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleGenerate = async () => {
    if (!targetFile || !sourceFile) {
      setError('Please upload both target and source face images')
      return
    }

    setIsLoading(true)
    setError(null)
    setResult(null)
    setLastQueued(null)

    try {
      const formData = new FormData()
      formData.append('target', targetFile)
      formData.append('source', sourceFile)
      formData.append('model', model.id)
      formData.append('enhance', enhance)
      formData.append('strength', strength)
      formData.append('blend', blendAmount)
      formData.append('face_index', swapAllFaces ? -1 : faceIndex)

      if (DEBUG) console.log('👤 FaceSwap request:', {
        model: model.id, enhance, strength, faceIndex
      })

      const res = await postForm(`${BACKEND_BASE}/face-swap`, formData)

      if (!res.ok) {
        throw new Error(res.data?.detail || 'Face swap request failed')
      }

      if (res.data?.prompt_id) {
        // Show queued confirmation
        setLastQueued({
          promptId: res.data.prompt_id,
          model: model.label
        })

        // Notify queue indicator
        if (onJobSubmitted) onJobSubmitted({ prompt_id: res.data.prompt_id })

        if (DEBUG) console.debug('📋 FaceSwap queued:', res.data.prompt_id)

        // Don't wait for completion - job will appear in queue/history when done
      } else if (res.data?.url) {
        setResult({ url: res.data.url })
      }

    } catch (err) {
      console.error('❌ FaceSwap error:', err)
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleDownload = () => {
    if (!result?.url) return
    const ext = targetFile?.type.startsWith('video/') ? 'mp4' : 'png'
    const a = document.createElement('a')
    a.href = result.url
    a.download = `face_swap_${Date.now()}.${ext}`
    a.click()
  }

  const swapInputs = () => {
    const tempFile = targetFile
    const tempPreview = targetPreview
    setTargetFile(sourceFile)
    setTargetPreview(sourcePreview)
    setSourceFile(tempFile)
    setSourcePreview(tempPreview)
    setResult(null)
    setDetectedFaces(null)
    setLastQueued(null)
  }

  return (
    <div className="space-y-4">
      {/* File Upload Section */}
      <div className="grid grid-cols-2 gap-4">
        {/* Target Image/Video */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Target (face to replace)
          </label>
          <div
            onClick={() => targetInputRef.current?.click()}
            onDrop={handleTargetDrop}
            onDragOver={handleDragOver}
            className="border-2 border-dashed border-gray-600 rounded-lg p-4 text-center cursor-pointer hover:border-purple-500 transition-colors aspect-square flex items-center justify-center"
          >
            <input
              ref={targetInputRef}
              type="file"
              accept="image/*,video/*"
              onChange={handleTargetDrop}
              className="hidden"
            />
            {targetPreview ? (
              <div className="relative w-full h-full">
                {targetFile?.type.startsWith('video/') ? (
                  <video src={targetPreview} className="w-full h-full object-cover rounded" muted />
                ) : (
                  <img src={targetPreview} alt="Target" className="w-full h-full object-cover rounded" />
                )}
                {detectedFaces && (
                  <div className="absolute bottom-1 right-1 bg-black/70 px-2 py-1 rounded text-xs">
                    {detectedFaces.length} face{detectedFaces.length !== 1 ? 's' : ''} detected
                  </div>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center gap-2 text-gray-400">
                <Upload className="w-6 h-6" />
                <span className="text-xs">Target image/video</span>
              </div>
            )}
          </div>
        </div>

        {/* Source Face */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Source (face to use)
          </label>
          <div
            onClick={() => sourceInputRef.current?.click()}
            onDrop={handleSourceDrop}
            onDragOver={handleDragOver}
            className="border-2 border-dashed border-gray-600 rounded-lg p-4 text-center cursor-pointer hover:border-blue-500 transition-colors aspect-square flex items-center justify-center"
          >
            <input
              ref={sourceInputRef}
              type="file"
              accept="image/*"
              onChange={handleSourceDrop}
              className="hidden"
            />
            {sourcePreview ? (
              <img src={sourcePreview} alt="Source" className="w-full h-full object-cover rounded" />
            ) : (
              <div className="flex flex-col items-center gap-2 text-gray-400">
                <Smile className="w-6 h-6" />
                <span className="text-xs">Source face</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Swap Button */}
      {(targetFile || sourceFile) && (
        <button
          onClick={swapInputs}
          className="w-full py-2 bg-gray-700 hover:bg-gray-600 rounded-lg flex items-center justify-center gap-2 text-sm"
        >
          <RefreshCw className="w-4 h-4" />
          Swap Target ↔ Source
        </button>
      )}

      {/* Detect Faces Button */}
      {targetFile && !targetFile.type.startsWith('video/') && (
        <button
          onClick={detectFaces}
          disabled={isLoading}
          className="w-full py-2 bg-gray-700 hover:bg-gray-600 rounded-lg flex items-center justify-center gap-2 text-sm"
        >
          <User className="w-4 h-4" />
          Detect Faces
        </button>
      )}

      {/* Face Selection (if multiple faces detected) */}
      {detectedFaces && detectedFaces.length > 1 && (
        <div className="bg-gray-800 rounded-lg p-3 space-y-2">
          <label className="block text-sm font-medium text-gray-300">
            Select Face to Replace
          </label>
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={swapAllFaces}
                onChange={(e) => setSwapAllFaces(e.target.checked)}
                className="rounded bg-gray-700 border-gray-600"
              />
              <span className="text-sm text-gray-300">Swap all faces</span>
            </label>
          </div>
          {!swapAllFaces && (
            <div className="flex gap-2 flex-wrap">
              {detectedFaces.map((face, idx) => (
                <button
                  key={idx}
                  onClick={() => setFaceIndex(idx)}
                  className={`px-3 py-1 text-sm rounded ${
                    faceIndex === idx
                      ? 'bg-purple-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  Face {idx + 1}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Model Selection */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Model
        </label>
        <div className="space-y-2">
          {FACE_MODELS.map(m => (
            <button
              key={m.id}
              onClick={() => setModel(m)}
              className={`w-full px-3 py-2 text-left rounded transition-colors ${
                model.id === m.id
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              <div className="font-medium text-sm">{m.label}</div>
              <div className="text-xs opacity-70">{m.description}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Enhancement */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Face Enhancement
        </label>
        <div className="grid grid-cols-2 gap-2">
          {ENHANCE_OPTIONS.map(e => (
            <button
              key={e.id}
              onClick={() => setEnhance(e.id)}
              className={`px-3 py-2 text-sm rounded transition-colors ${
                enhance === e.id
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {e.label}
            </button>
          ))}
        </div>
      </div>

      {/* Advanced Settings */}
      <div className="border border-gray-700 rounded-lg overflow-hidden">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full px-4 py-2 bg-gray-800 flex items-center justify-between text-gray-300 hover:bg-gray-750"
        >
          <span className="text-sm font-medium">Advanced Settings</span>
          <ChevronDown className={`w-4 h-4 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
        </button>

        {showAdvanced && (
          <div className="p-4 space-y-4 bg-gray-850">
            {/* Strength */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Swap Strength: {strength.toFixed(2)}
              </label>
              <input
                type="range"
                min={0.1}
                max={1}
                step={0.05}
                value={strength}
                onChange={(e) => setStrength(Number(e.target.value))}
                className="w-full accent-purple-500"
              />
              <span className="text-xs text-gray-500">Lower = more original features preserved</span>
            </div>

            {/* Blend */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Edge Blend: {blendAmount.toFixed(2)}
              </label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={blendAmount}
                onChange={(e) => setBlendAmount(Number(e.target.value))}
                className="w-full accent-purple-500"
              />
              <span className="text-xs text-gray-500">Blend face edges with background</span>
            </div>
          </div>
        )}
      </div>

      {/* Warning */}
      <div className="flex items-start gap-2 p-3 bg-yellow-900/30 border border-yellow-700/50 rounded-lg">
        <AlertCircle className="w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5" />
        <div className="text-sm text-yellow-200">
          <strong>Ethical Use:</strong> Only use face swap with consent of all parties involved.
          Creating non-consensual deepfakes is illegal in many jurisdictions.
        </div>
      </div>

      {/* Generate Button */}
      <button
        onClick={handleGenerate}
        disabled={isLoading || !targetFile || !sourceFile}
        className="w-full py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg font-semibold flex items-center justify-center gap-2 transition-colors"
      >
        {isLoading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Swapping... {progress > 0 && `${Math.round(progress)}%`}
          </>
        ) : (
          <>
            <User className="w-5 h-5" />
            Swap Face
          </>
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
            {targetFile?.type.startsWith('video/') ? (
              <video src={result.url} controls className="w-full" />
            ) : (
              <img src={result.url} alt="Result" className="w-full" />
            )}
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

      {/* Info */}
      <div className="text-xs text-gray-500 space-y-1">
        <p>👤 <strong>Face Swap</strong> replaces faces in images or videos using AI.</p>
        <p>📸 For best results, use clear frontal face photos with good lighting.</p>
        <p>🎬 Video processing may take longer depending on length and resolution.</p>
      </div>
    </div>
  )
}
