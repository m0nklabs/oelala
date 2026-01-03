import React, { useState, useRef } from 'react'
import BACKEND_BASE from '../config'
import { postForm } from '../api'
import { Upload, Play, Download, Loader, FileText, Image, Sliders } from 'lucide-react'
import PresetSelector from './PresetSelector'
import './VideoGenerator.css'
import './PresetSelector.css'

function VideoGenerator() {
  const [activeTab, setActiveTab] = useState('image') // 'image' or 'text'
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [prompt, setPrompt] = useState('')
  const [numFrames, setNumFrames] = useState(16)
  const [modelType, setModelType] = useState('light') // 'light', 'svd', 'wan2.2'
  const [isGenerating, setIsGenerating] = useState(false)
  const [generatedVideo, setGeneratedVideo] = useState(null)
  const [error, setError] = useState('')
  const [usePoseGuidance, setUsePoseGuidance] = useState(false)
  const [usePresets, setUsePresets] = useState(true) // New: toggle preset mode
  const [selectedPreset, setSelectedPreset] = useState(null)
  const [presetParameters, setPresetParameters] = useState({})
  const [extendMode, setExtendMode] = useState(false) // Sequential clip extension
  const [clipCount, setClipCount] = useState(1) // Number of sequential clips (1-5)
  const fileInputRef = useRef(null)

  const handleFileSelect = (event) => {
    const file = event.target.files[0]
    if (file) {
      setSelectedFile(file)
      setError('')

      // Create preview
      const reader = new FileReader()
      reader.onload = (e) => setPreview(e.target.result)
      reader.readAsDataURL(file)
    }
  }

  const handlePresetChange = (preset) => {
    setSelectedPreset(preset)
    // Update prompt from preset if available
    if (preset?.parameters?.prompt?.default) {
      setPrompt(preset.parameters.prompt.default)
    }
    // Update frames from preset
    if (preset?.parameters?.num_frames?.default) {
      setNumFrames(preset.parameters.num_frames.default)
    }
  }

  const handleParametersChange = (params) => {
    setPresetParameters(params)
    // Sync prompt if changed via preset
    if (params.prompt !== undefined) {
      setPrompt(params.prompt)
    }
    if (params.num_frames !== undefined) {
      setNumFrames(params.num_frames)
    }
  }

  const handleGenerate = async () => {
    if (activeTab === 'image' && !selectedFile) {
      setError('Please select an image first')
      return
    }

    if (activeTab === 'text' && !prompt.trim()) {
      setError('Please enter a text prompt')
      return
    }

    setIsGenerating(true)
    setError('')
    setGeneratedVideo(null)

    const formData = new FormData()
    let endpoint

    if (activeTab === 'image') {
      formData.append('file', selectedFile)
      formData.append('num_frames', numFrames.toString())
      formData.append('model_type', modelType)

      // Add prompt only for regular generation
      if (!usePoseGuidance) {
        formData.append('prompt', prompt)
      }

      // If using presets, add all preset parameters
      if (usePresets && selectedPreset) {
        formData.append('preset_id', selectedPreset.id)
        formData.append('preset_parameters', JSON.stringify(presetParameters))
      }

      // Add extend mode parameters for sequential generation
      if (extendMode && clipCount > 1) {
        formData.append('extend_mode', 'true')
        formData.append('clip_count', clipCount.toString())
      }

      // Select endpoint based on model type
      if (usePoseGuidance) {
        endpoint = `${BACKEND_BASE}/generate-pose`
      } else if (modelType === 'wan2.2-q6') {
        endpoint = `${BACKEND_BASE}/generate-wan22-q6-comfyui`
      } else if (modelType === 'wan2.2-enhanced') {
        endpoint = `${BACKEND_BASE}/generate-wan22-enhanced-comfyui`
      } else if (modelType === 'wan2.2') {
        endpoint = `${BACKEND_BASE}/generate-wan22-comfyui`
      } else {
        endpoint = `${BACKEND_BASE}/generate`
      }
    } else {
      // Text-to-video
      formData.append('prompt', prompt)
      formData.append('num_frames', numFrames.toString())
      formData.append('model_type', modelType)
      endpoint = `${BACKEND_BASE}/generate-text`
    }

    try {
      const result = await postForm(endpoint, formData)
      if (!result.ok) {
        setError(result.data?.detail || `Generation failed (status ${result.status})`)
      } else {
        setGeneratedVideo(result.data)
      }
    } catch (err) {
      setError(err.message || 'Failed to generate video')
    } finally {
      setIsGenerating(false)
    }
  }

  const handleDownload = () => {
    if (generatedVideo) {
  const link = document.createElement('a')
  link.href = `${BACKEND_BASE}${generatedVideo.video_url}`
      link.download = generatedVideo.output_video
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
  }

  return (
    <div className="video-generator">
      {/* Tab Navigation */}
      <div className="tab-navigation">
        <button
          className={`tab-button ${activeTab === 'image' ? 'active' : ''}`}
          onClick={() => setActiveTab('image')}
        >
          <Image size={18} />
          Image to Video
        </button>
        <button
          className={`tab-button ${activeTab === 'text' ? 'active' : ''}`}
          onClick={() => setActiveTab('text')}
        >
          <FileText size={18} />
          Text to Video
        </button>
      </div>

      {activeTab === 'image' && (
        <div className="upload-section">
          <div className="upload-area" onClick={() => fileInputRef.current?.click()}>
            {preview ? (
              <div className="preview-container">
                <img src={preview} alt="Preview" className="image-preview" />
                <div className="preview-overlay">
                  <Upload size={24} />
                  <span>Click to change image</span>
                </div>
              </div>
            ) : (
              <div className="upload-placeholder">
                <Upload size={48} />
                <h3>Upload an Image</h3>
                <p>Choose an image to transform into a video</p>
                <p className="file-types">Supports: JPG, PNG, WebP</p>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
            />
          </div>
        </div>
      )}

      {activeTab === 'text' && (
        <div className="text-input-section">
          <div className="text-input-area">
            <FileText size={48} />
            <h3>Describe Your Video</h3>
            <p>Enter a detailed description of the video you want to create</p>
          </div>
        </div>
      )}

      {/* Preset Mode Toggle */}
      <div className="mode-toggle">
        <button
          className={`mode-button ${usePresets ? 'active' : ''}`}
          onClick={() => setUsePresets(true)}
        >
          <Sliders size={16} />
          Use Presets
        </button>
        <button
          className={`mode-button ${!usePresets ? 'active' : ''}`}
          onClick={() => setUsePresets(false)}
        >
          <Image size={16} />
          Manual Mode
        </button>
      </div>

      {/* Preset Selector */}
      {usePresets && (
        <PresetSelector
          onPresetChange={handlePresetChange}
          onParametersChange={handleParametersChange}
          currentParameters={presetParameters}
        />
      )}

      {/* Extend Duration - Always Visible */}
      <div className="extend-duration-section">
        <div className="extend-mode-group">
          <label className="checkbox-label extend-toggle">
            <input
              type="checkbox"
              checked={extendMode}
              onChange={(e) => {
                setExtendMode(e.target.checked)
                if (!e.target.checked) setClipCount(1)
              }}
            />
            <span className="checkmark"></span>
            🎬 Extend Duration (Sequential Clips)
          </label>
          
          {extendMode && (
            <div className="extend-slider-container">
              <label htmlFor="clipCount">
                Number of Clips: {clipCount}
                <span className="clip-duration-info">
                  ≈ {((numFrames * clipCount) / 16).toFixed(1)}s total @ 16fps
                </span>
              </label>
              <input
                id="clipCount"
                type="range"
                min="1"
                max="5"
                value={clipCount}
                onChange={(e) => setClipCount(parseInt(e.target.value))}
                step="1"
                className="clip-slider"
              />
              <div className="range-labels clip-labels">
                <span>1</span>
                <span>2</span>
                <span>3</span>
                <span>4</span>
                <span>5</span>
              </div>
              <div className="extend-explanation">
                <small>
                  🔗 Each clip continues from the last frame of the previous clip.
                  {clipCount > 1 && (
                    <> <strong>{clipCount} clips × {numFrames} frames = {numFrames * clipCount} total frames</strong></>
                  )}
                </small>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Manual Controls (when not using presets) */}
      {!usePresets && (
      <div className="controls-section">
        <div className="control-group">
          <label htmlFor="prompt">
            {activeTab === 'text' ? 'Video Description *' : 'Video Prompt (Optional)'}
          </label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder={
              activeTab === 'text'
                ? "Describe in detail the video you want to create (e.g., 'A majestic eagle soaring through a mountain landscape at sunset')"
                : "Describe the movement or animation you want..."
            }
            rows={3}
            disabled={usePoseGuidance && activeTab === 'image'}
            required={activeTab === 'text'}
          />
        </div>

        {activeTab === 'image' && (
          <div className="control-group">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={usePoseGuidance}
                onChange={(e) => setUsePoseGuidance(e.target.checked)}
              />
              <span className="checkmark"></span>
              Use Pose-Guided Generation (AI analyzes body pose for natural movement)
            </label>
          </div>
        )}

        <div className="control-group">
          <label htmlFor="model">AI Model:</label>
          <select
            id="model"
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            className="model-select"
          >
            <option value="light">🚀 Lightweight (3.5GB - Fast & Stable)</option>
            <option value="svd">🎬 Stable Video Diffusion (7GB+ - High Quality)</option>
            <option value="wan2.2">⚡ Wan2.2 Q5 DisTorch (Fast - 4 steps)</option>
            <option value="wan2.2-q6">💎 Wan2.2 Q6 Quality (Best - 8 steps)</option>
            <option value="wan2.2-enhanced">🔥 Wan2.2 Enhanced NSFW (Lightning - 4 steps)</option>
          </select>
          <div className="model-description">
            {modelType === 'light' && (
              <small>💡 <strong>Lightweight:</strong> Fast generation, lower memory usage, great for text-to-video</small>
            )}
            {modelType === 'svd' && (
              <small>💡 <strong>SVD:</strong> High-quality video from images, requires more GPU memory</small>
            )}
            {modelType === 'wan2.2' && (
              <small>💡 <strong>Wan2.2 Q5:</strong> Fast DisTorch workflow, 4 steps, good balance of speed/quality</small>
            )}
            {modelType === 'wan2.2-q6' && (
              <small>💎 <strong>Wan2.2 Q6:</strong> Higher quality 6-bit model, 8 steps, dpm++ scheduler - best visual quality</small>
            )}
            {modelType === 'wan2.2-enhanced' && (
              <small>🔥 <strong>Enhanced NSFW:</strong> Lightning-fast 4 steps, optimized for adult content with Q4KM model</small>
            )}
          </div>
        </div>

        <div className="control-group">
          <label htmlFor="frames">
            Number of Frames: {numFrames}
            <span className="frame-info">
              {activeTab === 'text'
                ? ' (8-32 frames recommended for text-to-video)'
                : ' (8-32 frames, SVD uses 25 frames)'
              }
            </span>
          </label>
          <input
            id="frames"
            type="range"
            min="8"
            max="32"
            value={numFrames}
            onChange={(e) => setNumFrames(parseInt(e.target.value))}
            step="4"
          />
          <div className="range-labels">
            <span>8</span>
            <span>16</span>
            <span>24</span>
            <span>32</span>
          </div>
          <div className="frame-explanation">
            {activeTab === 'text' ? (
              <small>
                💡 <strong>Text-to-Video:</strong> Uses AI to generate both image and motion from your description.
                Higher frames = longer, more detailed videos. Wan2.1 supports up to 32 frames.
              </small>
            ) : (
              <small>
                💡 <strong>Image-to-Video:</strong> Animates your uploaded image.
                SVD model generates exactly 25 frames. Wan2.2 supports 8-32 frames.
              </small>
            )}
          </div>
        </div>

        <button
          className="generate-btn"
          onClick={handleGenerate}
          disabled={
            (activeTab === 'image' && !selectedFile) ||
            (activeTab === 'text' && !prompt.trim()) ||
            isGenerating
          }
        >
          {isGenerating ? (
            <>
              <Loader size={20} className="spinning" />
              Generating Video...
            </>
          ) : (
            <>
              <Play size={20} />
              Generate {activeTab === 'text' ? 'Text-to-Video' : 'Video'}
            </>
          )}
        </button>
      </div>
      )}

      {/* Generate Button (always visible) */}
      {usePresets && (
        <div className="preset-generate-section">
          <button
            className="generate-btn"
            onClick={handleGenerate}
            disabled={
              (activeTab === 'image' && !selectedFile) ||
              (activeTab === 'text' && !prompt.trim()) ||
              isGenerating ||
              !selectedPreset
            }
          >
            {isGenerating ? (
              <>
                <Loader size={20} className="spinning" />
                Generating with {selectedPreset?.name || 'preset'}...
              </>
            ) : (
              <>
                <Play size={20} />
                Generate with {selectedPreset?.name || 'Selected Preset'}
              </>
            )}
          </button>
        </div>
      )}

      {error && (
        <div className="error-message">
          ❌ {error}
        </div>
      )}

      {generatedVideo && (
        <div className="result-section">
          <h3>🎉 Video Generated Successfully!</h3>
          <div className="video-info">
            <p><strong>Prompt:</strong> {generatedVideo.prompt || 'None'}</p>
            <p><strong>Frames:</strong> {generatedVideo.num_frames}</p>
            <p><strong>Generated:</strong> {new Date(generatedVideo.timestamp).toLocaleString()}</p>
          </div>

          <div className="video-player">
            <video
              controls
              src={`${BACKEND_BASE}${generatedVideo.video_url}`}
              style={{ maxWidth: '100%', maxHeight: '400px' }}
            >
              Your browser does not support the video tag.
            </video>
          </div>

          <button className="download-btn" onClick={handleDownload}>
            <Download size={20} />
            Download Video
          </button>
        </div>
      )}
    </div>
  )
}

export default VideoGenerator
