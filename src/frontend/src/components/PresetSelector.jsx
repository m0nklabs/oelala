import { useState, useEffect } from 'react'
import { Settings, ChevronDown, ChevronUp, Sliders, Sparkles, Zap, Film } from 'lucide-react'
import BACKEND_BASE from '../config'
import './PresetSelector.css'

/**
 * PresetSelector - Component for selecting and configuring workflow presets
 *
 * Presets are workflow configurations that can be loaded to customize
 * video generation parameters like steps, CFG, seed, dimensions, etc.
 */
function PresetSelector({ onPresetChange, onParametersChange, _currentParameters }) {
  const [presets, setPresets] = useState([])
  const [selectedPreset, setSelectedPreset] = useState(null)
  const [parameters, setParameters] = useState({})
  const [isExpanded, setIsExpanded] = useState(true)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)

  // Fetch presets from backend on mount
  useEffect(() => {
    fetchPresets()
  }, [])

  const fetchPresets = async () => {
    try {
      setIsLoading(true)
      const response = await fetch(`${BACKEND_BASE}/api/presets`)
      if (!response.ok) throw new Error('Failed to fetch presets')
      const data = await response.json()
      setPresets(data.presets || [])

      // Auto-select first preset if available
      if (data.presets?.length > 0) {
        const firstPreset = data.presets[0]
        setSelectedPreset(firstPreset)
        initializeParameters(firstPreset)
      }
    } catch (err) {
      console.error('Failed to load presets:', err)
      setError(err.message)
      // Fall back to default presets
      setPresets(getDefaultPresets())
    } finally {
      setIsLoading(false)
    }
  }

  const initializeParameters = (preset) => {
    if (!preset?.parameters) return

    const initialParams = {}
    Object.entries(preset.parameters).forEach(([key, config]) => {
      // Skip image parameter - handled separately
      if (config.type === 'image') return
      initialParams[key] = config.default ?? config.value ?? ''
    })
    setParameters(initialParams)
    onParametersChange?.(initialParams)
  }

  const handlePresetSelect = (preset) => {
    setSelectedPreset(preset)
    initializeParameters(preset)
    onPresetChange?.(preset)
  }

  const handleParameterChange = (key, value, config) => {
    // Type coercion based on parameter type
    let typedValue = value
    if (config.type === 'integer') {
      typedValue = parseInt(value, 10)
    } else if (config.type === 'float') {
      typedValue = parseFloat(value)
    }

    const newParams = { ...parameters, [key]: typedValue }
    setParameters(newParams)
    onParametersChange?.(newParams)
  }

  const getCategoryIcon = (category) => {
    switch (category) {
      case 'ImageToVideo': return <Film size={16} />
      case 'TextToVideo': return <Sparkles size={16} />
      case 'TextToImage': return <Zap size={16} />
      default: return <Settings size={16} />
    }
  }

  const getPresetBadge = (preset) => {
    if (preset.name?.toLowerCase().includes('lightning') ||
        preset.name?.toLowerCase().includes('fast')) {
      return <span className="preset-badge fast">⚡ Fast</span>
    }
    if (preset.name?.toLowerCase().includes('quality') ||
        preset.name?.toLowerCase().includes('q6')) {
      return <span className="preset-badge quality">💎 Quality</span>
    }
    if (preset.name?.toLowerCase().includes('nsfw') ||
        preset.name?.toLowerCase().includes('enhanced')) {
      return <span className="preset-badge nsfw">🔥 Enhanced</span>
    }
    return null
  }

  const renderParameterInput = (key, config) => {
    const value = parameters[key] ?? config.default ?? ''

    // Skip image type - handled by upload component
    if (config.type === 'image') return null

    // String/textarea
    if (config.type === 'string') {
      return (
        <div className="param-group" key={key}>
          <label htmlFor={`param-${key}`}>
            {config.label || key}
            {config.description && (
              <span className="param-hint" title={config.description}>ℹ️</span>
            )}
          </label>
          <textarea
            id={`param-${key}`}
            value={value}
            onChange={(e) => handleParameterChange(key, e.target.value, config)}
            placeholder={config.description}
            rows={key.includes('prompt') ? 3 : 1}
          />
        </div>
      )
    }

    // Integer with range
    if (config.type === 'integer' && config.min !== undefined && config.max !== undefined) {
      return (
        <div className="param-group" key={key}>
          <label htmlFor={`param-${key}`}>
            {config.label || key}: <span className="param-value">{value}</span>
            {config.description && (
              <span className="param-hint" title={config.description}>ℹ️</span>
            )}
          </label>
          <input
            id={`param-${key}`}
            type="range"
            min={config.min}
            max={config.max}
            step={config.step || 1}
            value={value}
            onChange={(e) => handleParameterChange(key, e.target.value, config)}
          />
          <div className="range-labels">
            <span>{config.min}</span>
            <span>{config.max}</span>
          </div>
        </div>
      )
    }

    // Float with range
    if (config.type === 'float' && config.min !== undefined && config.max !== undefined) {
      return (
        <div className="param-group" key={key}>
          <label htmlFor={`param-${key}`}>
            {config.label || key}: <span className="param-value">{value.toFixed?.(2) || value}</span>
            {config.description && (
              <span className="param-hint" title={config.description}>ℹ️</span>
            )}
          </label>
          <input
            id={`param-${key}`}
            type="range"
            min={config.min}
            max={config.max}
            step={config.step || 0.1}
            value={value}
            onChange={(e) => handleParameterChange(key, e.target.value, config)}
          />
          <div className="range-labels">
            <span>{config.min}</span>
            <span>{config.max}</span>
          </div>
        </div>
      )
    }

    // Plain integer/float
    if (config.type === 'integer' || config.type === 'float') {
      return (
        <div className="param-group" key={key}>
          <label htmlFor={`param-${key}`}>
            {config.label || key}
            {config.description && (
              <span className="param-hint" title={config.description}>ℹ️</span>
            )}
          </label>
          <input
            id={`param-${key}`}
            type="number"
            value={value}
            onChange={(e) => handleParameterChange(key, e.target.value, config)}
            step={config.step || (config.type === 'float' ? 0.1 : 1)}
          />
        </div>
      )
    }

    // Boolean
    if (config.type === 'boolean') {
      return (
        <div className="param-group checkbox" key={key}>
          <label htmlFor={`param-${key}`}>
            <input
              id={`param-${key}`}
              type="checkbox"
              checked={!!value}
              onChange={(e) => handleParameterChange(key, e.target.checked, config)}
            />
            {config.label || key}
            {config.description && (
              <span className="param-hint" title={config.description}>ℹ️</span>
            )}
          </label>
        </div>
      )
    }

    return null
  }

  // Group parameters by category
  const groupedParameters = () => {
    if (!selectedPreset?.parameters) return {}

    const groups = {
      prompt: [],
      generation: [],
      dimensions: [],
      other: []
    }

    Object.entries(selectedPreset.parameters).forEach(([key, config]) => {
      if (config.type === 'image') return // Skip image

      if (key.includes('prompt')) {
        groups.prompt.push([key, config])
      } else if (['steps', 'cfg', 'seed', 'frame_rate'].includes(key)) {
        groups.generation.push([key, config])
      } else if (['width', 'height', 'num_frames'].includes(key)) {
        groups.dimensions.push([key, config])
      } else {
        groups.other.push([key, config])
      }
    })

    return groups
  }

  if (isLoading) {
    return (
      <div className="preset-selector loading">
        <Sliders className="spinning" size={24} />
        <span>Loading presets...</span>
      </div>
    )
  }

  return (
    <div className="preset-selector">
      {/* Preset Header */}
      <div className="preset-header" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="preset-title">
          <Sliders size={20} />
          <span>Workflow Preset</span>
          {selectedPreset && (
            <span className="selected-preset-name">{selectedPreset.name}</span>
          )}
        </div>
        {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
      </div>

      {isExpanded && (
        <div className="preset-content">
          {/* Preset Selection */}
          <div className="preset-list">
            {presets.map((preset) => (
              <div
                key={preset.id}
                className={`preset-card ${selectedPreset?.id === preset.id ? 'selected' : ''}`}
                onClick={() => handlePresetSelect(preset)}
              >
                <div className="preset-card-header">
                  {getCategoryIcon(preset.category)}
                  <span className="preset-name">{preset.name}</span>
                  {getPresetBadge(preset)}
                </div>
                <p className="preset-description">{preset.description}</p>
              </div>
            ))}
          </div>

          {/* Parameters */}
          {selectedPreset && (
            <div className="preset-parameters">
              <h4><Settings size={16} /> Parameters</h4>

              {/* Prompt Section */}
              {groupedParameters().prompt?.length > 0 && (
                <div className="param-section">
                  <h5>📝 Prompts</h5>
                  {groupedParameters().prompt.map(([key, config]) =>
                    renderParameterInput(key, config)
                  )}
                </div>
              )}

              {/* Generation Settings */}
              {groupedParameters().generation?.length > 0 && (
                <div className="param-section">
                  <h5>⚙️ Generation</h5>
                  <div className="param-grid">
                    {groupedParameters().generation.map(([key, config]) =>
                      renderParameterInput(key, config)
                    )}
                  </div>
                </div>
              )}

              {/* Dimensions */}
              {groupedParameters().dimensions?.length > 0 && (
                <div className="param-section">
                  <h5>📐 Dimensions</h5>
                  <div className="param-grid">
                    {groupedParameters().dimensions.map(([key, config]) =>
                      renderParameterInput(key, config)
                    )}
                  </div>
                </div>
              )}

              {/* Other */}
              {groupedParameters().other?.length > 0 && (
                <div className="param-section">
                  <h5>🔧 Other</h5>
                  {groupedParameters().other.map(([key, config]) =>
                    renderParameterInput(key, config)
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="preset-error">
          ⚠️ {error} - Using default presets
        </div>
      )}
    </div>
  )
}

// Default presets fallback if API fails. The legacy Wan 2.2 presets were
// removed together with the retired Wan 2.2 model family.
function getDefaultPresets() {
  return []
}

export default PresetSelector
