import React, { useState, useEffect, useMemo, useCallback } from 'react'
import { Search, RefreshCw, Filter, X, ChevronDown, ChevronUp, Layers, HardDrive, Tag } from 'lucide-react'
import { apiFetch } from '../api'

/**
 * Visual LoRA Browser with search, filter, and selection.
 *
 * Props:
 *   onSelect(lora) — called when user clicks a LoRA card
 *   selectedLoras — array of currently selected lora paths
 *   mode — "browse" (standalone) or "picker" (embedded in tool)
 */
export default function LoraBrowser({ onSelect, selectedLoras = [], mode = 'browse' }) {
  const [loras, setLoras] = useState([])
  const [categories, setCategories] = useState([])
  const [tags, setTags] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [activeCategory, setActiveCategory] = useState(null)
  const [activeTag, setActiveTag] = useState(null)
  const [activeBaseModel, setActiveBaseModel] = useState(null)
  const [activeNoise, setActiveNoise] = useState(null)
  const [sortBy, setSortBy] = useState('name')
  const [showFilters, setShowFilters] = useState(false)
  const [page, setPage] = useState(1)
  const perPage = 50

  // Fetch LoRAs
  const fetchLoras = useCallback(async (forceRefresh = false) => {
    setLoading(true)
    setError(null)
    try {
      if (forceRefresh) {
        await apiFetch('/api/loras/refresh', { method: 'POST' })
      }

      const params = new URLSearchParams()
      if (searchQuery) params.set('q', searchQuery)
      if (activeCategory) params.set('category', activeCategory)
      if (activeTag) params.set('tag', activeTag)
      if (activeBaseModel) params.set('base_model', activeBaseModel)
      if (activeNoise) params.set('noise', activeNoise)
      params.set('sort', sortBy)
      params.set('page', page.toString())
      params.set('per_page', perPage.toString())

      const resp = await apiFetch(`/api/loras?${params.toString()}`)
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json()
      setLoras(data.items || [])
      setCategories(data.categories || [])
      setTags(data.tags || [])
    } catch (err) {
      setError(err.message || 'Failed to load LoRAs')
    } finally {
      setLoading(false)
    }
  }, [searchQuery, activeCategory, activeTag, activeBaseModel, activeNoise, sortBy, page])

  useEffect(() => {
    fetchLoras()
  }, [fetchLoras])

  // Reset page when filters change
  useEffect(() => {
    setPage(1)
  }, [searchQuery, activeCategory, activeTag, activeBaseModel, activeNoise, sortBy])

  // Debounced search
  const [searchInput, setSearchInput] = useState('')
  useEffect(() => {
    const timer = setTimeout(() => setSearchQuery(searchInput), 300)
    return () => clearTimeout(timer)
  }, [searchInput])

  const selectedSet = useMemo(() => new Set(selectedLoras), [selectedLoras])

  const clearFilters = () => {
    setSearchInput('')
    setSearchQuery('')
    setActiveCategory(null)
    setActiveTag(null)
    setActiveBaseModel(null)
    setActiveNoise(null)
    setSortBy('name')
    setPage(1)
  }

  const hasActiveFilters = activeCategory || activeTag || activeBaseModel || activeNoise || searchQuery

  // Unique base models from tags
  const baseModels = useMemo(() => {
    return tags
      .filter(t => ['wan2.2', 'sdxl', 'pony', 'sd1.5'].includes(t.name))
      .map(t => t.name)
  }, [tags])

  return (
    <div className={`flex flex-col h-full ${mode === 'browse' ? 'p-4' : ''}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Layers className="w-5 h-5 text-purple-400" />
          <h2 className="text-lg font-semibold text-white">LoRA Browser</h2>
          <span className="text-xs text-zinc-500 bg-zinc-800 px-2 py-0.5 rounded-full">
            {loras.length} models
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`p-1.5 rounded transition-colors ${
              showFilters || hasActiveFilters
                ? 'bg-purple-600/30 text-purple-400'
                : 'text-zinc-400 hover:text-white hover:bg-zinc-700'
            }`}
            title="Toggle filters"
          >
            <Filter className="w-4 h-4" />
          </button>
          <button
            onClick={() => fetchLoras(true)}
            className="p-1.5 text-zinc-400 hover:text-white hover:bg-zinc-700 rounded transition-colors"
            title="Refresh LoRA list"
            disabled={loading}
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Search */}
      <div className="relative mb-3">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
        <input
          type="text"
          value={searchInput}
          onChange={(e) => setSearchInput(e.target.value)}
          placeholder="Search LoRAs by name, tag, or category..."
          className="w-full bg-zinc-800 border border-zinc-700 rounded-lg pl-10 pr-8 py-2 text-sm text-white placeholder-zinc-500 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500/50"
        />
        {searchInput && (
          <button
            onClick={() => { setSearchInput(''); setSearchQuery(''); }}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-white"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Filters Panel */}
      {showFilters && (
        <div className="mb-3 p-3 bg-zinc-800/50 border border-zinc-700 rounded-lg space-y-3">
          {/* Category filter */}
          <div>
            <label className="text-xs font-medium text-zinc-400 mb-1 block">Category</label>
            <div className="flex flex-wrap gap-1.5">
              <button
                onClick={() => setActiveCategory(null)}
                className={`px-2 py-0.5 text-xs rounded-full transition-colors ${
                  !activeCategory
                    ? 'bg-purple-600 text-white'
                    : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600'
                }`}
              >
                All
              </button>
              {categories.map(cat => (
                <button
                  key={cat.name}
                  onClick={() => setActiveCategory(cat.name === activeCategory ? null : cat.name)}
                  className={`px-2 py-0.5 text-xs rounded-full transition-colors ${
                    activeCategory === cat.name
                      ? 'bg-purple-600 text-white'
                      : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600'
                  }`}
                >
                  {cat.name} ({cat.count})
                </button>
              ))}
            </div>
          </div>

          {/* Base model filter */}
          {baseModels.length > 0 && (
            <div>
              <label className="text-xs font-medium text-zinc-400 mb-1 block">Base Model</label>
              <div className="flex flex-wrap gap-1.5">
                <button
                  onClick={() => setActiveBaseModel(null)}
                  className={`px-2 py-0.5 text-xs rounded-full transition-colors ${
                    !activeBaseModel
                      ? 'bg-purple-600 text-white'
                      : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600'
                  }`}
                >
                  All
                </button>
                {baseModels.map(model => (
                  <button
                    key={model}
                    onClick={() => setActiveBaseModel(model === activeBaseModel ? null : model)}
                    className={`px-2 py-0.5 text-xs rounded-full transition-colors ${
                      activeBaseModel === model
                        ? 'bg-purple-600 text-white'
                        : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600'
                    }`}
                  >
                    {model}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Noise level filter */}
          <div>
            <label className="text-xs font-medium text-zinc-400 mb-1 block">Noise Level</label>
            <div className="flex gap-1.5">
              {[null, 'high', 'low'].map(level => (
                <button
                  key={level || 'all'}
                  onClick={() => setActiveNoise(level)}
                  className={`px-2 py-0.5 text-xs rounded-full transition-colors ${
                    activeNoise === level
                      ? 'bg-purple-600 text-white'
                      : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600'
                  }`}
                >
                  {level || 'All'}
                </button>
              ))}
            </div>
          </div>

          {/* Tags */}
          <div>
            <label className="text-xs font-medium text-zinc-400 mb-1 block">Tags</label>
            <div className="flex flex-wrap gap-1.5 max-h-20 overflow-y-auto">
              {tags.filter(t => !['wan2.2', 'sdxl', 'pony', 'sd1.5'].includes(t.name)).map(tag => (
                <button
                  key={tag.name}
                  onClick={() => setActiveTag(tag.name === activeTag ? null : tag.name)}
                  className={`px-2 py-0.5 text-xs rounded-full transition-colors ${
                    activeTag === tag.name
                      ? 'bg-purple-600 text-white'
                      : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600'
                  }`}
                >
                  {tag.name} ({tag.count})
                </button>
              ))}
            </div>
          </div>

          {/* Sort + Clear */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <label className="text-xs text-zinc-400">Sort:</label>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="bg-zinc-700 border border-zinc-600 rounded text-xs text-white px-2 py-1 focus:outline-none"
              >
                <option value="name">Name</option>
                <option value="size">Size</option>
                <option value="modified">Recently Modified</option>
              </select>
            </div>
            {hasActiveFilters && (
              <button
                onClick={clearFilters}
                className="text-xs text-purple-400 hover:text-purple-300"
              >
                Clear all filters
              </button>
            )}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mb-3 p-2 bg-red-900/30 border border-red-800 rounded text-sm text-red-300">
          {error}
        </div>
      )}

      {/* LoRA Grid */}
      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-6 h-6 text-purple-400 animate-spin" />
          </div>
        ) : loras.length === 0 ? (
          <div className="text-center py-12 text-zinc-500">
            <Layers className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No LoRAs found</p>
            {hasActiveFilters && (
              <button
                onClick={clearFilters}
                className="mt-2 text-xs text-purple-400 hover:text-purple-300"
              >
                Clear filters
              </button>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
            {loras.map(lora => {
              const isSelected = selectedSet.has(lora.path)
              return (
                <button
                  key={lora.id}
                  onClick={() => onSelect?.(lora)}
                  className={`text-left p-3 rounded-lg border transition-all hover:scale-[1.01] ${
                    isSelected
                      ? 'bg-purple-600/20 border-purple-500 ring-1 ring-purple-500/50'
                      : 'bg-zinc-800/50 border-zinc-700 hover:border-zinc-500 hover:bg-zinc-800'
                  }`}
                >
                  {/* Name */}
                  <div className="font-medium text-sm text-white truncate" title={lora.name}>
                    {lora.name}
                  </div>

                  {/* Meta row */}
                  <div className="flex items-center gap-2 mt-1.5">
                    <span className="text-xs text-zinc-500 flex items-center gap-1">
                      <HardDrive className="w-3 h-3" />
                      {lora.size_mb} MB
                    </span>
                    {lora.category !== 'root' && (
                      <span className="text-xs text-zinc-500">
                        📁 {lora.category}
                      </span>
                    )}
                    {lora.noise_level && (
                      <span className={`text-xs px-1.5 py-0.5 rounded ${
                        lora.noise_level === 'high'
                          ? 'bg-orange-900/40 text-orange-400'
                          : 'bg-blue-900/40 text-blue-400'
                      }`}>
                        {lora.noise_level}
                      </span>
                    )}
                  </div>

                  {/* Tags */}
                  {lora.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-1.5">
                      {lora.tags.slice(0, 4).map(tag => (
                        <span
                          key={tag}
                          className="text-[10px] px-1.5 py-0.5 rounded-full bg-zinc-700/50 text-zinc-400"
                        >
                          {tag}
                        </span>
                      ))}
                      {lora.tags.length > 4 && (
                        <span className="text-[10px] text-zinc-500">
                          +{lora.tags.length - 4}
                        </span>
                      )}
                    </div>
                  )}

                  {/* Selected indicator */}
                  {isSelected && (
                    <div className="mt-2 text-xs text-purple-400 font-medium">
                      ✓ Selected
                    </div>
                  )}
                </button>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
