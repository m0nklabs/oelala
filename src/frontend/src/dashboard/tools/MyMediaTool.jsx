import React, { useCallback, useEffect, useState, useRef, useMemo } from 'react'
import { RefreshCw, Download, X, ChevronLeft, ChevronRight, Trash2, Check, FileJson, Image as ImageIcon, Heart, ArrowUpDown, Filter, HelpCircle, Clock, MessageCircle, Copy } from 'lucide-react'
import { BACKEND_BASE } from '../../config'

// Format video duration as MM:SS
const formatDuration = (seconds) => {
  if (!seconds || isNaN(seconds)) return null
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

// LocalStorage key for favorites
const FAVORITES_KEY = 'oelala_media_favorites'
const PROFILE_KEY = 'oelala_media_profile'

// Monitor profiles: columns only
const MONITOR_PROFILES = {
  '1280x1024': { cols: 4, label: '1280×1024' },
  '1080p': { cols: 5, label: '1080p' },
  '1440p': { cols: 6, label: '1440p' },
  '4k': { cols: 8, label: '4K' },
}

// Auto-detect best profile based on viewport width
const detectProfile = () => {
  const w = window.innerWidth
  if (w <= 1280) return '1280x1024'
  if (w <= 1920) return '1080p'
  if (w <= 2560) return '1440p'
  return '4k'
}

// Load saved profile
const loadProfile = () => {
  try {
    return localStorage.getItem(PROFILE_KEY) || 'auto'
  } catch {
    return 'auto'
  }
}

// Save profile
const saveProfile = (profile) => {
  try {
    localStorage.setItem(PROFILE_KEY, profile)
  } catch (e) {
    console.error('Failed to save profile:', e)
  }
}

// Load favorites from localStorage
const loadFavorites = () => {
  try {
    const stored = localStorage.getItem(FAVORITES_KEY)
    return stored ? new Set(JSON.parse(stored)) : new Set()
  } catch {
    return new Set()
  }
}

// Save favorites to localStorage
const saveFavorites = (favorites) => {
  try {
    localStorage.setItem(FAVORITES_KEY, JSON.stringify([...favorites]))
  } catch (e) {
    console.error('Failed to save favorites:', e)
  }
}

export default function MyMediaTool({ filter = 'all', selectionMode = false, onSelectItem = null }) {
  const [mediaList, setMediaList] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [stats, setStats] = useState({ videos: 0, images: 0 })
  const [selectedIndex, setSelectedIndex] = useState(null)
  const [selectedItems, setSelectedItems] = useState(new Set())
  const [lastClickedIndex, setLastClickedIndex] = useState(null)
  const [deleting, setDeleting] = useState(false)
  const [showMetadata, setShowMetadata] = useState(false)
  const [promptPopup, setPromptPopup] = useState(null) // { item, x, y } for prompt popup
  const [favorites, setFavorites] = useState(loadFavorites)
  const [sortBy, setSortBy] = useState('date') // 'date', 'name', 'size', 'favorites'
  const [sortOrder, setSortOrder] = useState('desc') // 'asc', 'desc'
  const [filterBy, setFilterBy] = useState('all') // 'all', 'favorites', 'non-favorites'
  const [hideStartImages, setHideStartImages] = useState(true)  // Hide start images by default
  const [profile, setProfile] = useState(loadProfile) // 'auto', '1280x1024', '1080p', '1440p', '4k'
  
  // Compute gridSize from profile
  const activeProfile = profile === 'auto' ? detectProfile() : profile
  const profileSettings = MONITOR_PROFILES[activeProfile] || MONITOR_PROFILES['1080p']
  const gridSize = profileSettings.cols
  const [showHelp, setShowHelp] = useState(false)
  const [visibleCount, setVisibleCount] = useState(100)
  const [thumbHeight, setThumbHeight] = useState(320)
  const [videoDurations, setVideoDurations] = useState({}) // filename -> duration in seconds
  const containerRef = useRef(null)

  // Calculate thumb height based on actual grid cell width (9:16 ratio)
  useEffect(() => {
    const calculateHeight = () => {
      if (containerRef.current) {
        const containerWidth = containerRef.current.clientWidth - 32 // minus padding
        const gap = 12
        const cellWidth = (containerWidth - (gap * (gridSize - 1))) / gridSize
        const height = Math.round(cellWidth * (16 / 9)) // 9:16 portrait = width * 16/9
        setThumbHeight(height)
      }
    }
    calculateHeight()
    window.addEventListener('resize', calculateHeight)
    return () => window.removeEventListener('resize', calculateHeight)
  }, [gridSize])

  // Reset visible count when filters/sort change
  useEffect(() => {
    setVisibleCount(100)
  }, [filterBy, sortBy, sortOrder, mediaList])

  const handleScroll = (e) => {
    const { scrollTop, clientHeight, scrollHeight } = e.target
    if (scrollHeight - scrollTop - clientHeight < 1000) {
      setVisibleCount(prev => Math.min(prev + 50, sortedMediaList.length))
    }
  }

  // Toggle favorite status for a media item
  const toggleFavorite = (filename, e) => {
    e?.stopPropagation()
    setFavorites(prev => {
      const newFavorites = new Set(prev)
      if (newFavorites.has(filename)) {
        newFavorites.delete(filename)
      } else {
        newFavorites.add(filename)
      }
      saveFavorites(newFavorites)
      return newFavorites
    })
  }

  // Filtered and sorted media list
  const sortedMediaList = useMemo(() => {
    // First filter
    let filtered = [...mediaList]
    if (filterBy === 'favorites') {
      filtered = filtered.filter(item => favorites.has(item.filename))
    } else if (filterBy === 'non-favorites') {
      filtered = filtered.filter(item => !favorites.has(item.filename))
    }
    
    // Then sort
    filtered.sort((a, b) => {
      let comparison = 0
      switch (sortBy) {
        case 'name':
          comparison = a.filename.localeCompare(b.filename)
          break
        case 'size':
          comparison = (a.size || 0) - (b.size || 0)
          break
        case 'favorites':
          const aFav = favorites.has(a.filename) ? 1 : 0
          const bFav = favorites.has(b.filename) ? 1 : 0
          comparison = aFav - bFav // Lower = non-fav, higher = fav
          break
        case 'non-favorites':
          const aNotFav = favorites.has(a.filename) ? 0 : 1
          const bNotFav = favorites.has(b.filename) ? 0 : 1
          comparison = aNotFav - bNotFav // Lower = fav, higher = non-fav
          break
        case 'date':
        default:
          comparison = (a.mtime || 0) - (b.mtime || 0) // Lower = older, higher = newer
          break
      }
      // desc = highest first (newest, largest, favorites first)
      // asc = lowest first (oldest, smallest, non-favorites first)
      return sortOrder === 'desc' ? -comparison : comparison
    })
    return filtered
  }, [mediaList, sortBy, sortOrder, filterBy, favorites])

  const fetchMedia = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      // Use grouped mode to pair videos with source images
      // For prompts filter, fetch all media with metadata then filter client-side
      const apiFilter = filter === 'prompts' ? 'all' : filter
      const res = await fetch(`${BACKEND_BASE}/list-comfyui-media?type=${apiFilter}&grouped=true&include_metadata=true&hide_start_images=${hideStartImages}`)
      if (!res.ok) throw new Error('Failed to fetch media')
      const data = await res.json()
      
      // For prompts view, filter to only items with prompts
      let media = data.media || []
      if (filter === 'prompts') {
        media = media.filter(item => 
          item.metadata?.positive_prompt || 
          item.metadata?.prompt
        )
      }
      
      setMediaList(media)
      setStats({ videos: data.videos || 0, images: data.images || 0 })
      setSelectedItems(new Set()) // Clear selection on refresh
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [filter, hideStartImages])

  useEffect(() => {
    fetchMedia()
  }, [fetchMedia])

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Help toggle (works everywhere)
      if (e.key === '?' || (e.key === '/' && e.shiftKey)) {
        e.preventDefault()
        setShowHelp(prev => !prev)
        return
      }
      
      // Profile cycling with +/- keys
      if (e.key === '+' || e.key === '=') {
        e.preventDefault()
        const profiles = ['auto', '1280x1024', '1080p', '1440p', '4k']
        setProfile(prev => {
          const idx = profiles.indexOf(prev)
          const next = profiles[(idx + 1) % profiles.length]
          saveProfile(next)
          return next
        })
        return
      }
      if (e.key === '-' || e.key === '_') {
        e.preventDefault()
        const profiles = ['auto', '1280x1024', '1080p', '1440p', '4k']
        setProfile(prev => {
          const idx = profiles.indexOf(prev)
          const next = profiles[(idx - 1 + profiles.length) % profiles.length]
          saveProfile(next)
          return next
        })
        return
      }
      
      if (selectedIndex === null) return
      if (e.key === 'Escape') {
        setSelectedIndex(null)
        setShowHelp(false)
      }
      if (e.key === 'ArrowLeft') setSelectedIndex(prev => prev > 0 ? prev - 1 : sortedMediaList.length - 1)
      if (e.key === 'ArrowRight') setSelectedIndex(prev => prev < sortedMediaList.length - 1 ? prev + 1 : 0)
      // F or H to toggle favorite
      if (e.key === 'f' || e.key === 'F' || e.key === 'h' || e.key === 'H') {
        const item = sortedMediaList[selectedIndex]
        if (item) toggleFavorite(item.filename)
      }
    }
    
    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [selectedIndex, sortedMediaList, favorites])

  // Handle item click with Ctrl/Shift support
  const handleItemClick = (idx, e) => {
    // If clicking checkbox area, toggle selection
    if (e.target.closest('.select-checkbox')) {
      e.stopPropagation()
      toggleSelection(idx, e)
      return
    }
    
    // In selection mode, call onSelectItem callback
    if (selectionMode && onSelectItem) {
      const item = sortedMediaList[idx]
      onSelectItem(item)
      return
    }
    
    // Regular click opens lightbox
    setSelectedIndex(idx)
  }

  const toggleSelection = (idx, e) => {
    e?.stopPropagation()
    
    setSelectedItems(prev => {
      const newSet = new Set(prev)
      
      // Shift+click: range select
      if (e?.shiftKey && lastClickedIndex !== null) {
        const start = Math.min(lastClickedIndex, idx)
        const end = Math.max(lastClickedIndex, idx)
        for (let i = start; i <= end; i++) {
          newSet.add(i)
        }
      }
      // Ctrl+click: toggle single
      else if (e?.ctrlKey || e?.metaKey) {
        if (newSet.has(idx)) {
          newSet.delete(idx)
        } else {
          newSet.add(idx)
        }
      }
      // Regular click: toggle single
      else {
        if (newSet.has(idx)) {
          newSet.delete(idx)
        } else {
          newSet.add(idx)
        }
      }
      
      return newSet
    })
    
    setLastClickedIndex(idx)
  }

  const selectAll = () => {
    setSelectedItems(new Set(mediaList.map((_, idx) => idx)))
  }

  const clearSelection = () => {
    setSelectedItems(new Set())
  }

  const handleDelete = async () => {
    if (selectedItems.size === 0) return
    
    // Check how many favorites are in selection - filter out undefined items
    const filenames = Array.from(selectedItems)
      .map(idx => sortedMediaList[idx]?.filename)
      .filter(Boolean)
    
    if (filenames.length === 0) {
      setError('No valid items selected for deletion')
      return
    }
    
    const favoritesInSelection = filenames.filter(f => favorites.has(f))
    const favCount = favoritesInSelection.length
    
    // Build confirmation message
    let message = `Delete ${filenames.length} item${filenames.length > 1 ? 's' : ''} and their associated files (source images, metadata)?`
    
    if (favCount > 0) {
      message = `⚠️ WARNING: ${favCount} favorite${favCount > 1 ? 's' : ''} selected!\n\n${message}\n\nFavorites to delete:\n• ${favoritesInSelection.slice(0, 5).join('\n• ')}${favCount > 5 ? `\n• ... and ${favCount - 5} more` : ''}`
    }
    
    const confirmed = window.confirm(message)
    if (!confirmed) return
    
    setDeleting(true)
    try {
      const res = await fetch(`${BACKEND_BASE}/delete-comfyui-media`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filenames })
      })
      
      if (!res.ok) throw new Error('Failed to delete')
      
      const result = await res.json()
      console.log('Deleted:', result)
      
      // Refresh the list
      await fetchMedia()
    } catch (err) {
      setError(`Delete failed: ${err.message}`)
    } finally {
      setDeleting(false)
    }
  }

  const handleDownload = (item, e) => {
    e?.stopPropagation()
    const link = document.createElement('a')
    link.href = `${BACKEND_BASE}${item.url}`
    link.download = item.filename
    link.click()
  }

  const handleDownloadMetadata = async (item, e) => {
    e?.stopPropagation()
    try {
      const res = await fetch(`${BACKEND_BASE}/comfyui-metadata/${item.filename}`)
      if (!res.ok) throw new Error('No metadata available')
      const data = await res.json()
      
      // Download as JSON
      const blob = new Blob([JSON.stringify(data.metadata, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `${item.base_name || item.filename.replace(/\.[^/.]+$/, '')}_metadata.json`
      link.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error('Failed to download metadata:', err)
    }
  }

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  }

  const selectedItem = selectedIndex !== null ? sortedMediaList[selectedIndex] : null
  const favoritesCount = mediaList.filter(item => favorites.has(item.filename)).length

  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100%',
      backgroundColor: 'var(--bg-primary)'
    }}>
      <style>{`
        /* ========== MEDIA GRID ========== */
        .media-grid {
          display: grid;
          gap: 12px;
          padding: 16px;
        }

        /* ========== THUMBNAIL CARD ========== */
        .thumb-card {
          position: relative;
          width: 100%;
          border-radius: 8px;
          overflow: hidden;
          cursor: pointer;
          background: #111;
        }
        .thumb-card:hover {
          outline: 2px solid var(--accent-color, #a855f7);
          z-index: 10;
        }
        .thumb-card.selected {
          outline: 3px solid var(--accent-color, #a855f7);
        }
        .thumb-card video,
        .thumb-card img {
          width: 100%;
          height: 100%;
          object-fit: cover;
          display: block;
        }

        /* ========== SELECTION CHECKBOX ========== */
        .select-checkbox {
          position: absolute;
          top: 8px;
          left: 8px;
          width: 24px;
          height: 24px;
          border-radius: 6px;
          background: rgba(0,0,0,0.7);
          border: 2px solid rgba(255,255,255,0.8);
          opacity: 0;
          transition: opacity 0.15s;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          z-index: 20;
        }
        .thumb-card:hover .select-checkbox,
        .thumb-card.selected .select-checkbox {
          opacity: 1;
        }
        .thumb-card.selected .select-checkbox {
          background: var(--accent-color, #a855f7);
          border-color: var(--accent-color, #a855f7);
        }

        /* ========== FAVORITE BUTTON ========== */
        .favorite-btn {
          position: absolute;
          top: 8px;
          left: 40px;
          width: 24px;
          height: 24px;
          border-radius: 6px;
          background: rgba(0,0,0,0.7);
          border: 2px solid rgba(255,255,255,0.8);
          opacity: 0;
          transition: opacity 0.15s;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          z-index: 20;
        }
        .thumb-card:hover .favorite-btn {
          opacity: 1;
        }
        .favorite-btn.is-favorite {
          opacity: 1;
          background: #ef4444;
          border-color: #ef4444;
        }

        /* ========== PROMPT BUBBLE BUTTON ========== */
        .prompt-bubble-btn {
          position: absolute;
          top: 6px;
          right: 6px;
          width: 24px;
          height: 24px;
          border-radius: 4px;
          background: transparent;
          border: none;
          opacity: 0;
          transition: all 0.15s ease;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          z-index: 21;
          font-size: 16px;
          line-height: 1;
          padding: 0;
          text-shadow: 0 1px 3px rgba(0,0,0,0.8);
        }
        .thumb-card:hover .prompt-bubble-btn {
          opacity: 1;
        }
        .prompt-bubble-btn:hover {
          transform: scale(1.2);
        }

        /* ========== PROMPT POPUP ========== */
        .prompt-popup-overlay {
          position: fixed;
          inset: 0;
          background: rgba(0,0,0,0.5);
          z-index: 1000;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .prompt-popup {
          background: var(--bg-secondary, #1f1f1f);
          border: 1px solid var(--border-color, #333);
          border-radius: 12px;
          padding: 20px;
          max-width: 600px;
          width: 90%;
          max-height: 80vh;
          overflow-y: auto;
          box-shadow: 0 20px 40px rgba(0,0,0,0.5);
        }
        .prompt-popup-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
          padding-bottom: 12px;
          border-bottom: 1px solid var(--border-color, #333);
        }
        .prompt-popup-title {
          font-size: 1rem;
          font-weight: 600;
          color: var(--text-primary, #fff);
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .prompt-popup-close {
          background: none;
          border: none;
          color: var(--text-muted, #888);
          cursor: pointer;
          padding: 4px;
          border-radius: 4px;
        }
        .prompt-popup-close:hover {
          background: rgba(255,255,255,0.1);
          color: var(--text-primary, #fff);
        }
        .prompt-popup-content {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }
        .prompt-section {
          background: var(--bg-tertiary, #2a2a2a);
          padding: 12px;
          border-radius: 8px;
        }
        .prompt-section-label {
          font-size: 0.75rem;
          font-weight: 600;
          color: var(--text-muted, #888);
          margin-bottom: 8px;
          display: flex;
          align-items: center;
          gap: 6px;
        }
        .prompt-section-text {
          font-size: 0.9rem;
          color: var(--text-primary, #fff);
          line-height: 1.5;
          white-space: pre-wrap;
          word-break: break-word;
        }
        .prompt-copy-btn {
          background: var(--accent-color, #a855f7);
          border: none;
          color: #fff;
          padding: 8px 16px;
          border-radius: 6px;
          cursor: pointer;
          font-size: 0.85rem;
          display: flex;
          align-items: center;
          gap: 6px;
          margin-top: 12px;
        }
        .prompt-copy-btn:hover {
          opacity: 0.9;
        }
        .prompt-media-preview {
          width: 80px;
          height: 80px;
          object-fit: cover;
          border-radius: 8px;
        }

        /* ========== SOURCE IMAGE BADGE ========== */
        .source-image-badge {
          position: absolute;
          top: 8px;
          right: 40px;
          padding: 3px 6px;
          border-radius: 4px;
          background: rgba(59, 130, 246, 0.9);
          color: #fff;
          font-size: 0.6rem;
          display: flex;
          align-items: center;
          gap: 3px;
          z-index: 20;
        }


        /* ========== MEDIA OVERLAY (hover info) ========== */
        .media-overlay {
          position: absolute;
          bottom: 0;
          left: 0;
          right: 0;
          padding: 8px;
          background: linear-gradient(transparent, rgba(0,0,0,0.8));
          opacity: 0;
          transition: opacity 0.15s;
          display: flex;
          justify-content: space-between;
          align-items: flex-end;
        }
        .thumb-card:hover .media-overlay {
          opacity: 1;
        }
        .media-filename {
          font-size: 0.7rem;
          color: #fff;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          max-width: 70%;
        }
        .media-size {
          font-size: 0.65rem;
          color: rgba(255,255,255,0.6);
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .media-duration {
          display: inline-flex;
          align-items: center;
          gap: 3px;
          background: rgba(0,0,0,0.4);
          padding: 1px 5px;
          border-radius: 3px;
        }
        .overlay-buttons {
          display: flex;
          gap: 4px;
        }
        .overlay-btn {
          padding: 4px;
          border-radius: 4px;
          background: rgba(255,255,255,0.2);
          border: none;
          color: #fff;
          cursor: pointer;
        }
        .overlay-btn:hover {
          background: rgba(255,255,255,0.3);
        }

        /* ========== LIGHTBOX ========== */
        .lightbox-overlay {
          position: fixed;
          inset: 0;
          background: rgba(0,0,0,0.95);
          z-index: 1000;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .lightbox-content {
          max-width: 90vw;
          max-height: 85vh;
          position: relative;
        }
        .lightbox-content video,
        .lightbox-content img {
          max-width: 90vw;
          max-height: 85vh;
          border-radius: 8px;
        }
        .lightbox-nav {
          position: absolute;
          top: 50%;
          transform: translateY(-50%);
          width: 48px;
          height: 48px;
          border-radius: 50%;
          background: rgba(255,255,255,0.1);
          border: none;
          color: #fff;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .lightbox-nav:hover {
          background: rgba(255,255,255,0.2);
        }
        .lightbox-close {
          position: absolute;
          top: 20px;
          right: 20px;
          width: 40px;
          height: 40px;
          border-radius: 50%;
          background: rgba(255,255,255,0.1);
          border: none;
          color: #fff;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1001;
        }
        .lightbox-close:hover {
          background: rgba(255,255,255,0.2);
        }
        .lightbox-info {
          position: absolute;
          bottom: 20px;
          left: 50%;
          transform: translateX(-50%);
          background: rgba(0,0,0,0.7);
          padding: 12px 20px;
          border-radius: 8px;
          display: flex;
          gap: 16px;
          align-items: center;
        }
        .lightbox-metadata {
          position: absolute;
          top: 20px;
          left: 20px;
          max-width: 400px;
          max-height: 60vh;
          overflow-y: auto;
          background: rgba(0,0,0,0.85);
          padding: 16px;
          border-radius: 8px;
          z-index: 1001;
        }
        .prompt-text {
          font-size: 0.85rem;
          color: rgba(255,255,255,0.9);
          line-height: 1.5;
          white-space: pre-wrap;
          word-break: break-word;
        }
        .prompt-label {
          font-size: 0.75rem;
          color: var(--accent-color, #a855f7);
          font-weight: 600;
          margin-bottom: 4px;
        }

        /* ========== BUTTONS & CONTROLS ========== */
        .delete-btn {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 6px 12px;
          border-radius: 6px;
          border: none;
          background: #dc2626;
          color: #fff;
          font-size: 0.85rem;
          cursor: pointer;
        }
        .delete-btn:hover {
          background: #b91c1c;
        }
        .delete-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        .header-btn {
          padding: 6px 10px;
          border-radius: 6px;
          border: none;
          background: rgba(255,255,255,0.1);
          color: var(--text-muted);
          font-size: 0.8rem;
          cursor: pointer;
        }
        .header-btn:hover {
          background: rgba(255,255,255,0.2);
        }
        .sort-select {
          padding: 6px 10px;
          border-radius: 6px;
          border: 1px solid var(--border-color);
          background: #1a1a1a;
          color: #e5e5e5;
          font-size: 0.8rem;
          cursor: pointer;
          outline: none;
        }
        .sort-select option {
          background: #1a1a1a;
          color: #e5e5e5;
        }
        .sort-btn {
          padding: 6px 8px;
          border-radius: 6px;
          border: none;
          background: rgba(255,255,255,0.1);
          color: var(--text-muted);
          cursor: pointer;
          display: flex;
          align-items: center;
        }
        .sort-btn:hover {
          background: rgba(255,255,255,0.2);
        }

        /* ========== ANIMATION ========== */
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>

      {/* Header with selection controls */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        padding: '12px 16px',
        borderBottom: '1px solid var(--border-color)',
        backgroundColor: 'var(--bg-secondary)',
        flexWrap: 'wrap',
        gap: '10px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
            {filter === 'all' ? 'All Media' : filter === 'video' ? 'Videos' : filter === 'image' ? 'Images' : 'Prompts'}
          </span>
          <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            {filter === 'prompts' ? (
              <>💬 {sortedMediaList.length} items with prompts</>
            ) : (
              <>🎬 {stats.videos} • 🖼️ {stats.images} • ❤️ {favoritesCount}</>
            )}
            {filterBy !== 'all' && ` • 📋 ${sortedMediaList.length} shown`}
          </span>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {/* Filter controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Filter size={14} style={{ color: 'var(--text-muted)' }} />
            <select 
              className="sort-select"
              value={filterBy}
              onChange={(e) => {
                setFilterBy(e.target.value)
                setSelectedItems(new Set()) // Clear selection when filter changes
              }}
            >
              <option value="all">All</option>
              <option value="favorites">❤️ Favorites</option>
              <option value="non-favorites">🤍 Non-favorites</option>
            </select>
            
            {/* Toggle to show/hide start images */}
            {(filter === 'all' || filter === 'image') && (
              <button
                className="sort-btn"
                onClick={() => setHideStartImages(prev => !prev)}
                title={hideStartImages ? 'Click to show video source images' : 'Hiding video source images'}
                style={{ 
                  background: !hideStartImages ? 'var(--accent-color, #a855f7)' : undefined,
                  color: !hideStartImages ? '#fff' : undefined,
                  fontSize: '0.75rem',
                  padding: '4px 8px'
                }}
              >
                📸{hideStartImages ? '' : '✓'}
              </button>
            )}
          </div>

          {/* Divider */}
          <div style={{ width: '1px', height: '20px', background: 'var(--border-color)', margin: '0 4px' }} />

          {/* Sort controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <ArrowUpDown size={14} style={{ color: 'var(--text-muted)' }} />
            <select 
              className="sort-select"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
            >
              <option value="date">Date</option>
              <option value="name">Name</option>
              <option value="size">Size</option>
              <option value="favorites">Favorites ❤️</option>
              <option value="non-favorites">Non-favorites 🤍</option>
            </select>
            <button
              className="sort-btn"
              onClick={() => setSortOrder(prev => prev === 'asc' ? 'desc' : 'asc')}
              title={sortOrder === 'asc' ? 'Ascending' : 'Descending'}
            >
              {sortOrder === 'asc' ? '↑' : '↓'}
            </button>
          </div>

          {/* Divider */}
          <div style={{ width: '1px', height: '20px', background: 'var(--border-color)', margin: '0 4px' }} />

          {/* Monitor profile selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '2px' }}>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginRight: '4px' }}>Profile:</span>
            {['auto', '1280x1024', '1080p', '1440p', '4k'].map((p) => (
              <button
                key={p}
                className="sort-btn"
                onClick={() => { setProfile(p); saveProfile(p); }}
                title={p === 'auto' ? `Auto-detect (currently ${detectProfile()})` : MONITOR_PROFILES[p]?.label || p}
                style={{ 
                  background: profile === p ? 'var(--accent-color, #a855f7)' : undefined,
                  color: profile === p ? '#fff' : undefined,
                  fontSize: '0.7rem',
                  padding: '4px 6px'
                }}
              >
                {p === 'auto' ? '⚡Auto' : (MONITOR_PROFILES[p]?.label || p)}
              </button>
            ))}
            <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem', marginLeft: '8px' }}>
              {gridSize} cols
            </span>
          </div>

          {/* Divider */}
          <div style={{ width: '1px', height: '20px', background: 'var(--border-color)', margin: '0 4px' }} />
          
          {/* Selection info and actions */}
          {selectedItems.size > 0 && (
            <>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                {selectedItems.size} selected
              </span>
              <button className="header-btn" onClick={clearSelection}>
                Clear
              </button>
              <button className="header-btn" onClick={selectAll}>
                Select All
              </button>
              <button 
                className="delete-btn" 
                onClick={handleDelete}
                disabled={deleting}
              >
                <Trash2 size={16} />
                {deleting ? 'Deleting...' : 'Delete'}
              </button>
            </>
          )}
          
          <button
            onClick={fetchMedia}
            disabled={loading}
            style={{
              padding: '8px',
              borderRadius: '6px',
              border: 'none',
              background: 'transparent',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center'
            }}
            title="Refresh"
          >
            <RefreshCw size={18} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
          </button>
          <button
            onClick={() => setShowHelp(true)}
            style={{
              padding: '6px',
              border: 'none',
              background: 'transparent',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center'
            }}
            title="Keyboard shortcuts (?)"
          >
            <HelpCircle size={18} />
          </button>
        </div>
      </div>

      {/* Keyboard Shortcuts Help Modal */}
      {showHelp && (
        <div 
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.8)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 2000
          }}
          onClick={() => setShowHelp(false)}
        >
          <div 
            style={{
              backgroundColor: 'var(--bg-primary, #1a1a1a)',
              borderRadius: '12px',
              padding: '24px',
              maxWidth: '500px',
              width: '90%',
              boxShadow: '0 20px 60px rgba(0,0,0,0.5)'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h3 style={{ margin: 0, color: 'var(--text-primary, #fff)', fontSize: '1.2rem' }}>⌨️ Keyboard Shortcuts</h3>
              <button
                onClick={() => setShowHelp(false)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                  padding: '4px'
                }}
              >
                <X size={20} />
              </button>
            </div>
            
            <div style={{ color: 'var(--text-secondary, #ccc)', fontSize: '0.9rem' }}>
              <div style={{ marginBottom: '16px' }}>
                <div style={{ color: 'var(--accent-color, #a855f7)', fontWeight: 600, marginBottom: '8px' }}>Grid View</div>
                <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr', gap: '6px 16px' }}>
                  <kbd style={{ background: '#333', padding: '2px 8px', borderRadius: '4px', fontSize: '0.85rem' }}>+</kbd>
                  <span>More columns (smaller thumbnails)</span>
                  <kbd style={{ background: '#333', padding: '2px 8px', borderRadius: '4px', fontSize: '0.85rem' }}>-</kbd>
                  <span>Fewer columns (larger thumbnails)</span>
                  <kbd style={{ background: '#333', padding: '2px 8px', borderRadius: '4px', fontSize: '0.85rem' }}>?</kbd>
                  <span>Show this help</span>
                </div>
              </div>
              
              <div style={{ marginBottom: '16px' }}>
                <div style={{ color: 'var(--accent-color, #a855f7)', fontWeight: 600, marginBottom: '8px' }}>Lightbox (Image View)</div>
                <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr', gap: '6px 16px' }}>
                  <kbd style={{ background: '#333', padding: '2px 8px', borderRadius: '4px', fontSize: '0.85rem' }}>←</kbd>
                  <span>Previous image</span>
                  <kbd style={{ background: '#333', padding: '2px 8px', borderRadius: '4px', fontSize: '0.85rem' }}>→</kbd>
                  <span>Next image</span>
                  <kbd style={{ background: '#333', padding: '2px 8px', borderRadius: '4px', fontSize: '0.85rem' }}>F / H</kbd>
                  <span>Toggle favorite ❤️</span>
                  <kbd style={{ background: '#333', padding: '2px 8px', borderRadius: '4px', fontSize: '0.85rem' }}>Esc</kbd>
                  <span>Close lightbox</span>
                </div>
              </div>
              
              <div>
                <div style={{ color: 'var(--accent-color, #a855f7)', fontWeight: 600, marginBottom: '8px' }}>Selection</div>
                <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr', gap: '6px 16px' }}>
                  <kbd style={{ background: '#333', padding: '2px 8px', borderRadius: '4px', fontSize: '0.85rem' }}>Ctrl+Click</kbd>
                  <span>Toggle single item</span>
                  <kbd style={{ background: '#333', padding: '2px 8px', borderRadius: '4px', fontSize: '0.85rem' }}>Shift+Click</kbd>
                  <span>Select range</span>
                </div>
              </div>
            </div>
            
            <div style={{ marginTop: '20px', paddingTop: '16px', borderTop: '1px solid var(--border-color, #333)', textAlign: 'center' }}>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Press <kbd style={{ background: '#333', padding: '2px 6px', borderRadius: '4px' }}>?</kbd> or <kbd style={{ background: '#333', padding: '2px 6px', borderRadius: '4px' }}>Esc</kbd> to close</span>
            </div>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{ 
          padding: '12px 16px', 
          backgroundColor: 'rgba(239, 68, 68, 0.1)', 
          color: '#ef4444',
          textAlign: 'center'
        }}>
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div style={{ 
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-muted)' 
        }}>
          <RefreshCw size={40} style={{ animation: 'spin 1s linear infinite', marginBottom: '16px' }} />
          <div>Loading media...</div>
        </div>
      )}

      {/* Empty State */}
      {!loading && mediaList.length === 0 && (
        <div style={{ 
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-muted)' 
        }}>
          <div style={{ fontSize: '4rem', marginBottom: '16px', opacity: 0.5 }}>📁</div>
          <div style={{ fontSize: '1.2rem', marginBottom: '8px' }}>No {filter === 'prompts' ? 'prompts' : filter === 'all' ? 'media' : filter + 's'} yet</div>
          <div style={{ fontSize: '0.9rem', opacity: 0.7 }}>Generated content will appear here</div>
        </div>
      )}

      {/* Prompts List View - Special layout for prompts filter */}
      {!loading && sortedMediaList.length > 0 && filter === 'prompts' && (
        <div 
          ref={containerRef}
          className="prompts-list"
          onScroll={handleScroll}
          style={{ 
            flex: 1, 
            overflowY: 'auto',
            overflowX: 'hidden',
            padding: '16px',
            display: 'flex',
            flexDirection: 'column',
            gap: '12px'
          }}
        >
          {sortedMediaList.slice(0, visibleCount).map((item, idx) => (
            <div
              key={item.filename}
              style={{
                display: 'flex',
                gap: '16px',
                padding: '16px',
                backgroundColor: 'var(--bg-secondary, #1f1f1f)',
                borderRadius: '12px',
                border: '1px solid var(--border-color, #333)',
                cursor: 'pointer',
                transition: 'border-color 0.15s',
              }}
              onClick={() => setSelectedIndex(idx)}
              onMouseEnter={(e) => e.currentTarget.style.borderColor = 'var(--accent-color, #a855f7)'}
              onMouseLeave={(e) => e.currentTarget.style.borderColor = 'var(--border-color, #333)'}
            >
              {/* Media thumbnail */}
              <div style={{ flexShrink: 0 }}>
                {item.type === 'video' ? (
                  <video
                    src={`${BACKEND_BASE}${item.url}`}
                    style={{ 
                      width: '100px', 
                      height: '100px', 
                      objectFit: 'cover', 
                      borderRadius: '8px' 
                    }}
                    autoPlay
                    loop
                    muted
                    playsInline
                  />
                ) : (
                  <img
                    src={`${BACKEND_BASE}${item.url}`}
                    alt={item.filename}
                    style={{ 
                      width: '100px', 
                      height: '100px', 
                      objectFit: 'cover', 
                      borderRadius: '8px' 
                    }}
                    loading="lazy"
                  />
                )}
              </div>
              
              {/* Prompt content */}
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'flex-start',
                  marginBottom: '8px'
                }}>
                  <div>
                    <div style={{ 
                      fontSize: '0.85rem', 
                      fontWeight: 600, 
                      color: 'var(--text-primary)',
                      marginBottom: '4px'
                    }}>
                      {item.filename}
                    </div>
                    <div style={{ 
                      fontSize: '0.75rem', 
                      color: 'var(--text-muted)' 
                    }}>
                      {item.type === 'video' ? '🎬' : '🖼️'} {formatSize(item.size)}
                      {item.metadata?.steps && ` • ${item.metadata.steps} steps`}
                      {item.metadata?.cfg && ` • CFG ${item.metadata.cfg}`}
                    </div>
                  </div>
                  
                  {/* Actions */}
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <button
                      style={{
                        background: 'var(--accent-color, #a855f7)',
                        border: 'none',
                        color: '#fff',
                        padding: '6px 12px',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontSize: '0.75rem',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px'
                      }}
                      onClick={(e) => {
                        e.stopPropagation()
                        const text = item.metadata?.positive_prompt || item.metadata?.prompt
                        navigator.clipboard.writeText(text)
                      }}
                    >
                      <Copy size={12} />
                      Copy
                    </button>
                    <button
                      className={favorites.has(item.filename) ? '' : ''}
                      style={{
                        background: favorites.has(item.filename) ? '#ef4444' : 'rgba(255,255,255,0.1)',
                        border: 'none',
                        color: '#fff',
                        padding: '6px',
                        borderRadius: '6px',
                        cursor: 'pointer'
                      }}
                      onClick={(e) => toggleFavorite(item.filename, e)}
                    >
                      <Heart 
                        size={14} 
                        fill={favorites.has(item.filename) ? '#fff' : 'none'}
                      />
                    </button>
                  </div>
                </div>
                
                {/* Prompt text */}
                <div style={{ 
                  fontSize: '0.9rem',
                  color: 'var(--text-primary)',
                  lineHeight: 1.5,
                  backgroundColor: 'var(--bg-tertiary, #2a2a2a)',
                  padding: '10px 12px',
                  borderRadius: '6px',
                  maxHeight: '100px',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  display: '-webkit-box',
                  WebkitLineClamp: 4,
                  WebkitBoxOrient: 'vertical'
                }}>
                  {item.metadata?.positive_prompt || item.metadata?.prompt}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Media Grid - Standard grid view */}
      {!loading && sortedMediaList.length > 0 && filter !== 'prompts' && (
        <div 
          ref={containerRef}
          className="media-grid"
          onScroll={handleScroll}
          style={{ 
            flex: 1, 
            overflowY: 'auto',
            overflowX: 'hidden',
            gridTemplateColumns: `repeat(${gridSize}, 1fr)`
          }}
        >
          {sortedMediaList.slice(0, visibleCount).map((item, idx) => (
            <div
              key={item.filename}
              className={`thumb-card ${selectedItems.has(idx) ? 'selected' : ''}`}
              style={{ height: `${thumbHeight}px` }}
              onClick={(e) => handleItemClick(idx, e)}
            >
              {/* Selection checkbox */}
              <div 
                className="select-checkbox"
                onClick={(e) => toggleSelection(idx, e)}
              >
                {selectedItems.has(idx) && <Check size={14} color="#fff" />}
              </div>

              {/* Favorite button */}
              <div 
                className={`favorite-btn ${favorites.has(item.filename) ? 'is-favorite' : ''}`}
                onClick={(e) => toggleFavorite(item.filename, e)}
                title={favorites.has(item.filename) ? 'Remove from favorites' : 'Add to favorites'}
              >
                <Heart 
                  size={14} 
                  color={favorites.has(item.filename) ? '#fff' : 'rgba(255,255,255,0.7)'}
                  fill={favorites.has(item.filename) ? '#fff' : 'none'}
                />
              </div>
              
              {/* Prompt bubble button - only show if item has a prompt */}
              {(item.metadata?.positive_prompt || item.metadata?.prompt) && (
                <button
                  className="prompt-bubble-btn"
                  onClick={(e) => {
                    e.stopPropagation()
                    setPromptPopup({ item })
                  }}
                  title="View prompt"
                >
                  💬
                </button>
              )}
              
              {/* Source image badge */}
              {item.has_source_image && (
                <div className="source-image-badge">
                  <ImageIcon size={10} />
                  <span>+IMG</span>
                </div>
              )}
              
              {/* Media content */}
              {item.type === 'video' ? (
                <video
                  src={`${BACKEND_BASE}${item.url}`}
                  autoPlay
                  loop
                  muted
                  playsInline
                  preload="metadata"
                  onLoadedMetadata={(e) => {
                    const duration = e.target.duration
                    if (duration && !videoDurations[item.filename]) {
                      setVideoDurations(prev => ({ ...prev, [item.filename]: duration }))
                    }
                  }}
                />
              ) : (
                <img
                  src={`${BACKEND_BASE}${item.url}`}
                  alt={item.filename}
                  loading="lazy"
                />
              )}

              <div className="media-overlay">
                <div>
                  <div className="media-filename">{item.filename}</div>
                  <div className="media-size">
                    {formatSize(item.size)}
                    {item.type === 'video' && videoDurations[item.filename] && (
                      <span className="media-duration">
                        <Clock size={10} />
                        {formatDuration(videoDurations[item.filename])}
                      </span>
                    )}
                  </div>
                </div>
                <div className="overlay-buttons">
                  {item.metadata?.has_metadata && (
                    <button 
                      className="overlay-btn"
                      onClick={(e) => handleDownloadMetadata(item, e)}
                      title="Download metadata JSON"
                    >
                      <FileJson size={14} />
                    </button>
                  )}
                  <button 
                    className="overlay-btn"
                    onClick={(e) => handleDownload(item, e)}
                    title="Download"
                  >
                    <Download size={14} />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Lightbox Modal */}
      {selectedItem && (
        <div className="lightbox-overlay" onClick={() => setSelectedIndex(null)}>
          <button className="lightbox-close" onClick={() => setSelectedIndex(null)}>
            <X size={24} />
          </button>
          
          {/* Metadata panel toggle */}
          {selectedItem.metadata?.has_metadata && (
            <button
              style={{
                position: 'absolute',
                top: '20px',
                left: '20px',
                padding: '8px 12px',
                borderRadius: '6px',
                background: showMetadata ? 'var(--accent-color, #a855f7)' : 'rgba(255,255,255,0.1)',
                border: 'none',
                color: '#fff',
                cursor: 'pointer',
                fontSize: '0.85rem',
                zIndex: 1002
              }}
              onClick={(e) => {
                e.stopPropagation()
                setShowMetadata(!showMetadata)
              }}
            >
              {showMetadata ? 'Hide Prompt' : 'Show Prompt'}
            </button>
          )}
          
          {/* Metadata panel */}
          {showMetadata && selectedItem.metadata && (
            <div className="lightbox-metadata" onClick={(e) => e.stopPropagation()}>
              {selectedItem.metadata.positive_prompt && (
                <div style={{ marginBottom: '16px' }}>
                  <div className="prompt-label">✨ Positive Prompt</div>
                  <div className="prompt-text">{selectedItem.metadata.positive_prompt}</div>
                </div>
              )}
              {selectedItem.metadata.negative_prompt && (
                <div>
                  <div className="prompt-label">🚫 Negative Prompt</div>
                  <div className="prompt-text" style={{ color: 'rgba(255,255,255,0.6)' }}>
                    {selectedItem.metadata.negative_prompt}
                  </div>
                </div>
              )}
            </div>
          )}
          
          {/* Navigation */}
          <button 
            className="lightbox-nav" 
            style={{ left: '20px' }}
            onClick={(e) => {
              e.stopPropagation()
              setSelectedIndex(prev => prev > 0 ? prev - 1 : sortedMediaList.length - 1)
            }}
          >
            <ChevronLeft size={28} />
          </button>
          
          <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
            {selectedItem.type === 'video' ? (
              <video
                src={`${BACKEND_BASE}${selectedItem.url}`}
                autoPlay
                loop
                controls
                style={{ borderRadius: '12px' }}
              />
            ) : (
              <img
                src={`${BACKEND_BASE}${selectedItem.url}`}
                alt={selectedItem.filename}
                style={{ borderRadius: '12px' }}
              />
            )}
          </div>
          
          <button 
            className="lightbox-nav" 
            style={{ right: '20px' }}
            onClick={(e) => {
              e.stopPropagation()
              setSelectedIndex(prev => prev < sortedMediaList.length - 1 ? prev + 1 : 0)
            }}
          >
            <ChevronRight size={28} />
          </button>

          {/* Info bar */}
          <div className="lightbox-info">
            <span style={{ color: '#fff', fontWeight: 500 }}>{selectedItem.filename}</span>
            <span style={{ color: 'rgba(255,255,255,0.6)' }}>{formatSize(selectedItem.size)}</span>
            {favorites.has(selectedItem.filename) && (
              <span style={{ color: '#ef4444', fontSize: '0.8rem' }}>❤️ Favorite</span>
            )}
            {selectedItem.has_source_image && (
              <span style={{ color: '#3b82f6', fontSize: '0.8rem' }}>📷 Has source image</span>
            )}
            <span style={{ color: 'rgba(255,255,255,0.5)' }}>{selectedIndex + 1} / {sortedMediaList.length}</span>
            <div style={{ display: 'flex', gap: '8px' }}>
              {/* Favorite toggle in lightbox */}
              <button 
                className="overlay-btn"
                onClick={(e) => toggleFavorite(selectedItem.filename, e)}
                title={favorites.has(selectedItem.filename) ? 'Remove from favorites' : 'Add to favorites'}
                style={{ background: favorites.has(selectedItem.filename) ? 'rgba(239, 68, 68, 0.5)' : undefined }}
              >
                <Heart 
                  size={16} 
                  fill={favorites.has(selectedItem.filename) ? '#ef4444' : 'none'}
                  color={favorites.has(selectedItem.filename) ? '#ef4444' : '#fff'}
                />
              </button>
              {selectedItem.has_source_image && selectedItem.source_image && (
                <button 
                  className="overlay-btn"
                  onClick={(e) => handleDownload(selectedItem.source_image, e)}
                  title="Download source image"
                >
                  <ImageIcon size={16} />
                </button>
              )}
              {selectedItem.metadata?.has_metadata && (
                <button 
                  className="overlay-btn"
                  onClick={(e) => handleDownloadMetadata(selectedItem, e)}
                  title="Download metadata JSON"
                >
                  <FileJson size={16} />
                </button>
              )}
              <button 
                className="overlay-btn"
                onClick={(e) => handleDownload(selectedItem, e)}
                title="Download"
              >
                <Download size={16} />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Prompt Popup Modal */}
      {promptPopup && (
        <div 
          className="prompt-popup-overlay" 
          onClick={() => setPromptPopup(null)}
        >
          <div 
            className="prompt-popup" 
            onClick={(e) => e.stopPropagation()}
          >
            <div className="prompt-popup-header">
              <div className="prompt-popup-title">
                <MessageCircle size={18} />
                Prompt Details
              </div>
              <button 
                className="prompt-popup-close"
                onClick={() => setPromptPopup(null)}
              >
                <X size={20} />
              </button>
            </div>
            
            <div className="prompt-popup-content">
              {/* Media preview */}
              <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                {promptPopup.item.type === 'video' ? (
                  <video
                    src={`${BACKEND_BASE}${promptPopup.item.url}`}
                    className="prompt-media-preview"
                    autoPlay
                    loop
                    muted
                    playsInline
                  />
                ) : (
                  <img
                    src={`${BACKEND_BASE}${promptPopup.item.url}`}
                    alt={promptPopup.item.filename}
                    className="prompt-media-preview"
                  />
                )}
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                    {promptPopup.item.filename}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                    {promptPopup.item.type === 'video' ? '🎬 Video' : '🖼️ Image'} • {formatSize(promptPopup.item.size)}
                    {promptPopup.item.type === 'video' && videoDurations[promptPopup.item.filename] && (
                      <> • {formatDuration(videoDurations[promptPopup.item.filename])}</>
                    )}
                    {promptPopup.item.metadata?.width && promptPopup.item.metadata?.height && (
                      <> • {promptPopup.item.metadata.width}×{promptPopup.item.metadata.height}</>
                    )}
                  </div>
                </div>
              </div>
              
              {/* Positive Prompt */}
              {(promptPopup.item.metadata?.positive_prompt || promptPopup.item.metadata?.prompt) && (
                <div className="prompt-section">
                  <div className="prompt-section-label">
                    ✨ Positive Prompt
                  </div>
                  <div className="prompt-section-text">
                    {promptPopup.item.metadata.positive_prompt || promptPopup.item.metadata.prompt}
                  </div>
                  <button 
                    className="prompt-copy-btn"
                    onClick={() => {
                      const text = promptPopup.item.metadata.positive_prompt || promptPopup.item.metadata.prompt
                      navigator.clipboard.writeText(text)
                    }}
                  >
                    <Copy size={14} />
                    Copy Prompt
                  </button>
                </div>
              )}
              
              {/* Negative Prompt */}
              {promptPopup.item.metadata?.negative_prompt && (
                <div className="prompt-section">
                  <div className="prompt-section-label">
                    🚫 Negative Prompt
                  </div>
                  <div className="prompt-section-text" style={{ color: 'var(--text-muted)' }}>
                    {promptPopup.item.metadata.negative_prompt}
                  </div>
                </div>
              )}
              
              {/* Generation settings if available */}
              {(promptPopup.item.metadata?.steps || promptPopup.item.metadata?.cfg || promptPopup.item.metadata?.seed || promptPopup.item.metadata?.sampler || promptPopup.item.metadata?.model) && (
                <div className="prompt-section">
                  <div className="prompt-section-label">
                    ⚙️ Generation Settings
                  </div>
                  <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', fontSize: '0.85rem' }}>
                    {promptPopup.item.metadata.steps && (
                      <span>Steps: <strong>{promptPopup.item.metadata.steps}</strong></span>
                    )}
                    {promptPopup.item.metadata.cfg && (
                      <span>CFG: <strong>{promptPopup.item.metadata.cfg}</strong></span>
                    )}
                    {promptPopup.item.metadata.seed && (
                      <span>Seed: <strong>{promptPopup.item.metadata.seed}</strong></span>
                    )}
                    {promptPopup.item.metadata.sampler && (
                      <span>Sampler: <strong>{promptPopup.item.metadata.sampler}</strong></span>
                    )}
                    {promptPopup.item.metadata.scheduler && (
                      <span>Scheduler: <strong>{promptPopup.item.metadata.scheduler}</strong></span>
                    )}
                  </div>
                  {promptPopup.item.metadata.model && (
                    <div style={{ marginTop: '8px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      Model: <strong style={{ color: 'var(--text-primary)' }}>{promptPopup.item.metadata.model}</strong>
                    </div>
                  )}
                </div>
              )}
              
              {/* LoRAs used */}
              {promptPopup.item.metadata?.loras && promptPopup.item.metadata.loras.length > 0 && (
                <div className="prompt-section">
                  <div className="prompt-section-label">
                    🎨 LoRAs Used
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.85rem' }}>
                    {promptPopup.item.metadata.loras.map((lora, idx) => (
                      <div key={idx} style={{ 
                        display: 'flex', 
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: '6px 10px',
                        backgroundColor: 'var(--bg-secondary)',
                        borderRadius: '4px'
                      }}>
                        <span style={{ 
                          fontFamily: 'monospace', 
                          fontSize: '0.8rem',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          maxWidth: '80%'
                        }}>
                          {lora.name}
                        </span>
                        <span style={{ 
                          color: 'var(--accent-color, #a855f7)', 
                          fontWeight: 600,
                          fontSize: '0.8rem'
                        }}>
                          {(lora.strength * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
