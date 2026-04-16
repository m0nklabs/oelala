import { useState, useEffect, useCallback, useRef } from 'react'
import { DEBUG } from '../config'
import { apiFetch } from '../api'

/**
 * useAdapterInfo — Fetch and cache adapter metadata from `/v2/adapters`.
 *
 * Returns the full adapter list with constraints, plus helpers for
 * looking up adapters by name, filtering by operation/media type, and
 * reading per-adapter constraints for dynamic UI rendering.
 *
 * Usage:
 *   const { adapters, getAdapter, findAdapters, isLoading, error, refresh } = useAdapterInfo()
 *
 *   // All adapters
 *   adapters.forEach(a => console.log(a.name, a.constraints))
 *
 *   // Find T2I adapters
 *   const t2iAdapters = findAdapters({ operation: 'generate', output_type: 'image' })
 *
 *   // Get specific adapter constraints
 *   const sdxl = getAdapter('sdxl-local-t2i')
 *   if (sdxl) {
 *     console.log(sdxl.constraints.max_width)       // 2048
 *     console.log(sdxl.constraints.supported_samplers) // ['dpmpp_2m', 'euler', ...]
 *     console.log(sdxl.constraints.max_loras)        // 3
 *   }
 *
 * Adapters are cached in memory (sessionStorage fallback). Call refresh()
 * to force a re-fetch.
 */

const CACHE_KEY = 'oelala_v2_adapters'
const CACHE_TTL_MS = 5 * 60 * 1000 // 5 minutes

export default function useAdapterInfo() {
  const [adapters, setAdapters] = useState(() => {
    // Try loading from session cache on initial render
    try {
      const cached = sessionStorage.getItem(CACHE_KEY)
      if (cached) {
        const { data, timestamp } = JSON.parse(cached)
        if (Date.now() - timestamp < CACHE_TTL_MS) {
          return data
        }
      }
    } catch { /* ignore corrupt cache */ }
    return []
  })
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const fetchedRef = useRef(false)

  /**
   * Fetch adapter list from backend.
   */
  const fetchAdapters = useCallback(async () => {
    setIsLoading(true)
    setError(null)

    try {
      if (DEBUG) console.log('📦 Fetching V2 adapters...')

      const res = await apiFetch('/v2/adapters')
      if (!res.ok) {
        throw new Error(`Failed to fetch adapters: HTTP ${res.status}`)
      }

      const body = await res.json()
      const adapterList = body.adapters || []

      if (DEBUG) console.log(`📦 Loaded ${adapterList.length} adapters`)

      setAdapters(adapterList)

      // Cache in sessionStorage
      try {
        sessionStorage.setItem(CACHE_KEY, JSON.stringify({
          data: adapterList,
          timestamp: Date.now(),
        }))
      } catch { /* quota exceeded — ignore */ }

      return adapterList
    } catch (err) {
      console.error('❌ Failed to load adapters:', err)
      setError(err.message)
      return []
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Fetch on mount if cache is empty or expired
  useEffect(() => {
    if (fetchedRef.current) return
    fetchedRef.current = true

    // Only fetch if we don't have cached data
    if (adapters.length === 0) {
      fetchAdapters()
    }
  }, [adapters.length, fetchAdapters])

  /**
   * Get a specific adapter by name.
   *
   * @param {string} name - Adapter name (e.g. 'sdxl-local-t2i')
   * @returns {Object|null} Adapter object or null
   */
  const getAdapter = useCallback((name) => {
    return adapters.find(a => a.name === name) || null
  }, [adapters])

  /**
   * Find adapters matching criteria.
   *
   * @param {Object} filters - Filter criteria
   * @param {string} [filters.operation] - Operation type ('generate', 'transform', etc.)
   * @param {string} [filters.output_type] - Output media type ('image', 'video', 'audio', 'text')
   * @param {string} [filters.compute] - Compute target ('local', 'cloud')
   * @param {string} [filters.model_family] - Model family ('sdxl', 'wan2.2', 'flux', etc.)
   * @returns {Array} Matching adapters
   */
  const findAdapters = useCallback((filters = {}) => {
    return adapters.filter(a => {
      if (filters.operation && !a.supported_ops?.includes(filters.operation)) return false
      if (filters.output_type && a.output_type !== filters.output_type) return false
      if (filters.compute && a.compute !== filters.compute) return false
      if (filters.model_family && a.model_family !== filters.model_family) return false
      return true
    })
  }, [adapters])

  /**
   * Get constraints for a specific adapter.
   *
   * @param {string} name - Adapter name
   * @returns {Object|null} Constraints object or null
   */
  const getConstraints = useCallback((name) => {
    const adapter = adapters.find(a => a.name === name)
    return adapter?.constraints || null
  }, [adapters])

  /**
   * Get all model families available (for LoRA filtering).
   *
   * @returns {string[]} Unique model families
   */
  const getModelFamilies = useCallback(() => {
    const families = new Set(adapters.map(a => a.model_family).filter(Boolean))
    return [...families].sort()
  }, [adapters])

  /**
   * Force re-fetch from backend.
   */
  const refresh = useCallback(() => {
    try { sessionStorage.removeItem(CACHE_KEY) } catch {}
    fetchedRef.current = false
    return fetchAdapters()
  }, [fetchAdapters])

  return {
    adapters,
    getAdapter,
    findAdapters,
    getConstraints,
    getModelFamilies,
    isLoading,
    error,
    refresh,
    count: adapters.length,
  }
}
