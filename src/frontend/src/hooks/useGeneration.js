import { useState, useCallback, useRef, useEffect } from 'react'
import { DEBUG } from '../config'
import { apiFetch } from '../api'

/**
 * useGeneration — Unified hook for dispatching generation requests via V2 API.
 *
 * Posts a GenerationRequest JSON body to `/v2/generate`, returns the result
 * (prompt_id, adapter_name, credits_used, etc.).
 *
 * Handles:
 * - Authenticated JSON POST to /v2/generate
 * - Credit estimation via /v2/estimate (optional pre-flight)
 * - Cancellation via AbortController
 * - Insufficient credits detection (dispatches window event)
 * - Error normalization for UI consumption
 *
 * Usage:
 *   const { generate, estimate, isLoading, error, lastResult, cancel } = useGeneration()
 *
 *   // Fire-and-forget (job tracking via QueueIndicator / ProgressTracker)
 *   const result = await generate({
 *     operation: 'generate',
 *     target_type: 'image',
 *     prompt: 'a cat in space',
 *     adapter_hint: 'sdxl-local-t2i',
 *     checkpoint: 'CyberRealistic_Pony_v14.1_FP16.safetensors',
 *     steps: 30,
 *     cfg: 7.5,
 *   })
 *   // result = { prompt_id, status, compute_target, credits_used, adapter_name, meta }
 *
 *   // Pre-flight cost check
 *   const est = await estimate({ operation: 'generate', target_type: 'video', ... })
 *   // est = { adapter, credits_required, constraints }
 *
 * Dispatches `insufficient-credits` CustomEvent on 402 responses.
 */
export default function useGeneration() {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [lastResult, setLastResult] = useState(null)
  const abortRef = useRef(null)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      if (abortRef.current) abortRef.current.abort()
    }
  }, [])

  /**
   * Parse error response from V2 API.
   * Handles both string and structured error objects.
   */
  const parseError = useCallback(async (res) => {
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      if (body?.detail) {
        if (typeof body.detail === 'object') {
          // Structured error (e.g. insufficient_credits)
          if (body.detail.error === 'insufficient_credits') {
            // Dispatch event for CreditsContext
            window.dispatchEvent(new CustomEvent('insufficient-credits', {
              detail: {
                required: body.detail.required,
                available: body.detail.available,
                packages: Array.isArray(body.detail.packages) ? body.detail.packages : [],
              }
            }))
            detail = `Not enough credits. (Need ${body.detail.required}, have ${body.detail.available})`
          } else {
            detail = JSON.stringify(body.detail)
          }
        } else {
          detail = String(body.detail)
        }
      }
    } catch {
      // JSON parse failed — use status text
      detail = `HTTP ${res.status} ${res.statusText}`
    }
    return detail
  }, [])

  /**
   * Submit a generation request to V2 API.
   *
   * @param {Object} request - GenerationRequest fields
   * @returns {Promise<Object|null>} GenerationResult or null on failure
   */
  const generate = useCallback(async (request) => {
    // Cancel any existing request
    if (abortRef.current) abortRef.current.abort()

    const controller = new AbortController()
    abortRef.current = controller

    setIsLoading(true)
    setError(null)

    try {
      if (DEBUG) console.log('🚀 V2 generate:', request.operation, request.target_type, request.adapter_hint || 'auto')

      const res = await apiFetch('/v2/generate', {
        method: 'POST',
        body: JSON.stringify(request),
        signal: controller.signal,
      })

      if (!res.ok) {
        const errMsg = await parseError(res)
        throw new Error(errMsg)
      }

      const result = await res.json()

      if (DEBUG) {
        console.log(`✅ V2 result: adapter=${result.adapter_name}, prompt_id=${result.prompt_id}, credits=${result.credits_used}`)
      }

      if (mountedRef.current) {
        setLastResult(result)
      }

      return result
    } catch (err) {
      if (err.name === 'AbortError') {
        if (DEBUG) console.log('🚫 V2 generate cancelled')
        return null
      }
      console.error('❌ V2 generate error:', err)
      if (mountedRef.current) {
        setError(err.message)
      }
      return null
    } finally {
      if (mountedRef.current) {
        setIsLoading(false)
      }
    }
  }, [parseError])

  /**
   * Estimate credit cost without executing.
   *
   * @param {Object} request - GenerationRequest fields
   * @returns {Promise<Object|null>} { adapter, credits_required, constraints } or null
   */
  const estimate = useCallback(async (request) => {
    try {
      if (DEBUG) console.log('💰 V2 estimate:', request.operation, request.target_type)

      const res = await apiFetch('/v2/estimate', {
        method: 'POST',
        body: JSON.stringify(request),
      })

      if (!res.ok) {
        const errMsg = await parseError(res)
        throw new Error(errMsg)
      }

      const result = await res.json()

      if (DEBUG) {
        console.log(`💰 V2 estimate: adapter=${result.adapter}, credits=${result.credits_required}`)
      }

      return result
    } catch (err) {
      console.error('❌ V2 estimate error:', err)
      return null
    }
  }, [parseError])

  /**
   * Cancel any in-progress generation request.
   */
  const cancel = useCallback(() => {
    if (abortRef.current) abortRef.current.abort()
    setIsLoading(false)
    setError(null)
  }, [])

  return {
    generate,
    estimate,
    isLoading,
    error,
    setError,
    lastResult,
    cancel,
  }
}
