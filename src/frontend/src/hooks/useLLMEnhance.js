import { useState, useCallback, useRef, useEffect } from 'react'
import { BACKEND_BASE, DEBUG } from '../config'

/**
 * useLLMEnhance — Hook for async LLM prompt enhancement via queue.
 *
 * Submits a prompt to /generate-prompt (returns immediately with job_id),
 * then polls /llm-job/{job_id} every 1s until completed or failed.
 *
 * Usage:
 *   const { enhance, isLoading, queuePosition, error, cancel } = useLLMEnhance()
 *
 *   const result = await enhance({
 *     input: 'a cat wearing sunglasses',
 *     style: 'cinematic',
 *     mode: 'expand',
 *     include_motion: true,
 *     model: 'GLM-4.7-Flash',
 *   })
 *   // result = { prompt, negative_prompt, motion_prompt, llm_used, ... }
 *
 * Returns null if enhancement fails or is cancelled.
 */
export default function useLLMEnhance() {
  const [isLoading, setIsLoading] = useState(false)
  const [queuePosition, setQueuePosition] = useState(null) // null | number (-1 = processing)
  const [error, setError] = useState(null)
  const abortRef = useRef(null) // AbortController for cancellation
  const pollTimerRef = useRef(null)
  const mountedRef = useRef(true)

  // Cleanup on unmount
  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      if (pollTimerRef.current) clearTimeout(pollTimerRef.current)
      if (abortRef.current) abortRef.current.abort()
    }
  }, [])

  /**
   * Poll for job result. Returns a promise that resolves with the result
   * or null on failure/cancellation.
   */
  const pollForResult = useCallback((jobId, signal) => {
    return new Promise((resolve) => {
      const poll = async () => {
        if (signal.aborted) {
          resolve(null)
          return
        }

        try {
          const res = await fetch(`${BACKEND_BASE}/llm-job/${jobId}`, { signal })
          if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Unknown error' }))
            throw new Error(err.detail || `HTTP ${res.status}`)
          }

          const data = await res.json()
          if (DEBUG) console.log(`🔄 LLM job ${jobId}: ${data.status} (pos: ${data.queue_position})`)

          if (!mountedRef.current) {
            resolve(null)
            return
          }

          if (data.status === 'completed') {
            setQueuePosition(null)
            resolve(data.result)
            return
          }

          if (data.status === 'failed') {
            setQueuePosition(null)
            throw new Error(data.error || 'Enhancement failed')
          }

          // Update queue position
          setQueuePosition(data.queue_position)

          // Poll again in 1s
          pollTimerRef.current = setTimeout(poll, 1000)
        } catch (err) {
          if (err.name === 'AbortError') {
            resolve(null)
            return
          }
          if (mountedRef.current) {
            setError(err.message)
          }
          resolve(null)
        }
      }

      poll()
    })
  }, [])

  /**
   * Submit a prompt enhancement request and wait for the result.
   *
   * @param {Object} params - Request parameters
   * @param {string} params.input - The prompt text to enhance
   * @param {string|null} params.style - Style preset (optional)
   * @param {string} params.mode - 'expand' | 'refine' | 'variations'
   * @param {boolean} params.include_negative - Include negative prompt
   * @param {boolean} params.include_motion - Include motion prompt
   * @param {boolean} params.use_llm - Use LLM (default true)
   * @param {string|null} params.model - Model override (optional)
   * @param {string|null} params.refine_instruction - Refine instruction (optional)
   * @returns {Promise<Object|null>} Enhanced prompt result or null on failure
   */
  const enhance = useCallback(async (params) => {
    // Cancel any existing request
    if (abortRef.current) abortRef.current.abort()
    if (pollTimerRef.current) clearTimeout(pollTimerRef.current)

    const controller = new AbortController()
    abortRef.current = controller

    setIsLoading(true)
    setError(null)
    setQueuePosition(null)

    try {
      // Submit to queue
      const res = await fetch(`${BACKEND_BASE}/generate-prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: params.input,
          style: params.style ?? null,
          mode: params.mode ?? 'expand',
          include_negative: params.include_negative ?? true,
          include_motion: params.include_motion ?? false,
          use_llm: params.use_llm ?? true,
          model: params.model ?? null,
          refine_instruction: params.refine_instruction ?? null,
          nsfw_intensity: params.nsfw_intensity ?? null,
        }),
        signal: controller.signal,
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Submit failed' }))
        throw new Error(err.detail || `HTTP ${res.status}`)
      }

      const data = await res.json()

      // Sync fallback — result is already in the response
      if (data.status === 'completed') {
        if (DEBUG) console.log('✨ LLM enhance (sync):', data)
        return data
      }

      // Async queue — poll for result
      if (data.status === 'queued' && data.job_id) {
        setQueuePosition(data.queue_position ?? 0)
        if (DEBUG) console.log(`📝 LLM job queued: ${data.job_id} (pos: ${data.queue_position})`)

        const result = await pollForResult(data.job_id, controller.signal)
        if (DEBUG && result) console.log('✨ LLM enhance result:', result)
        return result
      }

      // Unexpected response
      throw new Error('Unexpected response from /generate-prompt')
    } catch (err) {
      if (err.name === 'AbortError') {
        if (DEBUG) console.log('🚫 LLM enhance cancelled')
        return null
      }
      console.error('LLM enhance error:', err)
      if (mountedRef.current) {
        setError(err.message)
      }
      return null
    } finally {
      if (mountedRef.current) {
        setIsLoading(false)
      }
    }
  }, [pollForResult])

  /**
   * Cancel any in-progress enhancement request.
   */
  const cancel = useCallback(() => {
    if (abortRef.current) abortRef.current.abort()
    if (pollTimerRef.current) clearTimeout(pollTimerRef.current)
    setIsLoading(false)
    setQueuePosition(null)
    setError(null)
  }, [])

  return {
    enhance,
    isLoading,
    queuePosition,
    error,
    setError,
    cancel,
  }
}
