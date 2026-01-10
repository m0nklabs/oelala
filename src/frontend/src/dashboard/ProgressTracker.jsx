import React, { useEffect, useState, useCallback, useRef } from 'react'
import { Clock, Loader2, TrendingUp } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../config'

/**
 * ProgressTracker - Real-time progress indicator for ComfyUI jobs
 * Shows queue position, generation progress, ETA, and preview thumbnails
 */
export default function ProgressTracker({ promptId, onComplete }) {
  const [jobStatus, setJobStatus] = useState(null)
  const [queuePosition, setQueuePosition] = useState(null)
  const [progress, setProgress] = useState(0)
  const [eta, setEta] = useState(null)
  const [startTime, setStartTime] = useState(Date.now())
  const [currentNode, setCurrentNode] = useState('')
  const onCompleteRef = useRef(onComplete)

  // Keep onComplete ref up to date
  useEffect(() => {
    onCompleteRef.current = onComplete
  }, [onComplete])

  // Reset start time whenever we start tracking a new prompt
  useEffect(() => {
    if (!promptId) return
    setStartTime(Date.now())
    setProgress(0)
  }, [promptId])

  // Fetch job status
  const fetchJobStatus = useCallback(async () => {
    if (!promptId) return

    try {
      const res = await fetch(`${BACKEND_BASE}/comfyui/job/${promptId}`)
      if (!res.ok) return
      const data = await res.json()
      setJobStatus(data)

      // Update current node if available
      if (data.current_node) {
        setCurrentNode(data.current_node)
      }

      // Check if job completed
      if (data.status === 'completed' || data.status === 'failed') {
        if (onCompleteRef.current) onCompleteRef.current(data)
      }
    } catch (e) {
      if (DEBUG) console.debug('⚠️ Job status fetch failed:', e)
    }
  }, [promptId])

  // Fetch queue position
  const fetchQueue = useCallback(async () => {
    if (!promptId) return

    try {
      const res = await fetch(`${BACKEND_BASE}/comfyui/queue`)
      if (!res.ok) return
      const data = await res.json()

      // Find position in queue
      const runningIndex = data.running.findIndex(j => j.prompt_id === promptId)
      const pendingIndex = data.pending.findIndex(j => j.prompt_id === promptId)

      if (runningIndex >= 0) {
        setQueuePosition({ status: 'running', position: runningIndex })
      } else if (pendingIndex >= 0) {
        setQueuePosition({
          status: 'pending',
          position: data.running.length + pendingIndex,
        })
      } else {
        setQueuePosition(null)
      }
    } catch (e) {
      if (DEBUG) console.debug('⚠️ Queue fetch failed:', e)
    }
  }, [promptId])

  // Calculate ETA based on elapsed time and progress
  useEffect(() => {
    if (progress > 0 && progress < 100) {
      const elapsed = Date.now() - startTime
      const estimatedTotal = (elapsed / progress) * 100
      const remaining = estimatedTotal - elapsed
      setEta(Math.max(0, Math.round(remaining / 1000)))
    } else {
      setEta(null)
    }
  }, [progress, startTime])

  // Poll for updates - stops when job completes
  useEffect(() => {
    if (!promptId) return

    // Check if job is already in terminal state
    if (jobStatus && (jobStatus.status === 'completed' || jobStatus.status === 'failed')) {
      return
    }

    // Initial fetch
    fetchJobStatus()
    fetchQueue()

    // Set up polling
    const statusInterval = setInterval(fetchJobStatus, 2000)
    const queueInterval = setInterval(fetchQueue, 3000)

    return () => {
      clearInterval(statusInterval)
      clearInterval(queueInterval)
    }
  }, [promptId, fetchJobStatus, fetchQueue, jobStatus])

  // Update progress based on job status
  useEffect(() => {
    if (!jobStatus) return

    if (jobStatus.status === 'completed') {
      setProgress(100)
    } else if (jobStatus.status === 'running') {
      // Prefer real progress from backend when available
      const { progress: rawProgress, current_step, total_steps } = jobStatus

      if (typeof rawProgress === 'number' && !Number.isNaN(rawProgress)) {
        // Support both 0–1 (fraction) and 0–100 (percentage) ranges
        const normalized =
          rawProgress <= 1 ? rawProgress * 100 : rawProgress
        setProgress(Math.max(0, Math.min(100, Math.round(normalized))))
      } else if (
        typeof current_step === 'number' &&
        typeof total_steps === 'number' &&
        total_steps > 0
      ) {
        const ratio = current_step / total_steps
        const normalized = Math.max(0, Math.min(1, ratio)) * 100
        setProgress(Math.round(normalized))
      } else {
        // No reliable progress data; avoid misleading time-based increments.
        // Only bump once from 0 to a small "started" value.
        setProgress(prev => (prev === 0 ? 5 : prev))
      }
    } else if (jobStatus.status === 'queued') {
      setProgress(0)
    }
  }, [jobStatus])

  if (!promptId || !jobStatus) {
    return null
  }

  const formatEta = (seconds) => {
    if (!seconds || seconds <= 0) return 'calculating...'
    if (seconds < 60) return `${seconds}s`
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}m ${secs}s`
  }

  const isRunning = jobStatus.status === 'running'
  const isQueued = jobStatus.status === 'queued'
  const isCompleted = jobStatus.status === 'completed'

  return (
    <div
      style={{
        backgroundColor: 'var(--bg-secondary)',
        border: '1px solid var(--border-color)',
        borderRadius: '8px',
        padding: '16px',
        marginBottom: '16px',
      }}
    >
      {/* Header with status */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '12px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {isRunning && <Loader2 size={16} className="spin" color="#22c55e" />}
          {isQueued && <Clock size={16} color="#fbbf24" />}
          {isCompleted && <TrendingUp size={16} color="#3b82f6" />}
          <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>
            {isRunning && 'Generating...'}
            {isQueued && 'In Queue'}
            {isCompleted && 'Completed'}
          </span>
        </div>

        {/* Queue position indicator */}
        {queuePosition && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '0.75rem',
              color: 'var(--text-muted)',
              backgroundColor: 'var(--bg-input)',
              padding: '4px 8px',
              borderRadius: '4px',
            }}
          >
            <Clock size={12} />
            <span>
              {queuePosition.status === 'running'
                ? 'Running'
                : `Position: ${queuePosition.position + 1}`}
            </span>
          </div>
        )}
      </div>

      {/* Progress bar */}
      {!isCompleted && (
        <div
          style={{
            position: 'relative',
            width: '100%',
            height: '8px',
            backgroundColor: 'var(--bg-input)',
            borderRadius: '4px',
            overflow: 'hidden',
            marginBottom: '8px',
          }}
        >
          <div
            style={{
              position: 'absolute',
              left: 0,
              top: 0,
              height: '100%',
              width: `${progress}%`,
              backgroundColor: isRunning ? '#22c55e' : '#fbbf24',
              borderRadius: '4px',
              transition: 'width 0.3s ease-out',
              boxShadow: `0 0 8px ${isRunning ? 'rgba(34, 197, 94, 0.5)' : 'rgba(251, 191, 36, 0.5)'}`,
            }}
          />
        </div>
      )}

      {/* Progress details */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: '0.75rem',
          color: 'var(--text-muted)',
        }}
      >
        <span>
          {isRunning && `${progress}%`}
          {isQueued && 'Waiting to start...'}
          {isCompleted && 'Generation complete'}
        </span>

        {/* ETA */}
        {isRunning && eta !== null && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
            }}
          >
            <Clock size={12} />
            <span>ETA: {formatEta(eta)}</span>
          </div>
        )}
      </div>

      {/* Current processing node */}
      {currentNode && isRunning && (
        <div
          style={{
            marginTop: '8px',
            padding: '6px 8px',
            backgroundColor: 'var(--bg-input)',
            borderRadius: '4px',
            fontSize: '0.7rem',
            color: 'var(--text-secondary)',
          }}
        >
          {currentNode}
        </div>
      )}

      {/* Preview thumbnail (if available) */}
      {jobStatus.preview_url && (
        <div
          style={{
            marginTop: '12px',
            borderRadius: '4px',
            overflow: 'hidden',
            backgroundColor: 'var(--bg-input)',
          }}
        >
          <img
            src={jobStatus.preview_url}
            alt="Generation preview"
            style={{
              width: '100%',
              height: 'auto',
              display: 'block',
            }}
          />
        </div>
      )}
    </div>
  )
}
