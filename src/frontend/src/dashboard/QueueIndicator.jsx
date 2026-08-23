import React, { useEffect, useState, useCallback, useRef } from 'react'
import { Clock, Play, Loader2, X, CheckCircle, RefreshCw, Brain, AlertTriangle } from 'lucide-react'
import { BACKEND_BASE, DEBUG, getMediaUrl } from '../config'
import { apiFetch } from '../api'
import ProgressTracker from './ProgressTracker'

/**
 * QueueIndicator - Compact header indicator for ComfyUI queue
 * Shows running count and pending count, click for popup with details
 */
export default function QueueIndicator({ onJobComplete, refreshToken }) {
  const [queue, setQueue] = useState({ running: [], pending: [], failed: [], training: [], total_running: 0, total_pending: 0, total_failed: 0, total_training: 0 })
  const [completedJobs, setCompletedJobs] = useState([])
  const [showPopup, setShowPopup] = useState(false)
  const [notifiedIds, setNotifiedIds] = useState(new Set())
  const popupRef = useRef(null)
  const prevRunningRef = useRef([])

  const EMPTY_QUEUE = { running: [], pending: [], failed: [], training: [], total_running: 0, total_pending: 0, total_failed: 0, total_training: 0 }

  const fetchQueue = useCallback(async () => {
    try {
      const res = await apiFetch('/comfyui/queue')
      if (!res.ok) {
        setQueue(EMPTY_QUEUE)
        setCompletedJobs([])
        prevRunningRef.current = []
        return
      }
      const data = await res.json()
      setQueue(data)
    } catch (e) {
      setQueue(EMPTY_QUEUE)
      if (DEBUG) console.debug('⚠️ Queue fetch failed:', e)
    }
  }, [])

  const checkJobStatus = useCallback(async (promptId) => {
    try {
      const res = await apiFetch(`/comfyui/job/${promptId}`)
      if (!res.ok) return null
      return await res.json()
    } catch (e) {
      return null
    }
  }, [])

  // Poll queue status
  useEffect(() => {
    fetchQueue()
    const interval = setInterval(fetchQueue, 3000)
    return () => clearInterval(interval)
  }, [fetchQueue, refreshToken])

  // Check for completed jobs
  useEffect(() => {
    for (const job of completedJobs) {
      if (!notifiedIds.has(job.prompt_id) && job.status === 'completed' && (job.output_video || job.output_image || job.output_audio)) {
        if (onJobComplete) onJobComplete(job)
        setNotifiedIds(prev => new Set([...prev, job.prompt_id]))
      }
    }
  }, [completedJobs, notifiedIds, onJobComplete])

  // Watch for jobs completing (running + pending cloud jobs + recently disappeared)
  useEffect(() => {
    let cancelled = false

    // Detect jobs that were running but disappeared from the queue (completed between polls)
    const prevIds = new Set(prevRunningRef.current.map(j => j.prompt_id))
    const currIds = new Set(queue.running.map(j => j.prompt_id))
    const disappeared = prevRunningRef.current.filter(j => !currIds.has(j.prompt_id))
    prevRunningRef.current = [...queue.running]

    const watchJobs = async () => {
      // Poll running jobs for completion
      for (const job of queue.running) {
        if (cancelled) return
        const status = await checkJobStatus(job.prompt_id)
        if (cancelled) return
        if (status && (status.status === 'completed' || status.status === 'failed')) {
          setCompletedJobs(prev => {
            if (prev.some(j => j.prompt_id === status.prompt_id)) return prev
            return [...prev, status].slice(-10)
          })
        }
      }
      // Poll jobs that disappeared from queue (race condition fix)
      for (const job of disappeared) {
        if (cancelled) return
        if (notifiedIds.has(job.prompt_id)) continue
        const status = await checkJobStatus(job.prompt_id)
        if (cancelled) return
        if (status && (status.status === 'completed' || status.status === 'failed')) {
          setCompletedJobs(prev => {
            if (prev.some(j => j.prompt_id === status.prompt_id)) return prev
            return [...prev, status].slice(-10)
          })
        }
      }
      // Poll pending cloud jobs so their status advances (IN_QUEUE → IN_PROGRESS → COMPLETED)
      for (const job of queue.pending) {
        if (cancelled) return
        if (job.compute_target === 'cloud') {
          const status = await checkJobStatus(job.prompt_id)
          if (cancelled) return
          if (status && (status.status === 'completed' || status.status === 'failed')) {
            setCompletedJobs(prev => {
              if (prev.some(j => j.prompt_id === status.prompt_id)) return prev
              return [...prev, status].slice(-10)
            })
          }
        }
      }
    }
    if (queue.running.length > 0 || disappeared.length > 0 || queue.pending.some(j => j.compute_target === 'cloud')) watchJobs()
    return () => { cancelled = true }
  }, [queue.running, queue.pending, checkJobStatus, notifiedIds])

  // Close popup on click outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (popupRef.current && !popupRef.current.contains(e.target)) {
        setShowPopup(false)
      }
    }
    if (showPopup) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [showPopup])

  const cancelJob = async (promptId) => {
    try {
      // Check if this is a cloud job — cancel via RunPod API
      const job = [...queue.running, ...queue.pending].find(j => j.prompt_id === promptId)
      if (job?.compute_target === 'cloud' && job?.runpod_job_id) {
        await fetch(`${BACKEND_BASE}/runpod/cancel/${job.runpod_job_id}`, { method: 'POST' })
      } else {
        await fetch(`${BACKEND_BASE}/comfyui/queue/${promptId}`, { method: 'DELETE' })
      }
      fetchQueue()
    } catch (e) {
      console.error('Failed to cancel job:', e)
    }
  }

  const isRunning = queue.total_running > 0
  const hasFailed = (queue.total_failed || 0) > 0
  const isTraining = (queue.total_training || 0) > 0
  const totalJobs = queue.total_running + queue.total_pending + (queue.total_failed || 0) + (queue.total_training || 0)

  return (
    <div style={{ position: 'relative' }} ref={popupRef}>
      {/* Compact indicator button */}
      <button
        onClick={() => setShowPopup(!showPopup)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          padding: '6px 10px',
          backgroundColor: isRunning ? 'rgba(34, 197, 94, 0.15)' : hasFailed ? 'rgba(239, 68, 68, 0.15)' : isTraining ? 'rgba(168, 85, 247, 0.15)' : 'transparent',
          border: `1px solid ${isRunning ? '#22c55e' : hasFailed ? '#ef4444' : isTraining ? '#a855f7' : 'var(--border-color)'}`,
          borderRadius: '6px',
          cursor: 'pointer',
          color: 'var(--text-primary)',
          fontSize: '0.8rem',
        }}
        title={
          isRunning
            ? `${queue.total_running} running, ${queue.total_pending} queued${isTraining ? `, ${queue.total_training} training` : ''}`
            : isTraining
              ? `${queue.total_training} training`
              : 'No active jobs'
        }
      >
        <span style={{ fontSize: '14px' }}>
          {isRunning ? '⏳' : isTraining ? '🧠' : '🕐'}
        </span>
        <span style={{ fontWeight: 500 }}>
          {isRunning ? queue.total_running : 0}
        </span>
        {queue.total_pending > 0 && (
          <span style={{ color: 'var(--text-muted)' }}>+{queue.total_pending}</span>
        )}
        {isTraining && (
          <span style={{ color: '#a855f7', fontSize: '0.7rem' }}>🧠{queue.total_training}</span>
        )}
        {hasFailed && (
          <span style={{ color: '#ef4444', fontSize: '0.7rem' }}>⚠️{queue.total_failed}</span>
        )}
      </button>

      {/* Popup with details */}
      {showPopup && (
        <div style={{
          position: 'absolute',
          top: '100%',
          right: 0,
          marginTop: '8px',
          width: '320px',
          backgroundColor: 'var(--bg-panel)',
          border: '1px solid var(--border-color)',
          borderRadius: '8px',
          boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
          zIndex: 1000,
          overflow: 'hidden',
        }}>
          {/* Popup header */}
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '10px 12px',
            borderBottom: '1px solid var(--border-color)',
            backgroundColor: 'var(--bg-primary)',
          }}>
            <span style={{ fontWeight: 600, fontSize: '0.85rem' }}>Generation Queue</span>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                onClick={fetchQueue}
                style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: '2px' }}
              >
                <RefreshCw size={12} color="var(--text-muted)" />
              </button>
              <button
                onClick={() => setShowPopup(false)}
                style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: '2px' }}
              >
                <X size={14} color="var(--text-muted)" />
              </button>
            </div>
          </div>

          {/* Popup content */}
          <div style={{ maxHeight: '300px', overflowY: 'auto', padding: '8px' }}>
            {/* Running */}
            {queue.running.length > 0 && (
              <div style={{ marginBottom: '8px' }}>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '4px', textTransform: 'uppercase' }}>
                  Running
                </div>
                {queue.running.map((job) => (
                  <JobRow
                    key={job.prompt_id}
                    job={job}
                    status="running"
                    onCancel={cancelJob}
                    onJobComplete={fetchQueue}
                  />
                ))}
              </div>
            )}

            {/* Pending */}
            {queue.pending.length > 0 && (
              <div style={{ marginBottom: '8px' }}>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '4px', textTransform: 'uppercase' }}>
                  Pending
                </div>
                {queue.pending.map((job) => (
                  <JobRow key={job.prompt_id} job={job} status="pending" onCancel={cancelJob} />
                ))}
              </div>
            )}

            {/* Training */}
            {(queue.training || []).length > 0 && (
              <div style={{ marginBottom: '8px' }}>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '4px', textTransform: 'uppercase' }}>
                  LoRA Training
                </div>
                {queue.training.map((job) => (
                  <TrainingJobRow key={job.job_id} job={job} />
                ))}
              </div>
            )}

            {/* Failed (dismissable) */}
            {(queue.failed || []).length > 0 && (
              <div style={{ marginBottom: '8px' }}>
                <div style={{ fontSize: '0.7rem', color: '#ef4444', marginBottom: '4px', textTransform: 'uppercase' }}>
                  Failed
                </div>
                {queue.failed.map((job) => (
                  <JobRow key={job.prompt_id} job={job} status="failed" onCancel={cancelJob} />
                ))}
              </div>
            )}

            {/* Recent completed */}
            {completedJobs.length > 0 && (
              <div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '4px', textTransform: 'uppercase' }}>
                  Completed
                </div>
                {completedJobs.slice(-3).reverse().map((job) => (
                  <JobRow key={job.prompt_id} job={job} status="completed" />
                ))}
              </div>
            )}

            {/* Empty state */}
            {totalJobs === 0 && completedJobs.length === 0 && (
              <div style={{ textAlign: 'center', padding: '16px', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                No active jobs
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function JobRow({ job, status, onCancel, onJobComplete }) {
  const [showDetails, setShowDetails] = useState(status === 'running')
  const isCloud = job.compute_target === 'cloud'
  const isWindows = job.server === 'windows'
  const colors = { running: '#22c55e', pending: '#fbbf24', completed: '#3b82f6', failed: '#ef4444' }
  const Icon = { running: Loader2, pending: Clock, completed: CheckCircle, failed: AlertTriangle }[status]

  return (
    <div style={{ marginBottom: '4px' }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '6px 8px',
          backgroundColor: isCloud ? 'rgba(99, 102, 241, 0.08)' : isWindows ? 'rgba(14, 165, 233, 0.08)' : 'var(--bg-input)',
          borderRadius: '4px',
          fontSize: '0.8rem',
          cursor: status === 'running' ? 'pointer' : 'default',
          borderLeft: isCloud ? '2px solid #6366f1' : isWindows ? '2px solid #0ea5e9' : 'none',
        }}
        onClick={() => status === 'running' && setShowDetails(!showDetails)}
      >
        <Icon size={12} color={isCloud ? '#6366f1' : isWindows ? '#0ea5e9' : colors[status]} className={status === 'running' ? 'spin' : ''} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              fontWeight: 500,
            }}
          >
            {isCloud && <span title="Cloud Max (RunPod)" style={{ fontSize: '11px' }}>☁️</span>}
            {isWindows && <span title="MiniMax-H3 (Windows PC)" style={{ fontSize: '11px' }}>🪟</span>}
            {job.prompt || job.prompt_id.slice(0, 8)}
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            {isCloud && <span style={{ color: '#6366f1', marginRight: '4px' }}>Cloud</span>}
            {isWindows && <span style={{ color: '#0ea5e9', marginRight: '4px' }}>Windows PC</span>}
            {status === 'failed' && job.error && <span style={{ color: '#ef4444' }}>{job.error.slice(0, 60)}</span>}
            {status !== 'failed' && <>{job.resolution} {job.aspect_ratio} {job.num_frames && `• ${job.num_frames}f`}</>}
          </div>
        </div>
        {status !== 'completed' && onCancel && (
          <button
            onClick={(e) => {
              e.stopPropagation()
              onCancel(job.prompt_id)
            }}
            style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: '2px' }}
          >
            <X size={12} color="var(--text-muted)" />
          </button>
        )}
        {status === 'completed' && (job.output_video || job.output_image || job.output_audio) && (
          <a
            href={getMediaUrl(job.output_video || job.output_image || job.output_audio, job.signed_url)}
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: '#3b82f6', fontSize: '0.7rem' }}
            onClick={(e) => e.stopPropagation()}
          >
            View
          </a>
        )}
      </div>

      {/* Detailed progress tracker for running jobs */}
      {status === 'running' && showDetails && (
        <div style={{ marginTop: '4px', paddingLeft: '8px' }}>
          <ProgressTracker
            promptId={job.prompt_id}
            onComplete={onJobComplete}
          />
        </div>
      )}
    </div>
  )
}

function TrainingJobRow({ job }) {
  const progress = job.progress || 0
  const isRunning = job.status === 'running'

  return (
    <div style={{ marginBottom: '4px' }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '6px 8px',
          backgroundColor: 'var(--bg-input)',
          borderRadius: '4px',
          fontSize: '0.8rem',
        }}
      >
        <Brain size={12} color="#a855f7" className={isRunning ? 'spin' : ''} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            style={{
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              fontWeight: 500,
            }}
          >
            {job.name || job.trigger || 'Training'}
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            {job.trigger} • {job.images_count} photos • {job.steps_done}/{job.steps_total} steps
          </div>
          {isRunning && (
            <div style={{ marginTop: '4px' }}>
              <div
                style={{
                  width: '100%',
                  height: '4px',
                  backgroundColor: 'rgba(168, 85, 247, 0.2)',
                  borderRadius: '2px',
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    width: `${progress}%`,
                    height: '100%',
                    backgroundColor: '#a855f7',
                    borderRadius: '2px',
                    transition: 'width 0.5s ease-out',
                  }}
                />
              </div>
              <div style={{ fontSize: '0.65rem', color: '#a855f7', marginTop: '2px' }}>
                {progress}%
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
