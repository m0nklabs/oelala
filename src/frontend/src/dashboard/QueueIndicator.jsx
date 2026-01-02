import React, { useEffect, useState, useCallback, useRef } from 'react'
import { Clock, Play, Loader2, X, CheckCircle, RefreshCw } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../config'

/**
 * QueueIndicator - Compact header indicator for ComfyUI queue
 * Shows running count and pending count, click for popup with details
 */
export default function QueueIndicator({ onJobComplete, refreshToken }) {
  const [queue, setQueue] = useState({ running: [], pending: [], total_running: 0, total_pending: 0 })
  const [completedJobs, setCompletedJobs] = useState([])
  const [showPopup, setShowPopup] = useState(false)
  const [notifiedIds, setNotifiedIds] = useState(new Set())
  const popupRef = useRef(null)

  const fetchQueue = useCallback(async () => {
    try {
      const res = await fetch(`${BACKEND_BASE}/comfyui/queue`)
      if (!res.ok) return
      const data = await res.json()
      setQueue(data)
    } catch (e) {
      if (DEBUG) console.debug('⚠️ Queue fetch failed:', e)
    }
  }, [])

  const checkJobStatus = useCallback(async (promptId) => {
    try {
      const res = await fetch(`${BACKEND_BASE}/comfyui/job/${promptId}`)
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
      if (!notifiedIds.has(job.prompt_id) && job.status === 'completed' && job.output_video) {
        if (onJobComplete) onJobComplete(job)
        setNotifiedIds(prev => new Set([...prev, job.prompt_id]))
      }
    }
  }, [completedJobs, notifiedIds, onJobComplete])

  // Watch for jobs completing
  useEffect(() => {
    const watchJobs = async () => {
      for (const job of queue.running) {
        const status = await checkJobStatus(job.prompt_id)
        if (status && status.status === 'completed') {
          setCompletedJobs(prev => {
            if (prev.some(j => j.prompt_id === status.prompt_id)) return prev
            return [...prev, status].slice(-10)
          })
        }
      }
    }
    if (queue.running.length > 0) watchJobs()
  }, [queue.running, checkJobStatus])

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
      await fetch(`${BACKEND_BASE}/comfyui/queue/${promptId}`, { method: 'DELETE' })
      fetchQueue()
    } catch (e) {
      console.error('Failed to cancel job:', e)
    }
  }

  const isRunning = queue.total_running > 0
  const totalJobs = queue.total_running + queue.total_pending

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
          backgroundColor: isRunning ? 'rgba(34, 197, 94, 0.15)' : 'transparent',
          border: `1px solid ${isRunning ? '#22c55e' : 'var(--border-color)'}`,
          borderRadius: '6px',
          cursor: 'pointer',
          color: 'var(--text-primary)',
          fontSize: '0.8rem',
        }}
        title={isRunning ? `${queue.total_running} running, ${queue.total_pending} queued` : 'No active jobs'}
      >
        {isRunning ? (
          <Loader2 size={14} color="#22c55e" className="spin" />
        ) : (
          <Clock size={14} color="var(--text-muted)" />
        )}
        <span style={{ fontWeight: 500 }}>
          {isRunning ? queue.total_running : 0}
        </span>
        {queue.total_pending > 0 && (
          <span style={{ color: 'var(--text-muted)' }}>+{queue.total_pending}</span>
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
          backgroundColor: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          borderRadius: '8px',
          boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
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
                  <JobRow key={job.prompt_id} job={job} status="running" onCancel={cancelJob} />
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

function JobRow({ job, status, onCancel }) {
  const colors = { running: '#22c55e', pending: '#fbbf24', completed: '#3b82f6' }
  const Icon = { running: Loader2, pending: Clock, completed: CheckCircle }[status]

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      padding: '6px 8px',
      backgroundColor: 'var(--bg-input)',
      borderRadius: '4px',
      marginBottom: '4px',
      fontSize: '0.8rem',
    }}>
      <Icon size={12} color={colors[status]} className={status === 'running' ? 'spin' : ''} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ 
          whiteSpace: 'nowrap', 
          overflow: 'hidden', 
          textOverflow: 'ellipsis',
          fontWeight: 500,
        }}>
          {job.prompt || job.prompt_id.slice(0, 8)}
        </div>
        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
          {job.resolution} {job.aspect_ratio} {job.num_frames && `• ${job.num_frames}f`}
        </div>
      </div>
      {status !== 'completed' && onCancel && (
        <button
          onClick={() => onCancel(job.prompt_id)}
          style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: '2px' }}
        >
          <X size={12} color="var(--text-muted)" />
        </button>
      )}
      {status === 'completed' && job.output_video && (
        <a
          href={`${BACKEND_BASE}${job.output_video}`}
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: '#3b82f6', fontSize: '0.7rem' }}
        >
          View
        </a>
      )}
    </div>
  )
}
