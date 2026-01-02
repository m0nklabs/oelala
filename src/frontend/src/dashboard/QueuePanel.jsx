import React, { useEffect, useState, useCallback } from 'react'
import { Clock, Play, Loader2, X, CheckCircle, AlertCircle, RefreshCw } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../config'

/**
 * QueuePanel - Shows ComfyUI queue status with running and pending jobs
 * Polls for updates and shows job completion notifications
 */
export default function QueuePanel({ onJobComplete, refreshToken }) {
  const [queue, setQueue] = useState({ running: [], pending: [], total_running: 0, total_pending: 0 })
  const [completedJobs, setCompletedJobs] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [expanded, setExpanded] = useState(true)

  // Track which prompt_ids we've already notified about
  const [notifiedIds, setNotifiedIds] = useState(new Set())

  const fetchQueue = useCallback(async () => {
    try {
      const res = await fetch(`${BACKEND_BASE}/comfyui/queue`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setQueue(data)
      setError(null)
      
      if (DEBUG) console.debug('🐛 Queue:', data.total_running, 'running,', data.total_pending, 'pending')
    } catch (e) {
      setError('ComfyUI offline')
      if (DEBUG) console.debug('⚠️ Queue fetch failed:', e)
    }
  }, [])

  // Check for job completions
  const checkJobStatus = useCallback(async (promptId) => {
    try {
      const res = await fetch(`${BACKEND_BASE}/comfyui/job/${promptId}`)
      if (!res.ok) return null
      const data = await res.json()
      return data
    } catch (e) {
      return null
    }
  }, [])

  // Poll queue status
  useEffect(() => {
    fetchQueue()
    const interval = setInterval(fetchQueue, 3000) // Poll every 3 seconds
    return () => clearInterval(interval)
  }, [fetchQueue, refreshToken])

  // Check for newly completed jobs
  useEffect(() => {
    const checkCompletions = async () => {
      // Get all tracked job IDs from queue
      const allJobIds = [...queue.running, ...queue.pending].map(j => j.prompt_id)
      
      // Check jobs that were running but no longer in queue
      for (const job of completedJobs) {
        if (!notifiedIds.has(job.prompt_id) && job.status === 'completed' && job.output_video) {
          // Notify parent component
          if (onJobComplete) {
            onJobComplete(job)
          }
          setNotifiedIds(prev => new Set([...prev, job.prompt_id]))
        }
      }
    }
    
    checkCompletions()
  }, [queue, completedJobs, notifiedIds, onJobComplete])

  // Watch for jobs leaving the running queue (they might be complete)
  useEffect(() => {
    const watchJobs = async () => {
      for (const job of queue.running) {
        // When a job disappears from running, check if it completed
        const status = await checkJobStatus(job.prompt_id)
        if (status && status.status === 'completed') {
          setCompletedJobs(prev => {
            // Don't add duplicates
            if (prev.some(j => j.prompt_id === status.prompt_id)) return prev
            return [...prev, status].slice(-10) // Keep last 10
          })
        }
      }
    }
    
    // Only run when queue changes
    if (queue.running.length > 0) {
      watchJobs()
    }
  }, [queue.running, checkJobStatus])

  const cancelJob = async (promptId) => {
    try {
      const res = await fetch(`${BACKEND_BASE}/comfyui/queue/${promptId}`, { method: 'DELETE' })
      if (res.ok) {
        fetchQueue()
      }
    } catch (e) {
      console.error('Failed to cancel job:', e)
    }
  }

  const totalJobs = queue.total_running + queue.total_pending

  return (
    <div className="queue-panel" style={{
      backgroundColor: 'var(--bg-secondary)',
      borderRadius: '12px',
      border: '1px solid var(--border-color)',
      marginBottom: '16px',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div 
        onClick={() => setExpanded(!expanded)}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '12px 16px',
          cursor: 'pointer',
          borderBottom: expanded ? '1px solid var(--border-color)' : 'none',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Clock size={16} color="#fbbf24" />
          <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>Generation Queue</span>
          {totalJobs > 0 && (
            <span style={{
              backgroundColor: '#fbbf24',
              color: '#000',
              borderRadius: '10px',
              padding: '2px 8px',
              fontSize: '0.75rem',
              fontWeight: 600,
            }}>
              {totalJobs}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {error && (
            <span style={{ color: '#ef4444', fontSize: '0.75rem' }}>{error}</span>
          )}
          <button 
            onClick={(e) => { e.stopPropagation(); fetchQueue(); }}
            style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: '4px' }}
          >
            <RefreshCw size={14} color="var(--text-muted)" />
          </button>
          <span style={{ opacity: 0.5, fontSize: '0.8rem' }}>{expanded ? '▼' : '▶'}</span>
        </div>
      </div>

      {/* Content */}
      {expanded && (
        <div style={{ padding: '12px 16px' }}>
          {/* Running Jobs */}
          {queue.running.length > 0 && (
            <div style={{ marginBottom: '12px' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '8px', textTransform: 'uppercase' }}>
                Running
              </div>
              {queue.running.map((job) => (
                <JobCard key={job.prompt_id} job={job} status="running" onCancel={cancelJob} />
              ))}
            </div>
          )}

          {/* Pending Jobs */}
          {queue.pending.length > 0 && (
            <div style={{ marginBottom: '12px' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '8px', textTransform: 'uppercase' }}>
                Pending ({queue.pending.length})
              </div>
              {queue.pending.map((job) => (
                <JobCard key={job.prompt_id} job={job} status="pending" onCancel={cancelJob} />
              ))}
            </div>
          )}

          {/* Recently Completed */}
          {completedJobs.length > 0 && (
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '8px', textTransform: 'uppercase' }}>
                Recently Completed
              </div>
              {completedJobs.slice(-3).reverse().map((job) => (
                <JobCard key={job.prompt_id} job={job} status="completed" />
              ))}
            </div>
          )}

          {/* Empty State */}
          {totalJobs === 0 && completedJobs.length === 0 && (
            <div style={{ 
              textAlign: 'center', 
              padding: '20px',
              color: 'var(--text-muted)',
              fontSize: '0.85rem'
            }}>
              No jobs in queue
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function JobCard({ job, status, onCancel }) {
  const statusColors = {
    running: '#22c55e',
    pending: '#fbbf24',
    completed: '#3b82f6',
  }

  const StatusIcon = {
    running: Loader2,
    pending: Clock,
    completed: CheckCircle,
  }[status]

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '10px 12px',
      backgroundColor: 'var(--bg-input)',
      borderRadius: '8px',
      marginBottom: '6px',
      border: '1px solid var(--border-color)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flex: 1, minWidth: 0 }}>
        <StatusIcon 
          size={16} 
          color={statusColors[status]} 
          className={status === 'running' ? 'spin' : ''}
        />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ 
            fontSize: '0.85rem', 
            fontWeight: 500,
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}>
            {job.prompt || job.prompt_id.slice(0, 8)}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {job.resolution && `${job.resolution} ${job.aspect_ratio}`}
            {job.num_frames && ` • ${job.num_frames}f`}
            {job.queue_position > 0 && ` • #${job.queue_position} in queue`}
          </div>
        </div>
      </div>
      
      {status !== 'completed' && onCancel && (
        <button
          onClick={() => onCancel(job.prompt_id)}
          style={{
            background: 'transparent',
            border: 'none',
            cursor: 'pointer',
            padding: '4px',
            color: 'var(--text-muted)',
          }}
          title="Cancel job"
        >
          <X size={14} />
        </button>
      )}
      
      {status === 'completed' && job.output_video && (
        <a
          href={`${BACKEND_BASE}${job.output_video}`}
          target="_blank"
          rel="noopener noreferrer"
          style={{
            color: '#3b82f6',
            fontSize: '0.75rem',
            textDecoration: 'none',
          }}
        >
          View
        </a>
      )}
    </div>
  )
}
