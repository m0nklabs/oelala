import React, { useState, useEffect, useCallback } from 'react'
import { useAuth } from '../../contexts/AuthContext'
import { BACKEND_BASE } from '../../config'
import { apiFetch } from '../../api'
import {
  Cpu, HardDrive, Server, Activity,
  RefreshCw, Thermometer, MemoryStick,
  FileText, AlertCircle, CheckCircle,
  Clock, XCircle
} from 'lucide-react'

/**
 * Admin System Monitoring Tab
 * Displays GPU stats, service health, queue status, and logs
 */
export default function AdminSystemTab() {
  const { session, isAdmin } = useAuth()
  const [activeTab, setActiveTab] = useState('overview')
  const [loading, setLoading] = useState(true)

  // System data
  const [gpuData, setGpuData] = useState(null)
  const [healthData, setHealthData] = useState(null)
  const [queueData, setQueueData] = useState(null)
  const [logsData, setLogsData] = useState(null)
  const [selectedService, setSelectedService] = useState('oelala-backend')

  // AI Settings
  const [aiSettings, setAiSettings] = useState(null)
  const [aiSettingsLoading, setAiSettingsLoading] = useState(false)
  const [aiSettingsSaving, setAiSettingsSaving] = useState(false)
  const [editedPromptSystem, setEditedPromptSystem] = useState('')
  const [editedOllamaModel, setEditedOllamaModel] = useState('')

  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState(null)

  // Fetch all system data
  const fetchSystemData = useCallback(async () => {
    if (!isAdmin || !session) return

    setRefreshing(true)
    setError(null)

    try {
      // Fetch GPU, health, and queue in parallel
      const [gpuRes, healthRes, queueRes] = await Promise.all([
        apiFetch('/api/admin/system/gpu'),
        apiFetch('/api/admin/system/health'),
        apiFetch('/api/admin/system/queue'),
      ])

      if (gpuRes.ok) setGpuData(await gpuRes.json())
      if (healthRes.ok) setHealthData(await healthRes.json())
      if (queueRes.ok) setQueueData(await queueRes.json())

    } catch (err) {
      console.error('Failed to fetch system data:', err)
      setError('Failed to fetch system data')
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [isAdmin, session])

  // Fetch logs for selected service
  const fetchLogs = useCallback(async () => {
    if (!isAdmin || !session) return

    try {
      const res = await apiFetch(
        `/api/admin/system/logs?service=${selectedService}&lines=100`
      )
      if (res.ok) setLogsData(await res.json())
    } catch (err) {
      console.error('Failed to fetch logs:', err)
    }
  }, [isAdmin, session, selectedService])

  // Fetch AI settings
  const fetchAiSettings = useCallback(async () => {
    if (!isAdmin || !session) return
    setAiSettingsLoading(true)

    try {
      const res = await apiFetch('/api/admin/ai-settings')
      if (res.ok) {
        const data = await res.json()
        setAiSettings(data)
        setEditedPromptSystem(data.prompt_system || '')
        setEditedOllamaModel(data.ollama_model || '')
      }
    } catch (err) {
      console.error('Failed to fetch AI settings:', err)
    } finally {
      setAiSettingsLoading(false)
    }
  }, [isAdmin, session])

  // Save AI settings
  const saveAiSettings = async () => {
    if (!session) return
    setAiSettingsSaving(true)

    try {
      const res = await apiFetch('/api/admin/ai-settings', {
        method: 'POST',
        body: JSON.stringify({
          prompt_system: editedPromptSystem,
          ollama_model: editedOllamaModel,
        })
      })
      if (res.ok) {
        const data = await res.json()
        setAiSettings(data.settings)
        alert('AI settings saved!')
      } else {
        const err = await res.json()
        alert(`Failed to save: ${err.detail}`)
      }
    } catch (err) {
      console.error('Failed to save AI settings:', err)
      alert('Failed to save AI settings')
    } finally {
      setAiSettingsSaving(false)
    }
  }

  // Reset AI settings to defaults
  const resetAiSettings = async () => {
    if (!confirm('Reset AI settings to defaults?')) return
    if (!session) return

    try {
      const res = await apiFetch('/api/admin/ai-settings/reset', {
        method: 'POST',
      })
      if (res.ok) {
        const data = await res.json()
        setAiSettings(data.settings)
        setEditedPromptSystem(data.settings.prompt_system)
        setEditedOllamaModel(data.settings.ollama_model)
        alert('AI settings reset to defaults!')
      }
    } catch (err) {
      console.error('Failed to reset AI settings:', err)
    }
  }

  // Initial load
  useEffect(() => {
    fetchSystemData()
  }, [fetchSystemData])

  // Fetch logs when tab or service changes
  useEffect(() => {
    if (activeTab === 'logs') {
      fetchLogs()
    }
    if (activeTab === 'ai') {
      fetchAiSettings()
    }
  }, [activeTab, selectedService, fetchLogs, fetchAiSettings])

  // Auto-refresh every 10 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      if (activeTab === 'overview') fetchSystemData()
      if (activeTab === 'logs') fetchLogs()
    }, 10000)

    return () => clearInterval(interval)
  }, [activeTab, fetchSystemData, fetchLogs])

  if (!isAdmin) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center' }}>
        <AlertCircle size={48} style={{ color: '#ef4444', marginBottom: '1rem' }} />
        <h2>Access Denied</h2>
      </div>
    )
  }

  const tabs = [
    { id: 'overview', label: '📊 Overview', icon: <Activity size={16} /> },
    { id: 'queue', label: '📋 Queue', icon: <Clock size={16} /> },
    { id: 'logs', label: '📜 Logs', icon: <FileText size={16} /> },
    { id: 'ai', label: '🤖 AI Settings', icon: <Cpu size={16} /> },
  ]

  return (
    <div style={{ padding: '1.5rem' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 600, color: 'var(--text-primary)' }}>
          🖥️ System Monitoring
        </h2>
        <button
          onClick={() => {
            fetchSystemData()
            if (activeTab === 'logs') fetchLogs()
          }}
          disabled={refreshing}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.5rem 1rem',
            background: 'var(--bg-input)',
            border: '1px solid var(--border-color)',
            borderRadius: '6px',
            color: 'var(--text-primary)',
            cursor: refreshing ? 'not-allowed' : 'pointer',
            opacity: refreshing ? 0.6 : 1,
          }}
        >
          <RefreshCw size={16} className={refreshing ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              padding: '0.6rem 1rem',
              background: activeTab === tab.id ? 'var(--accent-color)' : 'transparent',
              border: 'none',
              borderRadius: '6px',
              color: activeTab === tab.id ? 'white' : 'var(--text-secondary)',
              cursor: 'pointer',
              fontWeight: activeTab === tab.id ? 600 : 400,
              transition: 'all 0.2s',
            }}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {loading ? (
        <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          Loading system data...
        </div>
      ) : (
        <>
          {activeTab === 'overview' && <OverviewTab gpuData={gpuData} healthData={healthData} queueData={queueData} />}
          {activeTab === 'queue' && <QueueTab queueData={queueData} onRefresh={fetchSystemData} />}
          {activeTab === 'logs' && (
            <LogsTab
              logsData={logsData}
              selectedService={selectedService}
              onServiceChange={setSelectedService}
              onRefresh={fetchLogs}
            />
          )}

          {activeTab === 'ai' && (
            <AISettingsTab
              aiSettings={aiSettings}
              aiSettingsLoading={aiSettingsLoading}
              aiSettingsSaving={aiSettingsSaving}
              editedPromptSystem={editedPromptSystem}
              editedOllamaModel={editedOllamaModel}
              onPromptSystemChange={setEditedPromptSystem}
              onOllamaModelChange={setEditedOllamaModel}
              onSave={saveAiSettings}
              onReset={resetAiSettings}
            />
          )}
        </>
      )}
    </div>
  )
}

// =============================================================================
// Overview Tab - GPU, Services, Disk
// =============================================================================

function OverviewTab({ gpuData, healthData, queueData }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* GPU Cards */}
      <section>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Cpu size={18} /> GPU Status
        </h3>
        {gpuData?.gpus?.length > 0 ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
            {gpuData.gpus.map((gpu) => (
              <GpuCard key={gpu.index} gpu={gpu} />
            ))}
          </div>
        ) : (
          <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)', background: 'var(--bg-card)', borderRadius: '8px' }}>
            No GPU data available
          </div>
        )}
      </section>

      {/* Services Status */}
      <section>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Server size={18} /> Services
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
          {healthData?.services && Object.entries(healthData.services).map(([name, info]) => (
            <ServiceCard key={name} name={name} info={info} />
          ))}
        </div>
      </section>

      {/* Queue Summary */}
      {queueData && (
        <section>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Clock size={18} /> Queue Summary
          </h3>
          <div style={{ display: 'flex', gap: '1rem' }}>
            <StatCard
              label="Running"
              value={queueData.running_count || 0}
              color="#10b981"
            />
            <StatCard
              label="Pending"
              value={queueData.pending_count || 0}
              color="#f59e0b"
            />
          </div>
        </section>
      )}

      {/* Disk Usage */}
      <section>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <HardDrive size={18} /> Disk Usage
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
          {healthData?.disk && Object.entries(healthData.disk).map(([name, info]) => (
            <DiskCard key={name} name={name} info={info} />
          ))}
        </div>
      </section>
    </div>
  )
}

function GpuCard({ gpu }) {
  const memPercent = gpu.memory_percent || 0
  const tempColor = gpu.temperature_c > 80 ? '#ef4444' : gpu.temperature_c > 65 ? '#f59e0b' : '#10b981'

  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border-color)',
      borderRadius: '8px',
      padding: '1rem',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
        <div>
          <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>GPU {gpu.index}</div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{gpu.name}</div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', color: tempColor }}>
          <Thermometer size={14} />
          <span style={{ fontWeight: 600 }}>{gpu.temperature_c}°C</span>
        </div>
      </div>

      {/* VRAM Bar */}
      <div style={{ marginBottom: '0.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>
          <span>VRAM</span>
          <span>{gpu.memory_used_mb}MB / {gpu.memory_total_mb}MB</span>
        </div>
        <div style={{ height: '8px', background: 'var(--bg-input)', borderRadius: '4px', overflow: 'hidden' }}>
          <div style={{
            width: `${memPercent}%`,
            height: '100%',
            background: memPercent > 90 ? '#ef4444' : memPercent > 70 ? '#f59e0b' : '#10b981',
            transition: 'width 0.3s',
          }} />
        </div>
      </div>

      {/* Utilization */}
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
        <span style={{ color: 'var(--text-muted)' }}>Utilization</span>
        <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{gpu.utilization_percent}%</span>
      </div>
    </div>
  )
}

function ServiceCard({ name, info }) {
  const isOnline = info.status === 'online'

  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border-color)',
      borderRadius: '8px',
      padding: '1rem',
      display: 'flex',
      alignItems: 'center',
      gap: '0.75rem',
    }}>
      {isOnline ? (
        <CheckCircle size={20} style={{ color: '#10b981', flexShrink: 0 }} />
      ) : (
        <XCircle size={20} style={{ color: '#ef4444', flexShrink: 0 }} />
      )}
      <div>
        <div style={{ fontWeight: 600, color: 'var(--text-primary)', textTransform: 'capitalize' }}>
          {name}
        </div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Port {info.port} • {info.status}
        </div>
      </div>
    </div>
  )
}

function DiskCard({ name, info }) {
  const percent = info.percent || 0

  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border-color)',
      borderRadius: '8px',
      padding: '1rem',
    }}>
      <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.5rem', textTransform: 'capitalize' }}>
        {name}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>
        <span>{info.used_gb}GB used</span>
        <span>{info.free_gb}GB free</span>
      </div>
      <div style={{ height: '8px', background: 'var(--bg-input)', borderRadius: '4px', overflow: 'hidden' }}>
        <div style={{
          width: `${percent}%`,
          height: '100%',
          background: percent > 90 ? '#ef4444' : percent > 75 ? '#f59e0b' : '#10b981',
        }} />
      </div>
      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem', textAlign: 'right' }}>
        {percent.toFixed(1)}% used
      </div>
    </div>
  )
}

function StatCard({ label, value, color }) {
  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border-color)',
      borderRadius: '8px',
      padding: '1rem',
      minWidth: '120px',
    }}>
      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>{label}</div>
      <div style={{ fontSize: '1.8rem', fontWeight: 700, color }}>{value}</div>
    </div>
  )
}

// =============================================================================
// Queue Tab
// =============================================================================

function QueueTab({ queueData, onRefresh }) {
  if (!queueData) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        Queue data unavailable
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Status Badge */}
      <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
        <StatusBadge status={queueData.status} />
        {queueData.message && (
          <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>{queueData.message}</span>
        )}
      </div>

      {/* Running Jobs */}
      <section>
        <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.75rem', color: '#10b981' }}>
          🔄 Running ({queueData.running?.length || 0})
        </h3>
        {queueData.running?.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {queueData.running.map((job, idx) => (
              <JobCard key={job.prompt_id || idx} job={job} status="running" />
            ))}
          </div>
        ) : (
          <div style={{ padding: '1rem', background: 'var(--bg-card)', borderRadius: '6px', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            No jobs currently running
          </div>
        )}
      </section>

      {/* Pending Jobs */}
      <section>
        <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.75rem', color: '#f59e0b' }}>
          ⏳ Pending ({queueData.pending?.length || 0})
        </h3>
        {queueData.pending?.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {queueData.pending.map((job, idx) => (
              <JobCard key={job.prompt_id || idx} job={job} status="pending" />
            ))}
          </div>
        ) : (
          <div style={{ padding: '1rem', background: 'var(--bg-card)', borderRadius: '6px', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            No jobs in queue
          </div>
        )}
      </section>
    </div>
  )
}

function StatusBadge({ status }) {
  const colors = {
    ok: { bg: '#10b98120', color: '#10b981', text: 'Online' },
    offline: { bg: '#ef444420', color: '#ef4444', text: 'Offline' },
    error: { bg: '#ef444420', color: '#ef4444', text: 'Error' },
    timeout: { bg: '#f59e0b20', color: '#f59e0b', text: 'Timeout' },
  }

  const style = colors[status] || colors.error

  return (
    <span style={{
      padding: '0.25rem 0.75rem',
      background: style.bg,
      color: style.color,
      borderRadius: '999px',
      fontSize: '0.8rem',
      fontWeight: 600,
    }}>
      {style.text}
    </span>
  )
}

function JobCard({ job, status }) {
  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border-color)',
      borderRadius: '6px',
      padding: '0.75rem 1rem',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
    }}>
      <div>
        <code style={{ fontSize: '0.85rem', color: 'var(--text-primary)' }}>
          {job.prompt_id}
        </code>
        {job.position && (
          <span style={{ marginLeft: '0.5rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Position #{job.position}
          </span>
        )}
      </div>
      <span style={{
        padding: '0.2rem 0.5rem',
        background: status === 'running' ? '#10b98120' : '#f59e0b20',
        color: status === 'running' ? '#10b981' : '#f59e0b',
        borderRadius: '4px',
        fontSize: '0.75rem',
        fontWeight: 500,
      }}>
        {status}
      </span>
    </div>
  )
}

// =============================================================================
// Logs Tab
// =============================================================================

function LogsTab({ logsData, selectedService, onServiceChange, onRefresh }) {
  const services = ['oelala-backend', 'comfyui', 'minio', 'oelala-frontend']

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Service Selector */}
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
        {services.map((svc) => (
          <button
            key={svc}
            onClick={() => onServiceChange(svc)}
            style={{
              padding: '0.5rem 1rem',
              background: selectedService === svc ? 'var(--accent-color)' : 'var(--bg-input)',
              border: selectedService === svc ? 'none' : '1px solid var(--border-color)',
              borderRadius: '6px',
              color: selectedService === svc ? 'white' : 'var(--text-secondary)',
              cursor: 'pointer',
              fontSize: '0.85rem',
              fontWeight: selectedService === svc ? 600 : 400,
            }}
          >
            {svc}
          </button>
        ))}
      </div>

      {/* Log Output */}
      <div style={{
        background: '#1e1e1e',
        border: '1px solid var(--border-color)',
        borderRadius: '8px',
        padding: '1rem',
        fontFamily: 'monospace',
        fontSize: '0.75rem',
        lineHeight: 1.5,
        maxHeight: '500px',
        overflowY: 'auto',
        color: '#d4d4d4',
      }}>
        {logsData?.lines?.length > 0 ? (
          logsData.lines.map((line, idx) => (
            <div key={idx} style={{
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-all',
              color: line.includes('ERROR') ? '#ef4444' :
                     line.includes('WARNING') ? '#f59e0b' :
                     line.includes('INFO') ? '#10b981' : '#d4d4d4',
            }}>
              {line}
            </div>
          ))
        ) : (
          <div style={{ color: 'var(--text-muted)' }}>No logs available</div>
        )}
      </div>

      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
        Showing last {logsData?.count || 0} lines • Auto-refreshes every 10s
      </div>
    </div>
  )
}

// =============================================================================
// AI Settings Tab
// =============================================================================

function AISettingsTab({
  aiSettings,
  aiSettingsLoading,
  aiSettingsSaving,
  editedPromptSystem,
  editedOllamaModel,
  onPromptSystemChange,
  onOllamaModelChange,
  onSave,
  onReset,
}) {
  if (aiSettingsLoading) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        Loading AI settings...
      </div>
    )
  }

  const hasChanges = aiSettings && (
    editedPromptSystem !== aiSettings.prompt_system ||
    editedOllamaModel !== aiSettings.ollama_model
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Ollama Model Selection */}
      <section style={{ background: 'var(--bg-card)', borderRadius: '8px', padding: '1.5rem' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          🤖 Ollama Model
        </h3>
        <div style={{ marginBottom: '0.5rem' }}>
          <select
            value={editedOllamaModel}
            onChange={(e) => onOllamaModelChange(e.target.value)}
            style={{
              width: '100%',
              padding: '0.75rem',
              background: 'var(--bg-input)',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              color: 'var(--text-primary)',
              fontSize: '0.9rem',
            }}
          >
            {aiSettings?.available_models?.map((model) => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        </div>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          Current: {aiSettings?.ollama_model || 'gemma2:9b'}
        </p>
      </section>

      {/* System Prompt Editor */}
      <section style={{ background: 'var(--bg-card)', borderRadius: '8px', padding: '1.5rem' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          📝 Prompt Enhancement System Prompt
        </h3>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
          This is the system prompt sent to the LLM when users click the ✨ enhance button.
        </p>
        <textarea
          value={editedPromptSystem}
          onChange={(e) => onPromptSystemChange(e.target.value)}
          rows={15}
          style={{
            width: '100%',
            padding: '1rem',
            background: 'var(--bg-input)',
            border: '1px solid var(--border-color)',
            borderRadius: '6px',
            color: 'var(--text-primary)',
            fontSize: '0.85rem',
            fontFamily: 'monospace',
            lineHeight: 1.6,
            resize: 'vertical',
          }}
        />
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
          {editedPromptSystem.length} characters
        </div>
      </section>

      {/* Action Buttons */}
      <div style={{ display: 'flex', gap: '1rem', justifyContent: 'flex-end' }}>
        <button
          onClick={onReset}
          style={{
            padding: '0.75rem 1.5rem',
            background: 'var(--bg-input)',
            border: '1px solid var(--border-color)',
            borderRadius: '6px',
            color: 'var(--text-primary)',
            cursor: 'pointer',
          }}
        >
          Reset to Defaults
        </button>
        <button
          onClick={onSave}
          disabled={aiSettingsSaving || !hasChanges}
          style={{
            padding: '0.75rem 1.5rem',
            background: hasChanges ? 'var(--accent-primary)' : 'var(--bg-input)',
            border: 'none',
            borderRadius: '6px',
            color: hasChanges ? 'white' : 'var(--text-muted)',
            cursor: aiSettingsSaving || !hasChanges ? 'not-allowed' : 'pointer',
            opacity: aiSettingsSaving ? 0.6 : 1,
          }}
        >
          {aiSettingsSaving ? 'Saving...' : 'Save Changes'}
        </button>
      </div>
    </div>
  )
}
