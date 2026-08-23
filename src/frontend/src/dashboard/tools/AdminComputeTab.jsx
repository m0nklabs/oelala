import React, { useState, useEffect, useCallback } from 'react'
import { useAuth } from '../../contexts/AuthContext'
import { apiFetch } from '../../api'
import {
  Server, RefreshCw, Plus, Pencil, Trash2, CheckCircle2, AlertCircle,
  Power,
} from 'lucide-react'

const EMPTY_FORM = {
  id: '',
  name: '',
  type: 'comfyui',
  base_url: '',
  enabled: true,
  model_families: [],
  notes: '',
}

export default function AdminComputeTab() {
  const { isAdmin } = useAuth()
  const [backends, setBackends] = useState([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)
  const [ok, setOk] = useState(null)
  const [showForm, setShowForm] = useState(false)
  const [editing, setEditing] = useState(null) // backend id being edited
  const [form, setForm] = useState({ ...EMPTY_FORM })
  const [familiesText, setFamiliesText] = useState('')

  // Known model families for checklist
  const KNOWN_FAMILIES = [
    'wan2.2', 'sdxl', 'flux', 'flux2', 'krea2', 'minimax_h3', 'ltx',
    'qwen_image_edit', 'i2i_edit_model', 'utility',
  ]

  const fetchBackends = useCallback(async () => {
    try {
      setLoading(true)
      const resp = await apiFetch('/api/admin/backends')
      if (!resp.ok) throw new Error('Failed to fetch compute backends')
      const data = await resp.json()
      setBackends(data.backends || [])
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (isAdmin) fetchBackends()
  }, [isAdmin, fetchBackends])

  const openCreate = () => {
    setForm({ ...EMPTY_FORM })
    setFamiliesText('')
    setEditing(null)
    setShowForm(true)
  }

  const openEdit = (b) => {
    setForm({
      id: b.id,
      name: b.name,
      type: b.type,
      base_url: b.base_url || '',
      enabled: b.enabled,
      model_families: b.model_families || [],
      notes: b.notes || '',
    })
    setFamiliesText((b.model_families || []).join(', '))
    setEditing(b.id)
    setShowForm(true)
  }

  const closeForm = () => {
    setShowForm(false)
    setEditing(null)
  }

  const parseFamilies = () => familiesText
    .split(',')
    .map(s => s.trim())
    .filter(Boolean)

  const toggleFamily = (fam) => {
    const current = parseFamilies()
    const next = current.includes(fam) ? current.filter(x => x !== fam) : [...current, fam]
    setFamiliesText(next.join(', '))
  }

  const save = async (e) => {
    e.preventDefault()
    setSaving(true)
    setError(null)
    setOk(null)
    try {
      const payload = {
        ...form,
        model_families: parseFamilies(),
      }
      const method = editing ? 'PUT' : 'POST'
      const url = editing ? `/api/admin/backends/${encodeURIComponent(editing)}` : '/api/admin/backends'
      const resp = await apiFetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}))
        throw new Error(detail.detail || 'Failed to save backend')
      }
      setOk(editing ? 'Backend updated' : 'Backend created')
      closeForm()
      fetchBackends()
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  const remove = async (b) => {
    if (!window.confirm(`Delete compute backend '${b.name}'?`)) return
    try {
      const resp = await apiFetch(`/api/admin/backends/${encodeURIComponent(b.id)}`, {
        method: 'DELETE',
      })
      if (!resp.ok) throw new Error('Failed to delete backend')
      setOk('Backend deleted')
      fetchBackends()
    } catch (err) {
      setError(err.message)
    }
  }

  const toggleEnabled = async (b) => {
    try {
      const method = 'PUT'
      const payload = {
        id: b.id,
        name: b.name,
        type: b.type,
        base_url: b.base_url || '',
        enabled: !b.enabled,
        model_families: b.model_families || [],
        notes: b.notes || '',
      }
      const resp = await apiFetch(`/api/admin/backends/${encodeURIComponent(b.id)}`, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!resp.ok) throw new Error('Failed to toggle backend')
      fetchBackends()
    } catch (err) {
      setError(err.message)
    }
  }

  if (!isAdmin) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center' }}>
        <AlertCircle size={48} style={{ color: '#ef4444', marginBottom: '1rem' }} />
        <h3>Access Denied</h3>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ margin: 0 }}>Compute Backends</h3>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            onClick={fetchBackends}
            style={{
              background: 'var(--bg-input)', border: '1px solid var(--border-color)',
              color: 'var(--text-primary)', padding: '0.5rem 0.9rem', borderRadius: '6px',
              cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.4rem',
            }}
          >
            <RefreshCw size={15} /> Refresh
          </button>
          <button
            onClick={openCreate}
            style={{
              background: 'var(--accent-color)', border: 'none', color: 'white',
              padding: '0.5rem 0.9rem', borderRadius: '6px', cursor: 'pointer',
              display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 600,
            }}
          >
            <Plus size={15} /> Add Backend
          </button>
        </div>
      </div>

      {error && (
        <div style={{ padding: '0.75rem 1rem', background: 'rgba(239,68,68,0.12)', border: '1px solid #ef4444', borderRadius: '6px', color: '#ef4444', fontSize: '0.85rem' }}>
          ⚠️ {error}
        </div>
      )}
      {ok && (
        <div style={{ padding: '0.75rem 1rem', background: 'rgba(16,185,129,0.12)', border: '1px solid #10b981', borderRadius: '6px', color: '#10b981', fontSize: '0.85rem' }}>
          ✅ {ok}
        </div>
      )}

      {/* Add/Edit form */}
      {showForm && (
        <form onSubmit={save} style={{
          background: 'var(--bg-card)', border: '1px solid var(--border-color)',
          borderRadius: '8px', padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.9rem',
        }}>
          <div style={{ fontWeight: 600 }}>{editing ? 'Edit backend' : 'Add backend'}</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.9rem' }}>
            <label style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', fontSize: '0.8rem' }}>
              ID (unique)
              <input value={form.id} disabled={!!editing} onChange={(e) => setForm({ ...form, id: e.target.value })}
                style={inputStyle} placeholder="my-pc-comfyui" />
            </label>
            <label style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', fontSize: '0.8rem' }}>
              Name
              <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })}
                style={inputStyle} placeholder="My PC ComfyUI" />
            </label>
            <label style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', fontSize: '0.8rem' }}>
              Type
              <select value={form.type} onChange={(e) => setForm({ ...form, type: e.target.value })} style={inputStyle}>
                <option value="comfyui">comfyui</option>
                <option value="runpod">runpod</option>
              </select>
            </label>
            <label style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', fontSize: '0.8rem' }}>
              Base URL (comfyui only)
              <input value={form.base_url} onChange={(e) => setForm({ ...form, base_url: e.target.value })}
                style={inputStyle} placeholder="http://host:8188" />
            </label>
            <label style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', fontSize: '0.8rem' }}>
              Notes
              <input value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })}
                style={inputStyle} placeholder="optional" />
            </label>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', fontSize: '0.85rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <input type="checkbox" checked={form.enabled}
                onChange={(e) => setForm({ ...form, enabled: e.target.checked })} />
              Enabled
            </label>
          </div>

          <div>
            <div style={{ fontSize: '0.8rem', marginBottom: '0.4rem' }}>Model families</div>
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.5rem' }}>
              {KNOWN_FAMILIES.map(fam => {
                const active = parseFamilies().includes(fam)
                return (
                  <button type="button" key={fam} onClick={() => toggleFamily(fam)}
                    style={{
                      padding: '0.3rem 0.6rem', borderRadius: '20px', border: '1px solid var(--border-color)',
                      background: active ? 'var(--accent-color)' : 'var(--bg-input)',
                      color: active ? 'white' : 'var(--text-primary)', cursor: 'pointer', fontSize: '0.75rem',
                    }}>
                    {fam}
                  </button>
                )
              })}
            </div>
            <input value={familiesText} onChange={(e) => setFamiliesText(e.target.value)}
              style={{ ...inputStyle, width: '100%' }} placeholder="comma-separated model families" />
          </div>

          <div style={{ display: 'flex', gap: '0.6rem', justifyContent: 'flex-end' }}>
            <button type="button" onClick={closeForm}
              style={{ background: 'transparent', border: '1px solid var(--border-color)', padding: '0.5rem 1rem', borderRadius: '6px', cursor: 'pointer', color: 'var(--text-primary)' }}>
              Cancel
            </button>
            <button type="submit" disabled={saving}
              style={{ background: 'var(--accent-color)', border: 'none', padding: '0.5rem 1.2rem', borderRadius: '6px', cursor: 'pointer', color: 'white', fontWeight: 600 }}>
              {saving ? 'Saving...' : 'Save'}
            </button>
          </div>
        </form>
      )}

      {/* List */}
      {loading ? (
        <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>Loading...</div>
      ) : backends.length === 0 ? (
        <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>No compute backends configured.</div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {backends.map((b) => (
            <div key={b.id} style={{
              background: 'var(--bg-card)', border: '1px solid var(--border-color)',
              borderRadius: '8px', padding: '1rem',
              opacity: b.enabled ? 1 : 0.6,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', minWidth: 0 }}>
                  <Server size={18} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      {b.name}
                      {b.enabled
                        ? <CheckCircle2 size={14} color="#10b981" />
                        : <AlertCircle size={14} color="#f59e0b" />}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      <span style={{ marginRight: '0.6rem' }}>{b.type === 'runpod' ? '☁️ runpod' : `🖥️ ${b.base_url}`}</span>
                      <span>id: {b.id}</span>
                    </div>
                  </div>
                </div>
                <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center', flexShrink: 0 }}>
                  <button onClick={() => toggleEnabled(b)} title={b.enabled ? 'Disable' : 'Enable'}
                    style={{ background: 'transparent', border: '1px solid var(--border-color)', borderRadius: '6px', padding: '0.35rem 0.6rem', cursor: 'pointer', color: b.enabled ? '#ef4444' : '#10b981' }}>
                    <Power size={14} />
                  </button>
                  <button onClick={() => openEdit(b)} title="Edit"
                    style={{ background: 'transparent', border: '1px solid var(--border-color)', borderRadius: '6px', padding: '0.35rem 0.6rem', cursor: 'pointer', color: 'var(--text-primary)' }}>
                    <Pencil size={14} />
                  </button>
                  <button onClick={() => remove(b)} title="Delete"
                    style={{ background: 'transparent', border: '1px solid var(--border-color)', borderRadius: '6px', padding: '0.35rem 0.6rem', cursor: 'pointer', color: '#ef4444' }}>
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
              <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap', marginTop: '0.6rem' }}>
                {(b.model_families || []).map(f => (
                  <span key={f} style={{
                    padding: '0.15rem 0.5rem', background: 'var(--bg-input)', border: '1px solid var(--border-color)',
                    borderRadius: '20px', fontSize: '0.7rem', color: 'var(--text-primary)',
                  }}>{f}</span>
                ))}
              </div>
              {b.notes && <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>{b.notes}</div>}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

const inputStyle = {
  background: 'var(--bg-input)',
  border: '1px solid var(--border-color)',
  color: 'var(--text-primary)',
  padding: '0.5rem 0.7rem',
  borderRadius: '6px',
  fontSize: '0.85rem',
  boxSizing: 'border-box',
}
