import React, { useState, useEffect, useRef } from 'react'
import { User, Save, RefreshCw, CheckCircle, AlertCircle, Twitter, Instagram, Youtube, Github, Globe, Link2, Camera } from 'lucide-react'
import { useAuth } from '../../contexts/AuthContext'
import { BACKEND_BASE } from '../../config'

/**
 * Profile editing tool - allows users to view and edit their profile.
 */
export default function ProfileTool() {
  const { user, token } = useAuth()
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)
  const [stats, setStats] = useState(null)
  const [uploadingAvatar, setUploadingAvatar] = useState(false)
  const avatarInputRef = useRef(null)

  // Profile form state
  const [profile, setProfile] = useState({
    username: '',
    display_name: '',
    bio: '',
    avatar_url: '',
    social_links: {},
    is_public: true,
  })

  // Fetch profile on mount
  useEffect(() => {
    if (!user || !token) {
      setLoading(false)
      return
    }
    fetchProfile()
    fetchStats()
  }, [user, token])

  async function handleAvatarUpload(e) {
    const file = e.target.files?.[0]
    if (!file) return
    setUploadingAvatar(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${BACKEND_BASE}/api/profile/me/avatar`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: form,
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || 'Upload failed')
      }
      const data = await res.json()
      setProfile((p) => ({ ...p, avatar_url: data.avatar_url }))
      setSuccess('Avatar updated!')
      setTimeout(() => setSuccess(null), 3000)
    } catch (err) {
      setError(err.message)
    } finally {
      setUploadingAvatar(false)
      // reset so same file can trigger onChange again
      if (avatarInputRef.current) avatarInputRef.current.value = ''
    }
  }

  async function fetchProfile() {
    try {
      setLoading(true)
      const res = await fetch(`${BACKEND_BASE}/api/profile/me`, {
        headers: { Authorization: `Bearer ${token}` },
      })

      if (res.ok) {
        const data = await res.json()
        setProfile({
          username: data.username || '',
          display_name: data.display_name || '',
          bio: data.bio || '',
          avatar_url: data.avatar_url || '',
          social_links: data.social_links || {},
          is_public: data.is_public ?? true,
        })
      }
    } catch (err) {
      console.error('Failed to fetch profile:', err)
      setError('Failed to load profile')
    } finally {
      setLoading(false)
    }
  }

  async function fetchStats() {
    try {
      const res = await fetch(`${BACKEND_BASE}/api/profile/me/stats`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (res.ok) {
        const data = await res.json()
        setStats(data)
      }
    } catch (err) {
      console.error('Failed to fetch stats:', err)
    }
  }

  async function handleSave() {
    if (!token) return

    try {
      setSaving(true)
      setError(null)
      setSuccess(null)

      const res = await fetch(`${BACKEND_BASE}/api/profile/me`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(profile),
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Failed to save profile')
      }

      setSuccess('Profile saved successfully!')
      setTimeout(() => setSuccess(null), 3000)
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  function handleInputChange(field, value) {
    setProfile(prev => ({ ...prev, [field]: value }))
  }

  function handleSocialLinkChange(platform, value) {
    setProfile(prev => ({
      ...prev,
      social_links: { ...prev.social_links, [platform]: value },
    }))
  }

  // Not logged in
  if (!user) {
    return (
      <div style={styles.container}>
        <div style={styles.emptyState}>
          <User size={48} style={{ opacity: 0.5, marginBottom: 16 }} />
          <h2>Sign in to view your profile</h2>
          <p style={{ color: 'var(--text-muted)' }}>
            Create an account to build your creator profile and showcase your work.
          </p>
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div style={styles.container}>
        <div style={styles.loading}>
          <RefreshCw size={24} className="spin" />
          <span>Loading profile...</span>
        </div>
      </div>
    )
  }

  const socialPlatforms = [
    { key: 'twitter', label: 'Twitter / X', icon: Twitter, placeholder: '@username' },
    { key: 'instagram', label: 'Instagram', icon: Instagram, placeholder: '@username' },
    { key: 'youtube', label: 'YouTube', icon: Youtube, placeholder: 'channel URL' },
    { key: 'github', label: 'GitHub', icon: Github, placeholder: 'username' },
    { key: 'website', label: 'Website', icon: Globe, placeholder: 'https://...' },
  ]

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <h1 style={styles.title}>
          <User size={24} />
          My Profile
        </h1>
        <button
          onClick={handleSave}
          disabled={saving}
          style={styles.saveButton}
        >
          {saving ? <RefreshCw size={16} className="spin" /> : <Save size={16} />}
          {saving ? 'Saving...' : 'Save Changes'}
        </button>
      </div>

      {/* Alerts */}
      {error && (
        <div style={styles.alert.error}>
          <AlertCircle size={16} />
          {error}
        </div>
      )}
      {success && (
        <div style={styles.alert.success}>
          <CheckCircle size={16} />
          {success}
        </div>
      )}

      <div style={styles.content}>
        {/* Left Column - Avatar & Stats */}
        <div style={styles.leftColumn}>
          {/* Avatar */}
          <div style={styles.avatarSection}>
            <div style={styles.avatarWrapper}>
              {profile.avatar_url ? (
                <img src={profile.avatar_url} alt="Avatar" style={styles.avatar} />
              ) : (
                <div style={styles.avatarPlaceholder}>
                  <User size={48} />
                </div>
              )}
              <button
                style={styles.avatarEditButton}
                title={uploadingAvatar ? 'Uploading…' : 'Upload avatar'}
                disabled={uploadingAvatar}
                onClick={() => avatarInputRef.current?.click()}
              >
                {uploadingAvatar ? <RefreshCw size={14} style={{ animation: 'spin 1s linear infinite' }} /> : <Camera size={14} />}
              </button>
              <input
                ref={avatarInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp,image/gif"
                style={{ display: 'none' }}
                onChange={handleAvatarUpload}
              />
            </div>
          </div>

          {/* Stats */}
          {stats && (
            <div style={styles.statsCard}>
              <h3 style={styles.statsTitle}>Stats</h3>
              <div style={styles.statRow}>
                <span>Media created</span>
                <strong>{stats.total_media || 0}</strong>
              </div>
              <div style={styles.statRow}>
                <span>Published</span>
                <strong>{stats.published_count || 0}</strong>
              </div>
              <div style={styles.statRow}>
                <span>Total likes</span>
                <strong>{stats.total_likes || 0}</strong>
              </div>
              <div style={styles.statRow}>
                <span>Followers</span>
                <strong>{stats.follower_count || 0}</strong>
              </div>
              <div style={styles.statRow}>
                <span>Following</span>
                <strong>{stats.following_count || 0}</strong>
              </div>
            </div>
          )}
        </div>

        {/* Right Column - Form */}
        <div style={styles.rightColumn}>
          {/* Basic Info */}
          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>Basic Information</h3>

            <div style={styles.formGroup}>
              <label style={styles.label}>Username</label>
              <input
                type="text"
                value={profile.username}
                onChange={(e) => handleInputChange('username', e.target.value)}
                placeholder="your-unique-username"
                style={styles.input}
                maxLength={30}
              />
              <span style={styles.hint}>3-30 characters, letters, numbers, _ and - only</span>
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>Display Name</label>
              <input
                type="text"
                value={profile.display_name}
                onChange={(e) => handleInputChange('display_name', e.target.value)}
                placeholder="Your Name"
                style={styles.input}
                maxLength={100}
              />
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>Bio</label>
              <textarea
                value={profile.bio}
                onChange={(e) => handleInputChange('bio', e.target.value)}
                placeholder="Tell others about yourself..."
                style={styles.textarea}
                rows={4}
                maxLength={500}
              />
              <span style={styles.hint}>{profile.bio.length}/500 characters</span>
            </div>

            <div style={styles.formGroup}>
              <label style={styles.checkboxLabel}>
                <input
                  type="checkbox"
                  checked={profile.is_public}
                  onChange={(e) => handleInputChange('is_public', e.target.checked)}
                  style={styles.checkbox}
                />
                Public profile
              </label>
              <span style={styles.hint}>Allow others to see your profile and published works</span>
            </div>
          </div>

          {/* Social Links */}
          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>
              <Link2 size={16} />
              Social Links
            </h3>

            {socialPlatforms.map(({ key, label, icon: Icon, placeholder }) => (
              <div key={key} style={styles.socialRow}>
                <Icon size={16} style={{ opacity: 0.6, flexShrink: 0 }} />
                <input
                  type="text"
                  value={profile.social_links[key] || ''}
                  onChange={(e) => handleSocialLinkChange(key, e.target.value)}
                  placeholder={placeholder}
                  style={styles.socialInput}
                />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

const styles = {
  container: {
    padding: '24px',
    maxWidth: '900px',
    margin: '0 auto',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '24px',
  },
  title: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    fontSize: '1.5rem',
    fontWeight: 600,
    color: 'var(--text-primary)',
    margin: 0,
  },
  saveButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '10px 20px',
    background: 'linear-gradient(135deg, #7c3aed, #a855f7)',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    fontSize: '0.9rem',
    fontWeight: 500,
    cursor: 'pointer',
    transition: 'opacity 0.2s',
  },
  alert: {
    error: {
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      padding: '12px 16px',
      background: 'rgba(239, 68, 68, 0.1)',
      border: '1px solid rgba(239, 68, 68, 0.3)',
      borderRadius: '8px',
      color: '#ef4444',
      marginBottom: '16px',
    },
    success: {
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      padding: '12px 16px',
      background: 'rgba(16, 185, 129, 0.1)',
      border: '1px solid rgba(16, 185, 129, 0.3)',
      borderRadius: '8px',
      color: '#10b981',
      marginBottom: '16px',
    },
  },
  content: {
    display: 'grid',
    gridTemplateColumns: '280px 1fr',
    gap: '24px',
  },
  leftColumn: {
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
  },
  rightColumn: {
    display: 'flex',
    flexDirection: 'column',
    gap: '24px',
  },
  avatarSection: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '12px',
    padding: '20px',
    background: 'var(--bg-card)',
    border: '1px solid var(--border-color)',
    borderRadius: '12px',
  },
  avatarWrapper: {
    position: 'relative',
  },
  avatar: {
    width: '120px',
    height: '120px',
    borderRadius: '50%',
    objectFit: 'cover',
    border: '3px solid var(--border-color)',
  },
  avatarPlaceholder: {
    width: '120px',
    height: '120px',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'var(--bg-input)',
    border: '3px solid var(--border-color)',
    color: 'var(--text-muted)',
  },
  avatarEditButton: {
    position: 'absolute',
    bottom: '4px',
    right: '4px',
    width: '32px',
    height: '32px',
    borderRadius: '50%',
    background: '#7c3aed',
    border: 'none',
    color: 'white',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
  },
  avatarUrlInput: {
    width: '100%',
    padding: '8px 12px',
    background: 'var(--bg-input)',
    border: '1px solid var(--border-color)',
    borderRadius: '6px',
    color: 'var(--text-primary)',
    fontSize: '0.8rem',
  },
  statsCard: {
    padding: '16px',
    background: 'var(--bg-card)',
    border: '1px solid var(--border-color)',
    borderRadius: '12px',
  },
  statsTitle: {
    fontSize: '0.9rem',
    fontWeight: 600,
    color: 'var(--text-primary)',
    marginBottom: '12px',
  },
  statRow: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '8px 0',
    borderBottom: '1px solid var(--border-color)',
    fontSize: '0.85rem',
    color: 'var(--text-secondary)',
  },
  section: {
    padding: '20px',
    background: 'var(--bg-card)',
    border: '1px solid var(--border-color)',
    borderRadius: '12px',
  },
  sectionTitle: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '1rem',
    fontWeight: 600,
    color: 'var(--text-primary)',
    marginBottom: '16px',
  },
  formGroup: {
    marginBottom: '16px',
  },
  label: {
    display: 'block',
    fontSize: '0.85rem',
    fontWeight: 500,
    color: 'var(--text-secondary)',
    marginBottom: '6px',
  },
  input: {
    width: '100%',
    padding: '10px 14px',
    background: 'var(--bg-input)',
    border: '1px solid var(--border-color)',
    borderRadius: '8px',
    color: 'var(--text-primary)',
    fontSize: '0.9rem',
  },
  textarea: {
    width: '100%',
    padding: '10px 14px',
    background: 'var(--bg-input)',
    border: '1px solid var(--border-color)',
    borderRadius: '8px',
    color: 'var(--text-primary)',
    fontSize: '0.9rem',
    resize: 'vertical',
    fontFamily: 'inherit',
  },
  hint: {
    display: 'block',
    fontSize: '0.75rem',
    color: 'var(--text-muted)',
    marginTop: '4px',
  },
  checkboxLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '0.9rem',
    color: 'var(--text-primary)',
    cursor: 'pointer',
  },
  checkbox: {
    width: '16px',
    height: '16px',
    accentColor: '#7c3aed',
  },
  socialRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    marginBottom: '10px',
  },
  socialInput: {
    flex: 1,
    padding: '8px 12px',
    background: 'var(--bg-input)',
    border: '1px solid var(--border-color)',
    borderRadius: '6px',
    color: 'var(--text-primary)',
    fontSize: '0.85rem',
  },
  loading: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '12px',
    padding: '40px',
    color: 'var(--text-muted)',
  },
  emptyState: {
    textAlign: 'center',
    padding: '60px 20px',
    color: 'var(--text-primary)',
  },
}
