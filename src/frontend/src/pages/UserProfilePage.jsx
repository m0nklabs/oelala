import React, { useState, useEffect, useCallback } from 'react'
import { User, UserPlus, UserMinus, ArrowLeft, Heart, Eye, Image as ImageIcon } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import { apiFetch } from '../api'
import { BACKEND_BASE } from '../config'

/**
 * Public user profile page — shows profile info, follow button,
 * follower/following counts, and the user's published gallery items.
 */
export default function UserProfilePage({ userId, username, onBack, onOpenItem }) {
  const { user, token } = useAuth()
  const [profile, setProfile] = useState(null)
  const [isFollowing, setIsFollowing] = useState(false)
  const [loading, setLoading] = useState(true)
  const [followLoading, setFollowLoading] = useState(false)
  const [media, setMedia] = useState([])
  const [followers, setFollowers] = useState([])
  const [following, setFollowing] = useState([])
  const [activeTab, setActiveTab] = useState('media') // media | followers | following

  const isOwnProfile = user && profile && user.id === profile.id

  // Fetch profile
  const fetchProfile = useCallback(async () => {
    try {
      setLoading(true)
      const endpoint = userId
        ? `/api/profile/id/${userId}`
        : `/api/profile/username/${username}`
      const res = await apiFetch(endpoint)
      if (res.ok) {
        const data = await res.json()
        setProfile(data)
      }
    } catch (err) {
      console.error('Failed to fetch profile:', err)
    } finally {
      setLoading(false)
    }
  }, [userId, username])

  // Check follow status
  const checkFollowing = useCallback(async (targetId) => {
    if (!token || !targetId) return
    try {
      const res = await apiFetch(`/api/profile/${targetId}/is-following`)
      if (res.ok) {
        const data = await res.json()
        setIsFollowing(data.is_following)
      }
    } catch (err) {
      console.error('Failed to check follow status:', err)
    }
  }, [token])

  // Fetch user's published media
  const fetchMedia = useCallback(async (targetId) => {
    try {
      const res = await apiFetch(`/api/gallery?user_id=${targetId}&limit=50`)
      if (res.ok) {
        const data = await res.json()
        setMedia(data.items || [])
      }
    } catch (err) {
      console.error('Failed to fetch user media:', err)
    }
  }, [])

  // Fetch followers
  const fetchFollowers = useCallback(async (targetId) => {
    try {
      const res = await apiFetch(`/api/profile/${targetId}/followers`)
      if (res.ok) {
        const data = await res.json()
        setFollowers(data.followers || [])
      }
    } catch (err) {
      console.error('Failed to fetch followers:', err)
    }
  }, [])

  // Fetch following
  const fetchFollowing = useCallback(async (targetId) => {
    try {
      const res = await apiFetch(`/api/profile/${targetId}/following`)
      if (res.ok) {
        const data = await res.json()
        setFollowing(data.following || [])
      }
    } catch (err) {
      console.error('Failed to fetch following:', err)
    }
  }, [])

  useEffect(() => {
    fetchProfile()
  }, [fetchProfile])

  useEffect(() => {
    if (profile?.id) {
      checkFollowing(profile.id)
      fetchMedia(profile.id)
    }
  }, [profile?.id, checkFollowing, fetchMedia])

  useEffect(() => {
    if (profile?.id && activeTab === 'followers') fetchFollowers(profile.id)
    if (profile?.id && activeTab === 'following') fetchFollowing(profile.id)
  }, [profile?.id, activeTab, fetchFollowers, fetchFollowing])

  // Follow / Unfollow
  const handleFollowToggle = async () => {
    if (!token || !profile) return
    setFollowLoading(true)
    try {
      const method = isFollowing ? 'DELETE' : 'POST'
      const res = await apiFetch(`/api/profile/${profile.id}/follow`, { method })
      if (res.ok) {
        const data = await res.json()
        setIsFollowing(data.followed)
        setProfile(prev => ({
          ...prev,
          follower_count: data.follower_count,
        }))
      }
    } catch (err) {
      console.error('Follow toggle failed:', err)
    } finally {
      setFollowLoading(false)
    }
  }

  if (loading) {
    return (
      <div style={styles.container}>
        <div style={styles.loading}>Loading profile...</div>
      </div>
    )
  }

  if (!profile) {
    return (
      <div style={styles.container}>
        <button onClick={onBack} style={styles.backButton}>
          <ArrowLeft size={18} /> Back
        </button>
        <div style={styles.notFound}>Profile not found</div>
      </div>
    )
  }

  const avatarSrc = profile.avatar_url
    ? (profile.avatar_url.startsWith('http') ? profile.avatar_url : `${BACKEND_BASE}${profile.avatar_url}`)
    : null

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <button onClick={onBack} style={styles.backButton}>
          <ArrowLeft size={18} /> Back
        </button>
      </div>

      {/* Profile card */}
      <div style={styles.profileCard}>
        <div style={styles.avatarSection}>
          {avatarSrc ? (
            <img src={avatarSrc} alt="avatar" style={styles.avatar} />
          ) : (
            <div style={styles.avatarPlaceholder}>
              <User size={48} />
            </div>
          )}
        </div>

        <div style={styles.profileInfo}>
          <h2 style={styles.displayName}>
            {profile.display_name || profile.username || 'User'}
          </h2>
          {profile.username && (
            <span style={styles.username}>@{profile.username}</span>
          )}
          {profile.bio && <p style={styles.bio}>{profile.bio}</p>}

          {/* Stats row */}
          <div style={styles.statsRow}>
            <button
              style={{ ...styles.statButton, ...(activeTab === 'media' ? styles.statButtonActive : {}) }}
              onClick={() => setActiveTab('media')}
            >
              <strong>{media.length}</strong>
              <span>Media</span>
            </button>
            <button
              style={{ ...styles.statButton, ...(activeTab === 'followers' ? styles.statButtonActive : {}) }}
              onClick={() => setActiveTab('followers')}
            >
              <strong>{profile.follower_count || 0}</strong>
              <span>Followers</span>
            </button>
            <button
              style={{ ...styles.statButton, ...(activeTab === 'following' ? styles.statButtonActive : {}) }}
              onClick={() => setActiveTab('following')}
            >
              <strong>{profile.following_count || 0}</strong>
              <span>Following</span>
            </button>
          </div>

          {/* Follow button (only for other users) */}
          {user && !isOwnProfile && (
            <button
              style={isFollowing ? styles.unfollowButton : styles.followButton}
              onClick={handleFollowToggle}
              disabled={followLoading}
            >
              {isFollowing ? (
                <><UserMinus size={16} /> Unfollow</>
              ) : (
                <><UserPlus size={16} /> Follow</>
              )}
            </button>
          )}
        </div>
      </div>

      {/* Tab content */}
      <div style={styles.tabContent}>
        {activeTab === 'media' && (
          <div style={styles.mediaGrid}>
            {media.length === 0 ? (
              <div style={styles.empty}>No published media yet</div>
            ) : (
              media.map(item => (
                <div
                  key={item.id}
                  style={styles.mediaCard}
                  onClick={() => onOpenItem?.(item)}
                >
                  <img
                    src={`${BACKEND_BASE}${item.thumbnail_url || item.media_url}`}
                    alt={item.title || ''}
                    style={styles.mediaThumb}
                    loading="lazy"
                  />
                  <div style={styles.mediaOverlay}>
                    <span><Heart size={12} /> {item.like_count || 0}</span>
                    <span><Eye size={12} /> {item.view_count || 0}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {activeTab === 'followers' && (
          <div style={styles.userList}>
            {followers.length === 0 ? (
              <div style={styles.empty}>No followers yet</div>
            ) : (
              followers.map(f => (
                <UserListItem key={f.id} user={f} onBack={onBack} />
              ))
            )}
          </div>
        )}

        {activeTab === 'following' && (
          <div style={styles.userList}>
            {following.length === 0 ? (
              <div style={styles.empty}>Not following anyone yet</div>
            ) : (
              following.map(f => (
                <UserListItem key={f.id} user={f} onBack={onBack} />
              ))
            )}
          </div>
        )}
      </div>
    </div>
  )
}

/** Compact user list item used in followers/following lists */
function UserListItem({ user: u }) {
  const avatarSrc = u.avatar_url
    ? (u.avatar_url.startsWith('http') ? u.avatar_url : `${BACKEND_BASE}${u.avatar_url}`)
    : null

  return (
    <div style={styles.userListItem}>
      {avatarSrc ? (
        <img src={avatarSrc} alt="" style={styles.userListAvatar} />
      ) : (
        <div style={styles.userListAvatarPlaceholder}><User size={20} /></div>
      )}
      <div style={styles.userListInfo}>
        <strong>{u.display_name || u.username || 'User'}</strong>
        {u.username && <span style={styles.userListUsername}>@{u.username}</span>}
      </div>
    </div>
  )
}

// =============================================================================
// Styles
// =============================================================================
const styles = {
  container: {
    maxWidth: 900,
    margin: '0 auto',
    padding: '24px 16px',
    color: '#e0e0e0',
  },
  loading: {
    textAlign: 'center',
    padding: 60,
    color: '#888',
    fontSize: 16,
  },
  notFound: {
    textAlign: 'center',
    padding: 60,
    color: '#888',
    fontSize: 18,
  },
  header: {
    marginBottom: 20,
  },
  backButton: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 6,
    background: 'transparent',
    border: '1px solid #444',
    color: '#bbb',
    padding: '8px 16px',
    borderRadius: 8,
    cursor: 'pointer',
    fontSize: 14,
  },
  profileCard: {
    display: 'flex',
    gap: 24,
    padding: 24,
    background: 'rgba(255,255,255,0.04)',
    borderRadius: 16,
    border: '1px solid rgba(255,255,255,0.08)',
    marginBottom: 24,
    flexWrap: 'wrap',
  },
  avatarSection: {
    flexShrink: 0,
  },
  avatar: {
    width: 120,
    height: 120,
    borderRadius: '50%',
    objectFit: 'cover',
    border: '3px solid #6c5ce7',
  },
  avatarPlaceholder: {
    width: 120,
    height: 120,
    borderRadius: '50%',
    background: 'rgba(108,92,231,0.15)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#6c5ce7',
    border: '3px solid #6c5ce7',
  },
  profileInfo: {
    flex: 1,
    minWidth: 200,
  },
  displayName: {
    margin: 0,
    fontSize: 24,
    fontWeight: 700,
    color: '#fff',
  },
  username: {
    color: '#888',
    fontSize: 14,
    display: 'block',
    marginBottom: 8,
  },
  bio: {
    color: '#bbb',
    fontSize: 14,
    lineHeight: 1.5,
    marginBottom: 12,
  },
  statsRow: {
    display: 'flex',
    gap: 4,
    marginBottom: 16,
  },
  statButton: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 2,
    padding: '8px 16px',
    background: 'transparent',
    border: '1px solid rgba(255,255,255,0.08)',
    borderRadius: 8,
    color: '#bbb',
    cursor: 'pointer',
    fontSize: 12,
    minWidth: 80,
  },
  statButtonActive: {
    background: 'rgba(108,92,231,0.15)',
    borderColor: '#6c5ce7',
    color: '#fff',
  },
  followButton: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 6,
    padding: '10px 24px',
    background: 'linear-gradient(135deg, #6c5ce7, #a855f7)',
    border: 'none',
    borderRadius: 8,
    color: '#fff',
    fontWeight: 600,
    fontSize: 14,
    cursor: 'pointer',
  },
  unfollowButton: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 6,
    padding: '10px 24px',
    background: 'transparent',
    border: '1px solid #666',
    borderRadius: 8,
    color: '#bbb',
    fontWeight: 600,
    fontSize: 14,
    cursor: 'pointer',
  },
  tabContent: {
    marginTop: 8,
  },
  mediaGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
    gap: 12,
  },
  mediaCard: {
    position: 'relative',
    borderRadius: 12,
    overflow: 'hidden',
    cursor: 'pointer',
    aspectRatio: '1 / 1',
    background: '#1a1a2e',
  },
  mediaThumb: {
    width: '100%',
    height: '100%',
    objectFit: 'cover',
  },
  mediaOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    display: 'flex',
    gap: 12,
    padding: '6px 10px',
    background: 'linear-gradient(transparent, rgba(0,0,0,0.7))',
    color: '#fff',
    fontSize: 12,
  },
  userList: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  userListItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    padding: '12px 16px',
    background: 'rgba(255,255,255,0.04)',
    borderRadius: 12,
    border: '1px solid rgba(255,255,255,0.06)',
  },
  userListAvatar: {
    width: 40,
    height: 40,
    borderRadius: '50%',
    objectFit: 'cover',
  },
  userListAvatarPlaceholder: {
    width: 40,
    height: 40,
    borderRadius: '50%',
    background: 'rgba(108,92,231,0.15)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#6c5ce7',
  },
  userListInfo: {
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
    color: '#fff',
    fontSize: 14,
  },
  userListUsername: {
    color: '#888',
    fontSize: 12,
  },
  empty: {
    textAlign: 'center',
    padding: 40,
    color: '#666',
    fontSize: 14,
  },
}
