import React, { useState, useEffect } from 'react'
import { useAuth } from '../../contexts/AuthContext'
import { BACKEND_BASE } from '../../config'
import { 
  Users, Search, Coins, Award, Shield, Crown, 
  ChevronDown, ChevronUp, Edit2, Trash2, Plus, Minus,
  TrendingUp, Activity
} from 'lucide-react'

export default function AdminPanel() {
  const { session, isAdmin } = useAuth()
  const [loading, setLoading] = useState(true)
  const [users, setUsers] = useState([])
  const [stats, setStats] = useState(null)
  const [selectedUser, setSelectedUser] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [filterTier, setFilterTier] = useState('')
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const [expandedUser, setExpandedUser] = useState(null)
  const [transactions, setTransactions] = useState([])

  // Credit adjustment modal
  const [showCreditModal, setShowCreditModal] = useState(false)
  const [creditAdjustUser, setCreditAdjustUser] = useState(null)
  const [creditAmount, setCreditAmount] = useState('')
  const [creditReason, setCreditReason] = useState('')

  // Fetch stats
  useEffect(() => {
    if (!isAdmin || !session) return
    
    const fetchStats = async () => {
      try {
        const response = await fetch(`${BACKEND_BASE}/api/admin/stats`, {
          headers: {
            Authorization: `Bearer ${session.access_token}`,
          },
        })
        
        if (response.ok) {
          const data = await response.json()
          setStats(data)
        }
      } catch (error) {
        console.error('Failed to fetch stats:', error)
      }
    }
    
    fetchStats()
  }, [isAdmin, session])

  // Fetch users
  useEffect(() => {
    if (!isAdmin || !session) return
    
    const fetchUsers = async () => {
      setLoading(true)
      try {
        const params = new URLSearchParams({
          page: page.toString(),
          per_page: '20',
        })
        
        if (filterTier) {
          params.append('tier', filterTier)
        }
        
        const response = await fetch(
          `${BACKEND_BASE}/api/admin/users?${params}`,
          {
            headers: {
              Authorization: `Bearer ${session.access_token}`,
            },
          }
        )
        
        if (response.ok) {
          const data = await response.json()
          setUsers(data.users)
          setTotal(data.total)
        }
      } catch (error) {
        console.error('Failed to fetch users:', error)
      } finally {
        setLoading(false)
      }
    }
    
    fetchUsers()
  }, [isAdmin, session, page, filterTier])

  // Fetch user transactions when expanded
  const fetchTransactions = async (userId) => {
    try {
      const response = await fetch(
        `${BACKEND_BASE}/api/admin/transactions/${userId}?limit=10`,
        {
          headers: {
            Authorization: `Bearer ${session.access_token}`,
          },
        }
      )
      
      if (response.ok) {
        const data = await response.json()
        setTransactions(data)
      }
    } catch (error) {
      console.error('Failed to fetch transactions:', error)
    }
  }

  const toggleUserDetails = (userId) => {
    if (expandedUser === userId) {
      setExpandedUser(null)
      setTransactions([])
    } else {
      setExpandedUser(userId)
      fetchTransactions(userId)
    }
  }

  const openCreditModal = (user) => {
    setCreditAdjustUser(user)
    setCreditAmount('')
    setCreditReason('')
    setShowCreditModal(true)
  }

  const handleCreditAdjust = async () => {
    if (!creditAmount || !creditReason || !creditAdjustUser) return
    
    try {
      const response = await fetch(`${BACKEND_BASE}/api/admin/credits/adjust`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${session.access_token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: creditAdjustUser.user_id,
          amount: parseInt(creditAmount),
          reason: creditReason,
        }),
      })
      
      if (response.ok) {
        // Refresh user list
        const params = new URLSearchParams({ page: page.toString(), per_page: '20' })
        if (filterTier) params.append('tier', filterTier)
        
        const refreshResponse = await fetch(
          `${BACKEND_BASE}/api/admin/users?${params}`,
          { headers: { Authorization: `Bearer ${session.access_token}` } }
        )
        
        if (refreshResponse.ok) {
          const data = await refreshResponse.json()
          setUsers(data.users)
        }
        
        setShowCreditModal(false)
        setCreditAdjustUser(null)
      }
    } catch (error) {
      console.error('Failed to adjust credits:', error)
      alert('Failed to adjust credits')
    }
  }

  const updateTier = async (userId, newTier) => {
    try {
      const response = await fetch(`${BACKEND_BASE}/api/admin/tier/update`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${session.access_token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          tier: newTier,
        }),
      })
      
      if (response.ok) {
        // Refresh users
        setUsers(users.map(u => 
          u.user_id === userId ? { ...u, tier: newTier } : u
        ))
      }
    } catch (error) {
      console.error('Failed to update tier:', error)
    }
  }

  const toggleStatus = async (userId, field, currentValue) => {
    try {
      const response = await fetch(`${BACKEND_BASE}/api/admin/status/toggle`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${session.access_token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          [field]: !currentValue,
        }),
      })
      
      if (response.ok) {
        // Refresh users
        setUsers(users.map(u => 
          u.user_id === userId ? { ...u, [field]: !currentValue } : u
        ))
      }
    } catch (error) {
      console.error('Failed to toggle status:', error)
    }
  }

  if (!isAdmin) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center' }}>
        <Shield size={48} style={{ color: '#ef4444', marginBottom: '1rem' }} />
        <h2>Access Denied</h2>
        <p>You need admin privileges to access this panel.</p>
      </div>
    )
  }

  const filteredUsers = users.filter(user => {
    if (!searchQuery) return true
    const query = searchQuery.toLowerCase()
    return (
      user.email?.toLowerCase().includes(query) ||
      user.user_id.toLowerCase().includes(query)
    )
  })

  return (
    <div style={{ padding: '1.5rem', maxWidth: '1400px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: '2rem' }}>
        <h1 style={{ fontSize: '1.8rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
          👑 Admin Panel
        </h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
          Manage users, credits, and system settings
        </p>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
          <StatsCard icon={<Users size={20} />} label="Total Users" value={stats.total_users} color="#3b82f6" />
          <StatsCard icon={<Coins size={20} />} label="Credits Issued" value={stats.total_credits_issued} color="#8b5cf6" />
          <StatsCard icon={<TrendingUp size={20} />} label="Credits Used" value={stats.total_credits_used} color="#10b981" />
          <StatsCard icon={<Crown size={20} />} label="VIP Users" value={stats.total_vips} color="#f59e0b" />
        </div>
      )}

      {/* Filters */}
      <div style={{ 
        background: 'var(--bg-card)', 
        border: '1px solid var(--border-color)',
        borderRadius: '8px',
        padding: '1rem',
        marginBottom: '1.5rem',
        display: 'flex',
        gap: '1rem',
        flexWrap: 'wrap',
        alignItems: 'center'
      }}>
        <div style={{ flex: '1 1 300px', position: 'relative' }}>
          <Search size={16} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
          <input
            type="text"
            placeholder="Search by email or user ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{
              width: '100%',
              padding: '0.6rem 0.8rem 0.6rem 2.5rem',
              background: 'var(--bg-input)',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              color: 'var(--text-primary)',
              fontSize: '0.9rem',
            }}
          />
        </div>
        
        <select
          value={filterTier}
          onChange={(e) => { setFilterTier(e.target.value); setPage(1); }}
          style={{
            padding: '0.6rem 0.8rem',
            background: 'var(--bg-input)',
            border: '1px solid var(--border-color)',
            borderRadius: '6px',
            color: 'var(--text-primary)',
            fontSize: '0.9rem',
          }}
        >
          <option value="">All Tiers</option>
          <option value="free">Free</option>
          <option value="pro">Pro</option>
          <option value="vip">VIP</option>
        </select>
      </div>

      {/* Users Table */}
      <div style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border-color)',
        borderRadius: '8px',
        overflow: 'hidden'
      }}>
        {loading ? (
          <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
            Loading users...
          </div>
        ) : filteredUsers.length === 0 ? (
          <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
            No users found
          </div>
        ) : (
          <div>
            {filteredUsers.map((user) => (
              <UserRow
                key={user.user_id}
                user={user}
                expanded={expandedUser === user.user_id}
                onToggle={() => toggleUserDetails(user.user_id)}
                onCreditAdjust={() => openCreditModal(user)}
                onTierChange={(tier) => updateTier(user.user_id, tier)}
                onToggleVIP={() => toggleStatus(user.user_id, 'is_vip', user.is_vip)}
                onToggleAdmin={() => toggleStatus(user.user_id, 'is_admin', user.is_admin)}
                transactions={transactions}
              />
            ))}
          </div>
        )}
      </div>

      {/* Pagination */}
      {total > 20 && (
        <div style={{ marginTop: '1.5rem', display: 'flex', justifyContent: 'center', gap: '0.5rem' }}>
          <button
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
            style={{
              padding: '0.5rem 1rem',
              background: 'var(--bg-input)',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              color: 'var(--text-primary)',
              cursor: page === 1 ? 'not-allowed' : 'pointer',
              opacity: page === 1 ? 0.5 : 1,
            }}
          >
            Previous
          </button>
          <span style={{ padding: '0.5rem 1rem', color: 'var(--text-secondary)' }}>
            Page {page} of {Math.ceil(total / 20)}
          </span>
          <button
            onClick={() => setPage(p => p + 1)}
            disabled={page >= Math.ceil(total / 20)}
            style={{
              padding: '0.5rem 1rem',
              background: 'var(--bg-input)',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              color: 'var(--text-primary)',
              cursor: page >= Math.ceil(total / 20) ? 'not-allowed' : 'pointer',
              opacity: page >= Math.ceil(total / 20) ? 0.5 : 1,
            }}
          >
            Next
          </button>
        </div>
      )}

      {/* Credit Adjustment Modal */}
      {showCreditModal && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0,0,0,0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
        }}>
          <div style={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border-color)',
            borderRadius: '12px',
            padding: '2rem',
            maxWidth: '500px',
            width: '90%',
          }}>
            <h3 style={{ marginBottom: '1.5rem', fontSize: '1.3rem' }}>Adjust Credits</h3>
            <p style={{ color: 'var(--text-muted)', marginBottom: '1rem', fontSize: '0.9rem' }}>
              User: {creditAdjustUser?.email}
            </p>
            
            <div style={{ marginBottom: '1rem' }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Amount (positive to add, negative to subtract)
              </label>
              <input
                type="number"
                value={creditAmount}
                onChange={(e) => setCreditAmount(e.target.value)}
                placeholder="e.g., 100 or -50"
                style={{
                  width: '100%',
                  padding: '0.6rem 0.8rem',
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  color: 'var(--text-primary)',
                  fontSize: '0.9rem',
                }}
              />
            </div>
            
            <div style={{ marginBottom: '1.5rem' }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Reason
              </label>
              <input
                type="text"
                value={creditReason}
                onChange={(e) => setCreditReason(e.target.value)}
                placeholder="e.g., Compensation for issue #123"
                style={{
                  width: '100%',
                  padding: '0.6rem 0.8rem',
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  color: 'var(--text-primary)',
                  fontSize: '0.9rem',
                }}
              />
            </div>
            
            <div style={{ display: 'flex', gap: '0.75rem' }}>
              <button
                onClick={handleCreditAdjust}
                disabled={!creditAmount || !creditReason}
                style={{
                  flex: 1,
                  padding: '0.6rem 1rem',
                  background: 'linear-gradient(135deg, #7c3aed, #a855f7)',
                  border: 'none',
                  borderRadius: '6px',
                  color: 'white',
                  fontWeight: 500,
                  cursor: (!creditAmount || !creditReason) ? 'not-allowed' : 'pointer',
                  opacity: (!creditAmount || !creditReason) ? 0.5 : 1,
                }}
              >
                Confirm
              </button>
              <button
                onClick={() => setShowCreditModal(false)}
                style={{
                  flex: 1,
                  padding: '0.6rem 1rem',
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  color: 'var(--text-secondary)',
                  cursor: 'pointer',
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function StatsCard({ icon, label, value, color }) {
  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border-color)',
      borderRadius: '8px',
      padding: '1.25rem',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
        <div style={{ color }}>{icon}</div>
        <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>{label}</span>
      </div>
      <div style={{ fontSize: '1.8rem', fontWeight: 600, color: 'var(--text-primary)' }}>
        {value.toLocaleString()}
      </div>
    </div>
  )
}

function UserRow({ user, expanded, onToggle, onCreditAdjust, onTierChange, onToggleVIP, onToggleAdmin, transactions }) {
  const tierColors = {
    free: '#6b7280',
    pro: '#8b5cf6',
    vip: '#f59e0b',
  }

  return (
    <div style={{ 
      borderBottom: '1px solid var(--border-color)',
      transition: 'background 0.15s',
    }}>
      {/* Main Row */}
      <div
        onClick={onToggle}
        style={{
          padding: '1rem',
          display: 'grid',
          gridTemplateColumns: '2fr 1fr 1fr 1fr 100px',
          gap: '1rem',
          alignItems: 'center',
          cursor: 'pointer',
          background: expanded ? 'rgba(124,58,237,0.05)' : 'transparent',
        }}
      >
        <div>
          <div style={{ fontWeight: 500, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
            {user.email || 'No email'}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
            {user.user_id.slice(0, 8)}...
          </div>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Coins size={14} style={{ color: '#a78bfa' }} />
          <span style={{ fontWeight: 600 }}>{user.balance}</span>
        </div>
        
        <div>
          <span style={{
            padding: '0.25rem 0.6rem',
            borderRadius: '4px',
            fontSize: '0.75rem',
            fontWeight: 600,
            background: `${tierColors[user.tier]}22`,
            color: tierColors[user.tier],
          }}>
            {user.tier.toUpperCase()}
          </span>
        </div>
        
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          {user.is_admin && <Shield size={16} style={{ color: '#ef4444' }} title="Admin" />}
          {user.is_vip && <Crown size={16} style={{ color: '#f59e0b' }} title="VIP" />}
        </div>
        
        <div style={{ textAlign: 'right' }}>
          {expanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </div>
      </div>

      {/* Expanded Details */}
      {expanded && (
        <div style={{ padding: '1rem', background: 'rgba(124,58,237,0.03)', borderTop: '1px solid var(--border-color)' }}>
          {/* Actions */}
          <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
            <button
              onClick={(e) => { e.stopPropagation(); onCreditAdjust(); }}
              style={{
                padding: '0.5rem 0.9rem',
                background: 'linear-gradient(135deg, #7c3aed, #a855f7)',
                border: 'none',
                borderRadius: '6px',
                color: 'white',
                fontSize: '0.85rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
              }}
            >
              <Coins size={14} />
              Adjust Credits
            </button>
            
            <select
              value={user.tier}
              onChange={(e) => { e.stopPropagation(); onTierChange(e.target.value); }}
              onClick={(e) => e.stopPropagation()}
              style={{
                padding: '0.5rem 0.9rem',
                background: 'var(--bg-input)',
                border: '1px solid var(--border-color)',
                borderRadius: '6px',
                color: 'var(--text-primary)',
                fontSize: '0.85rem',
                cursor: 'pointer',
              }}
            >
              <option value="free">Free Tier</option>
              <option value="pro">Pro Tier</option>
              <option value="vip">VIP Tier</option>
            </select>
            
            <button
              onClick={(e) => { e.stopPropagation(); onToggleVIP(); }}
              style={{
                padding: '0.5rem 0.9rem',
                background: user.is_vip ? '#f59e0b' : 'var(--bg-input)',
                border: user.is_vip ? 'none' : '1px solid var(--border-color)',
                borderRadius: '6px',
                color: user.is_vip ? 'white' : 'var(--text-primary)',
                fontSize: '0.85rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
              }}
            >
              <Crown size={14} />
              {user.is_vip ? 'Remove VIP' : 'Grant VIP'}
            </button>
            
            <button
              onClick={(e) => { e.stopPropagation(); onToggleAdmin(); }}
              style={{
                padding: '0.5rem 0.9rem',
                background: user.is_admin ? '#ef4444' : 'var(--bg-input)',
                border: user.is_admin ? 'none' : '1px solid var(--border-color)',
                borderRadius: '6px',
                color: user.is_admin ? 'white' : 'var(--text-primary)',
                fontSize: '0.85rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
              }}
            >
              <Shield size={14} />
              {user.is_admin ? 'Remove Admin' : 'Grant Admin'}
            </button>
          </div>

          {/* Stats */}
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', 
            gap: '0.75rem',
            marginBottom: '1rem'
          }}>
            <StatItem label="Lifetime Purchased" value={user.lifetime_purchased} />
            <StatItem label="Lifetime Used" value={user.lifetime_used} />
            <StatItem label="Current Balance" value={user.balance} />
            <StatItem label="Member Since" value={new Date(user.created_at).toLocaleDateString()} />
          </div>

          {/* Recent Transactions */}
          {transactions.length > 0 && (
            <div>
              <h4 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '0.75rem', color: 'var(--text-secondary)' }}>
                Recent Transactions
              </h4>
              <div style={{ fontSize: '0.8rem' }}>
                {transactions.slice(0, 5).map((tx) => (
                  <div
                    key={tx.id}
                    style={{
                      padding: '0.5rem',
                      borderBottom: '1px solid var(--border-color)',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                    }}
                  >
                    <div>
                      <span style={{ 
                        color: tx.amount > 0 ? '#10b981' : '#ef4444',
                        fontWeight: 600,
                        marginRight: '0.5rem'
                      }}>
                        {tx.amount > 0 ? '+' : ''}{tx.amount}
                      </span>
                      <span style={{ color: 'var(--text-muted)' }}>
                        {tx.description || tx.type}
                      </span>
                    </div>
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                      {new Date(tx.created_at).toLocaleDateString()}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function StatItem({ label, value }) {
  return (
    <div style={{
      padding: '0.75rem',
      background: 'var(--bg-input)',
      borderRadius: '6px',
    }}>
      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>
        {label}
      </div>
      <div style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)' }}>
        {typeof value === 'number' ? value.toLocaleString() : value}
      </div>
    </div>
  )
}
