import React from 'react'
import { useAuth } from '../contexts/AuthContext'
import { LogIn, User, LogOut, Loader2 } from 'lucide-react'

export default function UserMenu() {
  const { user, loading, signInWithGoogle, signOut } = useAuth()

  if (loading) {
    return (
      <div className="user-menu loading">
        <Loader2 size={16} className="spin" />
      </div>
    )
  }

  if (!user) {
    return (
      <button 
        className="login-btn"
        onClick={signInWithGoogle}
        title="Sign in with Google"
      >
        <LogIn size={16} />
        <span>Login</span>
      </button>
    )
  }

  return (
    <div className="user-menu">
      <div className="user-info" title={user.email}>
        {user.user_metadata?.avatar_url ? (
          <img 
            src={user.user_metadata.avatar_url} 
            alt="Avatar" 
            className="user-avatar"
          />
        ) : (
          <User size={16} />
        )}
        <span className="user-name">
          {user.user_metadata?.full_name || user.email?.split('@')[0]}
        </span>
      </div>
      <button 
        className="logout-btn"
        onClick={signOut}
        title="Sign out"
      >
        <LogOut size={14} />
      </button>
    </div>
  )
}
