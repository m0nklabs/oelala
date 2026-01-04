import React, { createContext, useContext, useEffect, useState } from 'react'
import { supabase, isAuthEnabled } from '../lib/supabase'

const AuthContext = createContext({
  user: null,
  session: null,
  loading: true,
  signInWithGoogle: async () => {},
  signInWithGithub: async () => {},
  signOut: async () => {},
  isAdult: false,
})

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [session, setSession] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!isAuthEnabled()) {
      setLoading(false)
      return
    }

    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session)
      setUser(session?.user ?? null)
      setLoading(false)
    })

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (event, session) => {
        console.log('🔐 Auth event:', event)
        setSession(session)
        setUser(session?.user ?? null)
        setLoading(false)
      }
    )

    return () => subscription.unsubscribe()
  }, [])

  const signInWithGoogle = async () => {
    if (!isAuthEnabled()) {
      console.warn('Auth not enabled')
      return
    }
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: window.location.origin,
      },
    })
    if (error) console.error('Google sign-in error:', error)
  }

  const signInWithGithub = async () => {
    if (!isAuthEnabled()) {
      console.warn('Auth not enabled')
      return
    }
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'github',
      options: {
        redirectTo: window.location.origin,
      },
    })
    if (error) console.error('GitHub sign-in error:', error)
  }

  const signOut = async () => {
    if (!isAuthEnabled()) return
    const { error } = await supabase.auth.signOut()
    if (error) console.error('Sign-out error:', error)
  }

  // Check if user has verified adult status
  // For now, all logged-in users are considered adults
  // TODO: Add proper age verification
  const isAdult = !!user

  const value = {
    user,
    session,
    loading,
    signInWithGoogle,
    signInWithGithub,
    signOut,
    isAdult,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
