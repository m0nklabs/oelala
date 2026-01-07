import React, { createContext, useContext, useState, useEffect } from 'react'
import { useAuth } from './AuthContext'

const NSFWContext = createContext({
  nsfwEnabled: false,
  setNsfwEnabled: () => {},
})

const STORAGE_KEY = 'oelala_nsfw_enabled'

export function NSFWProvider({ children }) {
  const { user } = useAuth()

  // Initialize from localStorage, default to false (SFW mode)
  const [nsfwEnabled, setNsfwEnabledInternal] = useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      return stored === 'true'
    } catch {
      return false
    }
  })

  // NSFW can only be enabled if user is logged in
  // Force disable when user logs out
  useEffect(() => {
    if (!user && nsfwEnabled) {
      setNsfwEnabledInternal(false)
    }
  }, [user, nsfwEnabled])

  // Wrapper that prevents enabling NSFW for guests
  const setNsfwEnabled = (value) => {
    if (value && !user) {
      // Can't enable NSFW without being logged in
      return
    }
    setNsfwEnabledInternal(value)
  }

  // Persist to localStorage on change (only if logged in)
  useEffect(() => {
    try {
      if (user) {
        localStorage.setItem(STORAGE_KEY, nsfwEnabled.toString())
      }
    } catch {
      // localStorage not available
    }
  }, [nsfwEnabled, user])

  // Effective NSFW state: always false for guests
  const effectiveNsfwEnabled = user ? nsfwEnabled : false

  return (
    <NSFWContext.Provider value={{ nsfwEnabled: effectiveNsfwEnabled, setNsfwEnabled }}>
      {children}
    </NSFWContext.Provider>
  )
}

export function useNSFW() {
  return useContext(NSFWContext)
}

// Helper to filter items based on NSFW state
export function filterNSFW(items, nsfwEnabled, isNsfwFn = (item) => item.nsfw) {
  if (nsfwEnabled) {
    return items // Show all when NSFW is enabled
  }
  return items.filter(item => !isNsfwFn(item))
}

export default NSFWContext
