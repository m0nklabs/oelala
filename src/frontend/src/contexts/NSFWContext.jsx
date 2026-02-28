import React, { createContext, useContext, useState, useEffect } from 'react'
import { useAuth } from './AuthContext'
import AgeVerificationModal from '../components/AgeVerificationModal'

const NSFWContext = createContext({
  nsfwEnabled: false,
  setNsfwEnabled: () => {},
  ageVerified: false,
})

const STORAGE_KEY = 'oelala_nsfw_enabled'
const AGE_VERIFIED_KEY = 'oelala_age_verified'

export function NSFWProvider({ children }) {
  const { user } = useAuth()

  // Age verification state — persisted in localStorage, cleared on logout
  const [ageVerified, setAgeVerifiedInternal] = useState(() => {
    try {
      return localStorage.getItem(AGE_VERIFIED_KEY) === 'true'
    } catch {
      return false
    }
  })

  // Whether the age verification modal is currently visible
  const [showAgeModal, setShowAgeModal] = useState(false)

  // Initialize from localStorage, default to false (SFW mode)
  const [nsfwEnabled, setNsfwEnabledInternal] = useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      return stored === 'true'
    } catch {
      return false
    }
  })

  // Force-disable NSFW and clear age verification when user logs out
  useEffect(() => {
    if (!user) {
      if (nsfwEnabled) setNsfwEnabledInternal(false)
      if (ageVerified) {
        setAgeVerifiedInternal(false)
        try { localStorage.removeItem(AGE_VERIFIED_KEY) } catch {}
      }
    }
  }, [user]) // eslint-disable-line react-hooks/exhaustive-deps

  // Wrapper: requires login + age verification before enabling NSFW
  const setNsfwEnabled = (value) => {
    if (value && !user) return // Must be logged in
    if (value && !ageVerified) {
      // Show age gate — actual enable happens in confirmAge()
      setShowAgeModal(true)
      return
    }
    setNsfwEnabledInternal(value)
  }

  // Called when user confirms they are 18+ in the modal
  const confirmAge = () => {
    try {
      localStorage.setItem(AGE_VERIFIED_KEY, 'true')
    } catch {}
    setAgeVerifiedInternal(true)
    setNsfwEnabledInternal(true)
    setShowAgeModal(false)
  }

  // Persist NSFW preference to localStorage (only when logged in)
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
    <NSFWContext.Provider value={{ nsfwEnabled: effectiveNsfwEnabled, setNsfwEnabled, ageVerified }}>
      {children}
      {showAgeModal && (
        <AgeVerificationModal
          onConfirm={confirmAge}
          onCancel={() => setShowAgeModal(false)}
        />
      )}
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
