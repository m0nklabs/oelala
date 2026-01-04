import React, { createContext, useContext, useState, useEffect } from 'react'

const NSFWContext = createContext({
  nsfwEnabled: false,
  setNsfwEnabled: () => {},
})

const STORAGE_KEY = 'oelala_nsfw_enabled'

export function NSFWProvider({ children }) {
  // Initialize from localStorage, default to false (SFW mode)
  const [nsfwEnabled, setNsfwEnabled] = useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      return stored === 'true'
    } catch {
      return false
    }
  })

  // Persist to localStorage on change
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, nsfwEnabled.toString())
    } catch {
      // localStorage not available
    }
  }, [nsfwEnabled])

  return (
    <NSFWContext.Provider value={{ nsfwEnabled, setNsfwEnabled }}>
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
