/**
 * useToolProfile — Auto-save/load settings per tool per user.
 *
 * Usage:
 *   const { settings, updateSettings, profiles, ... } = useToolProfile('image_to_video', DEFAULT_SETTINGS)
 *
 * - On mount: loads active profile from backend → merges with defaults → calls onLoad(settings)
 * - On change: debounced auto-save (800ms) to backend
 * - Supports named profiles (save as, switch, delete)
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import { apiFetch } from '../api'
import { useAuth } from '../contexts/AuthContext'

const DEBOUNCE_MS = 800

/**
 * @param {string} toolName — e.g. 'image_to_video'
 * @param {Object} defaultSettings — full default settings object (used as fallback)
 * @param {Object} options
 * @param {Function} options.onLoad — called with loaded settings after merge with defaults
 * @returns {{ settings, updateSettings, saveAs, profiles, switchProfile, deleteProfile, loaded, saving, activeProfile }}
 */
export function useToolProfile(toolName, defaultSettings = {}, options = {}) {
  const { user } = useAuth()
  const [settings, setSettings] = useState(defaultSettings)
  const [profiles, setProfiles] = useState([])
  const [presets, setPresets] = useState([])
  const [activeProfile, setActiveProfile] = useState('default')
  const [loaded, setLoaded] = useState(false)
  const [saving, setSaving] = useState(false)

  const debounceRef = useRef(null)
  const settingsRef = useRef(settings)
  const skipAutoSave = useRef(false)

  settingsRef.current = settings

  // ── Load active profile on mount ──────────────────────────────────────
  useEffect(() => {
    if (!user) {
      // Not logged in → use defaults, mark loaded
      setSettings(defaultSettings)
      setLoaded(true)
      return
    }

    let cancelled = false

    async function load() {
      try {
        const resp = await apiFetch(`/api/settings/${toolName}`)
        if (cancelled) return

        if (resp.ok) {
          const profile = await resp.json()
          if (profile && profile.settings) {
            // Merge: saved settings override defaults, but new defaults fill gaps
            const merged = { ...defaultSettings, ...profile.settings }
            skipAutoSave.current = true
            setSettings(merged)
            setActiveProfile(profile.profile_name || 'default')
            options.onLoad?.(merged)
          } else {
            // No profile yet → use defaults
            setSettings(defaultSettings)
            options.onLoad?.(defaultSettings)
          }
        } else if (resp.status === 404 || resp.status === 204) {
          // No profile exists → use defaults
          setSettings(defaultSettings)
          options.onLoad?.(defaultSettings)
        } else {
          console.warn(`⚙️ Profile load failed (${resp.status}), using defaults`)
          setSettings(defaultSettings)
        }
      } catch (err) {
        console.error('⚙️ Profile load error:', err)
        setSettings(defaultSettings)
      } finally {
        if (!cancelled) {
          setLoaded(true)
          // Reset skip flag after a tick so the initial setSettings doesn't trigger auto-save
          setTimeout(() => { skipAutoSave.current = false }, 50)
        }
      }
    }

    load()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id, toolName])

  // ── Auto-save on settings change (debounced) ─────────────────────────
  useEffect(() => {
    if (!user || !loaded || skipAutoSave.current) return

    // Clear previous debounce
    if (debounceRef.current) clearTimeout(debounceRef.current)

    debounceRef.current = setTimeout(async () => {
      try {
        setSaving(true)
        await apiFetch(`/api/settings/${toolName}`, {
          method: 'PUT',
          body: JSON.stringify({ settings: settingsRef.current }),
        })
      } catch (err) {
        console.error('⚙️ Auto-save failed:', err)
      } finally {
        setSaving(false)
      }
    }, DEBOUNCE_MS)

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings, user?.id, loaded])

  // ── Update settings (individual fields) ───────────────────────────────
  const updateSettings = useCallback((updates) => {
    setSettings(prev => ({ ...prev, ...updates }))
  }, [])

  // ── Save as named profile ─────────────────────────────────────────────
  const saveAs = useCallback(async (profileName) => {
    if (!user) return null
    try {
      const resp = await apiFetch(`/api/settings/${toolName}/profiles`, {
        method: 'POST',
        body: JSON.stringify({
          profile_name: profileName,
          settings: settingsRef.current,
        }),
      })
      if (resp.ok) {
        const profile = await resp.json()
        await loadProfiles()
        return profile
      }
      const err = await resp.json().catch(() => ({}))
      throw new Error(err.detail || 'Failed to save profile')
    } catch (err) {
      console.error('⚙️ Save as failed:', err)
      throw err
    }
  }, [user, toolName])

  // ── Load profiles list ────────────────────────────────────────────────
  const loadProfiles = useCallback(async () => {
    if (!user) return
    try {
      const resp = await apiFetch(`/api/settings/${toolName}/profiles`)
      if (resp.ok) {
        const data = await resp.json()
        setProfiles(data.profiles || [])
        if (data.active_profile) setActiveProfile(data.active_profile)
      }
    } catch (err) {
      console.error('⚙️ Load profiles failed:', err)
    }
  }, [user, toolName])

  // ── Switch profile ────────────────────────────────────────────────────
  const switchProfile = useCallback(async (profileName) => {
    if (!user) return
    try {
      const resp = await apiFetch(`/api/settings/${toolName}/profiles/${encodeURIComponent(profileName)}/activate`, {
        method: 'PUT',
      })
      if (resp.ok) {
        const profile = await resp.json()
        skipAutoSave.current = true
        setSettings({ ...defaultSettings, ...profile.settings })
        setActiveProfile(profile.profile_name)
        options.onLoad?.({ ...defaultSettings, ...profile.settings })
        setTimeout(() => { skipAutoSave.current = false }, 50)
      }
    } catch (err) {
      console.error('⚙️ Switch profile failed:', err)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user, toolName, defaultSettings])

  // ── Delete profile ────────────────────────────────────────────────────
  const deleteProfile = useCallback(async (profileName) => {
    if (!user) return
    try {
      const resp = await apiFetch(`/api/settings/${toolName}/profiles/${encodeURIComponent(profileName)}`, {
        method: 'DELETE',
      })
      if (resp.ok) {
        await loadProfiles()
        // If deleted the active one, switch to default
        if (profileName === activeProfile) {
          await switchProfile('default')
        }
      }
    } catch (err) {
      console.error('⚙️ Delete profile failed:', err)
    }
  }, [user, toolName, activeProfile, loadProfiles, switchProfile])

  // ── Load factory presets (no auth needed) ─────────────────────────────
  const loadPresets = useCallback(async () => {
    try {
      const resp = await apiFetch(`/api/settings/${toolName}/presets`)
      if (resp.ok) {
        const data = await resp.json()
        setPresets(data.presets || [])
      }
    } catch (err) {
      console.error('⚙️ Load presets failed:', err)
    }
  }, [toolName])

  // Load presets on mount (no auth needed)
  useEffect(() => {
    loadPresets()
  }, [loadPresets])

  // ── Apply a factory preset ────────────────────────────────────────────
  const applyPreset = useCallback((preset) => {
    if (!preset?.settings) return
    const merged = { ...defaultSettings, ...preset.settings }
    skipAutoSave.current = true
    setSettings(merged)
    options.onLoad?.(merged)
    // Allow auto-save after a tick so the preset gets persisted as current settings
    setTimeout(() => { skipAutoSave.current = false }, 50)
    // Trigger a save after applying
    setTimeout(() => {
      setSettings(prev => ({ ...prev }))  // Force re-render to trigger auto-save
    }, 100)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defaultSettings])

  return {
    settings,
    updateSettings,
    saveAs,
    profiles,
    presets,
    loadProfiles,
    loadPresets,
    switchProfile,
    deleteProfile,
    applyPreset,
    loaded,
    saving,
    activeProfile,
  }
}
