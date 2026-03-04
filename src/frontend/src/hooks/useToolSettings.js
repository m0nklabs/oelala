/**
 * useToolSettings — Lightweight localStorage-based settings persistence per tool.
 *
 * Remembers last-used settings automatically.  User presses "Reset to defaults"
 * to go back to OOTB defaults.
 *
 * Usage:
 *   const DEFAULTS = { prompt: '', steps: 20, cfg: 7.0 }
 *   const { initial, save, resetDefaults } = useToolSettings('text_to_video', DEFAULTS)
 *
 *   // Use `initial` as useState defaults:
 *   const [prompt, setPrompt] = useState(initial.prompt)
 *   const [steps, setSteps] = useState(initial.steps)
 *
 *   // Auto-save via snapshot:
 *   const snapshot = useMemo(() => ({ prompt, steps }), [prompt, steps])
 *   useEffect(() => { save(snapshot) }, [snapshot, save])
 *
 *   // Reset button:
 *   <ResetDefaultsButton onReset={() => { const d = resetDefaults(); setPrompt(d.prompt); ... }} />
 */

import { useMemo, useRef, useCallback } from 'react'

const STORAGE_PREFIX = 'oelala_settings_'
const DEBOUNCE_MS = 600

/**
 * @param {string} toolName  — e.g. 'text_to_video'
 * @param {Object} defaults  — full default settings object
 * @returns {{ initial: Object, save: Function, resetDefaults: Function }}
 */
export function useToolSettings(toolName, defaults) {
  const key = `${STORAGE_PREFIX}${toolName}`

  // Load from localStorage on first render — merged with defaults so new
  // fields added later are automatically picked up.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const initial = useMemo(() => {
    try {
      const raw = localStorage.getItem(key)
      if (raw) {
        const saved = JSON.parse(raw)
        return { ...defaults, ...saved }
      }
    } catch { /* corrupt data — ignore */ }
    return { ...defaults }
  }, []) // intentionally empty — only on mount

  const debounceRef = useRef(null)

  /** Debounced save to localStorage. Call on every settings change. */
  const save = useCallback((snapshot) => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      try {
        localStorage.setItem(key, JSON.stringify(snapshot))
      } catch { /* quota exceeded — silently ignore */ }
    }, DEBOUNCE_MS)
  }, [key])

  /** Clear saved data and return the original defaults. */
  const resetDefaults = useCallback(() => {
    try { localStorage.removeItem(key) } catch {}
    return { ...defaults }
  }, [key, defaults])

  return { initial, save, resetDefaults }
}
