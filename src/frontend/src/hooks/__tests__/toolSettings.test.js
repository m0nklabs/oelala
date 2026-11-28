/** Tests for the localStorage-backed per-tool settings hook. */
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useToolSettings } from '../useToolSettings'

const DEFAULTS = { prompt: '', steps: 20, cfg: 7.0 }

beforeEach(() => {
  localStorage.clear()
  vi.useFakeTimers()
})
afterEach(() => {
  vi.useRealTimers()
})

describe('useToolSettings', () => {
  it('starts from the defaults when nothing is stored', () => {
    const { result } = renderHook(() => useToolSettings('t2v', DEFAULTS))
    expect(result.current.initial).toEqual(DEFAULTS)
  })

  it('merges stored settings over the defaults, so new fields appear', () => {
    localStorage.setItem('oelala_settings_t2v', JSON.stringify({ steps: 30 }))
    const { result } = renderHook(() => useToolSettings('t2v', DEFAULTS))
    expect(result.current.initial).toEqual({ prompt: '', steps: 30, cfg: 7.0 })
  })

  it('save persists the snapshot (debounced) under the tool key', async () => {
    const { result } = renderHook(() => useToolSettings('t2v', DEFAULTS))
    act(() => result.current.save({ prompt: 'hi', steps: 8, cfg: 3 }))
    expect(localStorage.getItem('oelala_settings_t2v')).toBeNull()   // debounce holds
    act(() => { vi.advanceTimersByTime(700) })
    expect(JSON.parse(localStorage.getItem('oelala_settings_t2v'))).toEqual({
      prompt: 'hi', steps: 8, cfg: 3,
    })
  })

  it('resetDefaults clears the stored key and returns the defaults', () => {
    localStorage.setItem('oelala_settings_t2v', JSON.stringify({ steps: 30 }))
    const { result } = renderHook(() => useToolSettings('t2v', DEFAULTS))
    let d
    act(() => { d = result.current.resetDefaults() })
    expect(localStorage.getItem('oelala_settings_t2v')).toBeNull()
    expect(d).toEqual(DEFAULTS)
  })

  it('survives corrupt stored JSON by falling back to defaults', () => {
    localStorage.setItem('oelala_settings_t2v', '{not json')
    const { result } = renderHook(() => useToolSettings('t2v', DEFAULTS))
    expect(result.current.initial).toEqual(DEFAULTS)
  })

  it('keys are isolated per tool', () => {
    localStorage.setItem('oelala_settings_i2v', JSON.stringify({ steps: 99 }))
    const { result } = renderHook(() => useToolSettings('t2v', DEFAULTS))
    expect(result.current.initial.steps).toBe(20)
  })
})
