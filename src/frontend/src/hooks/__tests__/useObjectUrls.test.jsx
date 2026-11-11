import { renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useObjectUrl, useTrackedObjectUrls } from '../useObjectUrls'

// An object URL pins its blob in memory until it is revoked, so every path that
// drops a preview has to revoke it. These are the three bugs that reviewers found
// by eye in #244; now they fail a test instead.
describe('useObjectUrl', () => {
  beforeEach(() => {
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {})
  })
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('revokes the url on unmount', () => {
    const { unmount } = renderHook(() => useObjectUrl('blob:video'))
    expect(URL.revokeObjectURL).not.toHaveBeenCalled()
    unmount()
    expect(URL.revokeObjectURL).toHaveBeenCalledExactlyOnceWith('blob:video')
  })

  it('revokes the previous url when it is replaced, not the new one', () => {
    const { rerender, unmount } = renderHook(({ url }) => useObjectUrl(url), {
      initialProps: { url: 'blob:first' },
    })
    rerender({ url: 'blob:second' })

    expect(URL.revokeObjectURL).toHaveBeenCalledTimes(1)
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:first')

    unmount()
    expect(URL.revokeObjectURL).toHaveBeenCalledTimes(2)
    expect(URL.revokeObjectURL).toHaveBeenLastCalledWith('blob:second')
  })

  it('does not revoke when there is no url', () => {
    const { unmount, rerender } = renderHook(({ url }) => useObjectUrl(url), {
      initialProps: { url: '' },
    })
    rerender({ url: '' })
    unmount()
    expect(URL.revokeObjectURL).not.toHaveBeenCalled()
  })

  it('does not revoke the same url twice on rerender', () => {
    const { rerender } = renderHook(({ url }) => useObjectUrl(url), {
      initialProps: { url: 'blob:same' },
    })
    rerender({ url: 'blob:same' })
    expect(URL.revokeObjectURL).not.toHaveBeenCalled()
  })
})

describe('useTrackedObjectUrls', () => {
  beforeEach(() => {
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {})
  })
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('revokes only the urls that leave the list', () => {
    const { rerender } = renderHook(({ urls }) => useTrackedObjectUrls(urls), {
      initialProps: { urls: ['blob:a', 'blob:b'] },
    })
    rerender({ urls: ['blob:b', 'blob:c'] })

    expect(URL.revokeObjectURL).toHaveBeenCalledTimes(1)
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:a')
  })

  it('revokes nothing while the list only grows', () => {
    const { rerender } = renderHook(({ urls }) => useTrackedObjectUrls(urls), {
      initialProps: { urls: ['blob:a'] },
    })
    rerender({ urls: ['blob:a', 'blob:b'] })
    expect(URL.revokeObjectURL).not.toHaveBeenCalled()
  })

  it('revokes the urls that are still live on unmount, including later ones', () => {
    // The bug this catches: reading a captured Set at cleanup time instead of
    // `ref.current` revokes nothing, because the captured set is the initial
    // empty one.
    const { rerender, unmount } = renderHook(({ urls }) => useTrackedObjectUrls(urls), {
      initialProps: { urls: [] },
    })
    rerender({ urls: ['blob:late'] })
    unmount()

    expect(URL.revokeObjectURL).toHaveBeenCalledExactlyOnceWith('blob:late')
  })

  it('accepts a fresh array with the same contents without revoking', () => {
    const { rerender } = renderHook(({ urls }) => useTrackedObjectUrls(urls), {
      initialProps: { urls: ['blob:a'] },
    })
    rerender({ urls: ['blob:a'] })
    expect(URL.revokeObjectURL).not.toHaveBeenCalled()
  })

  it('survives an empty list', () => {
    const { unmount } = renderHook(() => useTrackedObjectUrls([]))
    unmount()
    expect(URL.revokeObjectURL).not.toHaveBeenCalled()
  })
})
