/** Tests for the object-URL lifecycle hooks (memory-leak sensitive). */
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useObjectUrl, useTrackedObjectUrls } from '../useObjectUrls'

const revoked = []
beforeEach(() => {
  revoked.length = 0
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(u => revoked.push(u))
})
afterEach(() => {
  vi.restoreAllMocks()
})

describe('useObjectUrl', () => {
  it('revokes the url when it is replaced', () => {
    const { rerender } = renderHook(({ url }) => useObjectUrl(url), {
      initialProps: { url: 'blob:a' },
    })
    rerender({ url: 'blob:b' })
    expect(revoked).toEqual(['blob:a'])
  })

  it('revokes on unmount', () => {
    const { unmount } = renderHook(({ url }) => useObjectUrl(url), {
      initialProps: { url: 'blob:x' },
    })
    unmount()
    expect(revoked).toEqual(['blob:x'])
  })

  it('does nothing for falsy urls', () => {
    const { rerender, unmount } = renderHook(({ url }) => useObjectUrl(url), {
      initialProps: { url: null },
    })
    rerender({ url: undefined })
    unmount()
    expect(revoked).toEqual([])
  })

  it('does not revoke while the url stays the same', () => {
    const { rerender } = renderHook(({ url }) => useObjectUrl(url), {
      initialProps: { url: 'blob:same' },
    })
    rerender({ url: 'blob:same' })
    expect(revoked).toEqual([])
  })
})

describe('useTrackedObjectUrls', () => {
  it('revokes only dropped urls when the array changes', () => {
    const { rerender } = renderHook(({ urls }) => useTrackedObjectUrls(urls), {
      initialProps: { urls: ['blob:a', 'blob:b'] },
    })
    rerender({ urls: ['blob:b', 'blob:c'] })
    expect(revoked).toEqual(['blob:a'])
  })

  it('tolerates a fresh array with identical contents', () => {
    const { rerender } = renderHook(({ urls }) => useTrackedObjectUrls(urls), {
      initialProps: { urls: ['blob:a'] },
    })
    rerender({ urls: ['blob:a'] })
    expect(revoked).toEqual([])
  })

  it('revokes everything on unmount', () => {
    const { unmount } = renderHook(({ urls }) => useTrackedObjectUrls(urls), {
      initialProps: { urls: ['blob:a', 'blob:b'] },
    })
    unmount()
    expect(revoked).toEqual(['blob:a', 'blob:b'])
  })

  it('handles an empty list', () => {
    const { rerender, unmount } = renderHook(({ urls }) => useTrackedObjectUrls(urls), {
      initialProps: { urls: [] },
    })
    rerender({ urls: ['blob:a'] })
    unmount()
    expect(revoked).toEqual(['blob:a'])
  })
})
