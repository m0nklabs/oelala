/**
 * Unit tests for the security-critical api.js helpers.
 *
 * The suite guards three properties the whole app relies on:
 *  1. apiFetch refuses to fetch cross-origin endpoints it has not allow-listed.
 *  2. The Supabase bearer token is attached only to backend/browser origins —
 *     never to the storage origins, which are fetchable but unauthenticated.
 *  3. The legacy helpers normalise the 402 insufficient-credits envelope and
 *     dispatch the event CreditsContext listens for.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { apiFetch, postForm, postJson } from '../api'

vi.mock('../lib/supabase', () => ({
  supabase: {
    auth: {
      getSession: vi.fn(async () => ({
        data: { session: { access_token: 'tok-123', user: { email: 'user@_.test.invalid' } } },
        error: null,
      })),
    },
  },
}))

vi.mock('../config', () => ({
  BACKEND_BASE: 'https://api.test.invalid',
  STORAGE_BASE: 'https://storage.test.invalid',
  DEBUG: false,
}))

const fetchMock = vi.fn(async () => ({
  ok: true,
  status: 200,
  text: async () => '',
  json: async () => ({}),
}))

beforeEach(() => {
  fetchMock.mockClear()
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('apiFetch — cross-origin guard', () => {
  it('rejects a non-allow-listed origin instead of fetching it', async () => {
    await expect(apiFetch('https://evil.example.com/steal')).rejects.toThrow(
      'Blocked cross-origin API request',
    )
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe('apiFetch — auth attachment', () => {
  it('attaches the bearer token on backend-origin requests', async () => {
    await apiFetch('/user/profile')
    const [, init] = fetchMock.mock.calls[0]
    expect(init.headers.Authorization).toBe('Bearer tok-123')
    expect(init.credentials).toBe('same-origin')
  })

  it('never attaches the token to storage-origin requests', async () => {
    // storage.oelala.xyz is on the fetch allow-list but not the auth list
    await apiFetch('https://storage.oelala.xyz/media/generated/x.mp4')
    const [url, init] = fetchMock.mock.calls[0]
    expect(String(url)).toContain('storage.oelala.xyz')
    expect(init.headers.Authorization).toBeUndefined()
  })

  it('sets Content-Type application/json for a plain body', async () => {
    await apiFetch('/x', { method: 'POST', body: JSON.stringify({ a: 1 }) })
    const [, init] = fetchMock.mock.calls[0]
    expect(init.headers['Content-Type']).toBe('application/json')
  })

  it('leaves Content-Type unset for FormData (browser sets the boundary)', async () => {
    await apiFetch('/x', { method: 'POST', body: new FormData() })
    const [, init] = fetchMock.mock.calls[0]
    expect(init.headers['Content-Type']).toBeUndefined()
  })

  it('adds the _cors bust param for cached static media paths', async () => {
    await apiFetch('/media/generated/oelala_t2v_x.mp4')
    expect(String(fetchMock.mock.calls[0][0])).toContain('_cors=1')
  })
})

describe('postForm — 402 insufficient credits envelope', () => {
  it('dispatches the event CreditsContext listens for and stringifies the detail', async () => {
    const events = []
    const onCredits = (e) => events.push(e.detail)
    window.addEventListener('insufficient-credits', onCredits)
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 402,
      text: async () =>
        JSON.stringify({ detail: { error: 'insufficient_credits', required: 5, available: 2, packages: [] } }),
    })

    try {
      const res = await postForm('/generate', new FormData())

      expect(events).toEqual([{ required: 5, available: 2, packages: [] }])
      expect(res.data.detail).toBe('Not enough credits. (Need 5, have 2)')
    } finally {
      // Remove the exact reference registered above — a fresh anonymous
      // function would never match, leaving the listener attached for the
      // rest of the suite (PR-Piet finding, 2026-11-22).
      window.removeEventListener('insufficient-credits', onCredits)
    }
  })

  it('passes other object details through as JSON text, not [object Object]', async () => {
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 400,
      text: async () => JSON.stringify({ detail: { error: 'bad_request', msg: 'nope' } }),
    })

    const res = await postForm('/x', new FormData())
    expect(res.data.detail).toBe('{"error":"bad_request","msg":"nope"}')
  })
})

describe('postJson — 402 handling', () => {
  it('dispatches the insufficient-credits event for postJson too', async () => {
    const events = []
    const onCredits = (e) => events.push(e.detail)
    window.addEventListener('insufficient-credits', onCredits)
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 402,
      text: async () => JSON.stringify({ detail: { error: 'insufficient_credits', required: 3, available: 1, packages: [] } }),
    })

    try {
      await postJson('/x', { a: 1 })
      expect(events).toHaveLength(1)
    } finally {
      window.removeEventListener('insufficient-credits', onCredits)
    }
  })
})
