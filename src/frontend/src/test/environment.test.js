import { describe, expect, it } from 'vitest'

// The hook tests spy on URL.revokeObjectURL. jsdom does not implement the blob-URL
// API and the environment mixes jsdom's Blob with Node's URL, so setup.js installs
// a deterministic stub. This test fails loudly if that stops happening, instead of
// nine lifecycle tests failing later with a confusing matcher error.
describe('test environment', () => {
  it('provides the blob-URL API the hooks under test rely on', () => {
    expect(typeof URL.createObjectURL).toBe('function')
    expect(typeof URL.revokeObjectURL).toBe('function')
  })

  it('produces a distinct object url per call', () => {
    const first = URL.createObjectURL()
    const second = URL.createObjectURL()
    expect(first).toMatch(/^blob:/)
    expect(first).not.toBe(second)
  })

  it('revokes without throwing', () => {
    expect(() => URL.revokeObjectURL('blob:test/1')).not.toThrow()
  })
})
