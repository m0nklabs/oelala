/** Tests for the media-segment resolver (pure, fetch injected). */
import { describe, expect, it, vi } from 'vitest'
import { resolveUserMedia, PROFILE_MEDIA_SEGMENTS } from '../userMedia'

const okResponse = () => ({ ok: true, blob: async () => new Blob(['x']) })
const missResponse = () => ({ ok: false })

describe('resolveUserMedia', () => {
  it('returns the first segment that has the file', async () => {
    const fetchFn = vi.fn().mockResolvedValue(okResponse())
    const found = await resolveUserMedia(fetchFn, 'frame.png')
    expect(found.segment).toBe('uploads')
    expect(fetchFn).toHaveBeenCalledTimes(1)
    expect(fetchFn).toHaveBeenCalledWith('/user/media/uploads/frame.png')
  })

  it('falls back to the next segment when uploads/ misses', async () => {
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(missResponse())   // uploads/
      .mockResolvedValueOnce(okResponse())     // images/
    const found = await resolveUserMedia(fetchFn, 'generated_000.png')
    expect(found.segment).toBe('images')
    expect(fetchFn).toHaveBeenNthCalledWith(2, '/user/media/images/generated_000.png')
  })

  it('reaches videos/ for a video source', async () => {
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(missResponse())
      .mockResolvedValueOnce(missResponse())
      .mockResolvedValueOnce(okResponse())
    expect((await resolveUserMedia(fetchFn, 'clip.mp4')).segment).toBe('videos')
  })

  it('returns null when no segment has the file', async () => {
    const fetchFn = vi.fn().mockResolvedValue(missResponse())
    expect(await resolveUserMedia(fetchFn, 'missing.png')).toBeNull()
    expect(fetchFn).toHaveBeenCalledTimes(PROFILE_MEDIA_SEGMENTS.length)
  })

  it('returns null for an empty name without calling the API', async () => {
    const fetchFn = vi.fn()
    expect(await resolveUserMedia(fetchFn, '')).toBeNull()
    expect(fetchFn).not.toHaveBeenCalled()
  })

  it('encodes names that would otherwise break the URL', async () => {
    const fetchFn = vi.fn().mockResolvedValue(okResponse())
    await resolveUserMedia(fetchFn, 'my frame #1.png')
    expect(fetchFn).toHaveBeenCalledWith('/user/media/uploads/my%20frame%20%231.png')
  })
})
