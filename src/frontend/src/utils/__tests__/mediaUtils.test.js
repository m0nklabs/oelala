/** Tests for the media-type detection helpers (pure functions). */
import { describe, expect, it } from 'vitest'
import {
  getMediaType,
  getFileExtension,
  isVideo,
  isImage,
  isAudio,
} from '../mediaUtils'

describe('getMediaType', () => {
  it('classifies video extensions', () => {
    expect(getMediaType('clip.mp4')).toBe('video')
    expect(getMediaType('clip.MOV')).toBe('video')
    expect(getMediaType('clip.mkv')).toBe('video')
  })

  it('classifies image extensions', () => {
    expect(getMediaType('shot.png')).toBe('image')
    expect(getMediaType('shot.JPEG')).toBe('image')
    expect(getMediaType('shot.webp')).toBe('image')
  })

  it('classifies audio extensions', () => {
    expect(getMediaType('song.mp3')).toBe('audio')
    expect(getMediaType('song.flac')).toBe('audio')
  })

  it('defaults unknown extensions to image', () => {
    expect(getMediaType('doc.xyz')).toBe('image')
  })

  it('handles empty and extension-less names as image', () => {
    expect(getMediaType('')).toBe('image')
    expect(getMediaType(null)).toBe('image')
    expect(getMediaType('noext')).toBe('image')
    expect(getMediaType('trailing.')).toBe('image')
  })
})

describe('getFileExtension', () => {
  it('returns the lowercase extension', () => {
    expect(getFileExtension('photo.PNG')).toBe('png')
  })

  it('returns empty for no-extension names and empty input', () => {
    expect(getFileExtension('noext')).toBe('')
    expect(getFileExtension('')).toBe('')
    expect(getFileExtension(null)).toBe('')
  })

  it('returns empty for trailing dots', () => {
    expect(getFileExtension('trailing.')).toBe('')
  })
})

describe('isVideo/isImage/isAudio agree with getMediaType', () => {
  it('isVideo', () => {
    expect(isVideo('a.mp4')).toBe(true)
    expect(isVideo('a.png')).toBe(false)
  })
  it('isImage', () => {
    expect(isImage('a.png')).toBe(true)
    expect(isImage('a.mp4')).toBe(false)
  })
  it('isAudio', () => {
    expect(isAudio('a.mp3')).toBe(true)
    expect(isAudio('a.png')).toBe(false)
  })
})
