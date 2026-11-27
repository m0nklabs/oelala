/** Tests for the generation-time estimation helpers (pure functions). */
import { describe, expect, it } from 'vitest'
import {
  estimateI2VTime,
  estimateT2VTime,
  formatTime,
  estimateQueueWait,
} from '../timeEstimates'

describe('estimateI2VTime', () => {
  it('uses the 480p base for a standard 6s job', () => {
    const e = estimateI2VTime({ resolution: '480p', duration: 6, steps: 6 })
    expect(e.seconds).toBe(90)
  })

  it('scales by the resolution multiplier', () => {
    const base = estimateI2VTime({ resolution: '480p', duration: 6, steps: 6 })
    const hd = estimateI2VTime({ resolution: '720p', duration: 6, steps: 6 })
    expect(hd.seconds).toBe(base.seconds * 2)
    const fhd = estimateI2VTime({ resolution: '1080p', duration: 6, steps: 6 })
    expect(fhd.seconds).toBe(base.seconds * 4)
  })

  it('adds 15s per extra video second beyond 6', () => {
    const base = estimateI2VTime({ resolution: '480p', duration: 6 })
    const longer = estimateI2VTime({ resolution: '480p', duration: 8 })
    expect(longer.seconds).toBe(base.seconds + 2 * 15)
  })

  it('multiplies by steps/6 when steps exceed 6', () => {
    const base = estimateI2VTime({ resolution: '480p', duration: 6 })
    const slow = estimateI2VTime({ resolution: '480p', duration: 6, steps: 12 })
    expect(slow.seconds).toBe(Math.round(base.seconds * 2))
  })

  it('returns a range of 80%-130% and a formatted string', () => {
    const e = estimateI2VTime({})
    expect(e.min).toBe(Math.round(e.seconds * 0.8))
    expect(e.max).toBe(Math.round(e.seconds * 1.3))
    expect(e.formatted).toMatch(/s$|min$|m /)
    expect(e.range).toContain(' - ')
  })

  it('falls back to 1.0 for unknown resolutions', () => {
    const base = estimateI2VTime({ resolution: '480p', duration: 6 })
    const weird = estimateI2VTime({ resolution: '1440p', duration: 6 })
    expect(weird.seconds).toBe(base.seconds)
  })
})

describe('estimateT2VTime', () => {
  it('adds the T2I step time on top of the I2V estimate', () => {
    const i2v = estimateI2VTime({ resolution: '480p', duration: 41 / 16, steps: 6 })
    const t2v = estimateT2VTime({ resolution: '480p', numFrames: 41, steps: 6, t2iSteps: 20 })
    expect(t2v.seconds).toBe(i2v.seconds + 30)
    expect(t2v.min).toBe(i2v.min + 24)
    expect(t2v.max).toBe(i2v.max + 36)
  })
})

describe('formatTime', () => {
  it('formats seconds under a minute', () => {
    expect(formatTime(45)).toBe('45s')
  })

  it('formats whole minutes', () => {
    expect(formatTime(120)).toBe('2 min')
  })

  it('formats minutes plus seconds', () => {
    expect(formatTime(95)).toBe('1m 35s')
  })
})

describe('estimateQueueWait', () => {
  it('multiplies position by the average job time', () => {
    const w = estimateQueueWait(3, 120)
    expect(w.seconds).toBe(360)
    expect(w.formatted).toBe('6 min')
    expect(w.position).toBe(4)
  })

  it('uses 120s as the default job time', () => {
    expect(estimateQueueWait(1).seconds).toBe(120)
  })
})
