import { describe, expect, it } from 'vitest'

import { NAV_GROUPS, TOOL_IDS } from '../nav'

describe('navigation', () => {
  it('has one nav entry per tool id and no duplicates', () => {
    const ids = NAV_GROUPS.flatMap((group) => group.items.map((item) => item.id))
    expect(new Set(ids).size).toBe(ids.length)
    expect(new Set(ids)).toEqual(new Set(Object.values(TOOL_IDS)))
  })

  it('uses real tool ids, so a typo cannot create a dead nav entry', () => {
    const known = new Set(Object.values(TOOL_IDS))
    for (const group of NAV_GROUPS) {
      for (const item of group.items) {
        expect(known.has(item.id), `${group.id}/${item.id}`).toBe(true)
        expect(item.label).toBeTruthy()
      }
    }
  })

  it('gives every group an id and a title', () => {
    for (const group of NAV_GROUPS) {
      expect(group.id).toBeTruthy()
      expect(group.title).toBeTruthy()
      expect(group.items.length).toBeGreaterThan(0)
    }
  })
})
