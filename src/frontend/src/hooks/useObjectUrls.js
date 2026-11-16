import { useEffect, useRef } from 'react'

/**
 * Revoke a single object URL when it is replaced or the component unmounts.
 *
 * An object URL keeps its whole blob alive until it is revoked, and a source video
 * is large, so a preview that is swapped out must not stay in memory.
 *
 * @param {string} url
 */
export function useObjectUrl(url) {
  useEffect(() => {
    if (!url) return undefined
    return () => URL.revokeObjectURL(url)
  }, [url])
}

/**
 * Revoke every object URL that leaves `urls`, and all of them on unmount.
 *
 * `urls` may be a fresh array on every render: the set comparison makes the effect
 * idempotent and only dropped URLs are revoked.
 *
 * @param {string[]} urls
 */
export function useTrackedObjectUrls(urls) {
  const live = useRef(new Set())

  useEffect(() => {
    const active = new Set(urls)
    for (const url of live.current) {
      if (!active.has(url)) URL.revokeObjectURL(url)
    }
    live.current = active
  }, [urls])

  useEffect(
    () => () => {
      // Read .current at cleanup time. The effect above replaces the Set on every
      // change, so a reference captured here would still point at the initial empty
      // one and revoke nothing.
      for (const url of live.current) URL.revokeObjectURL(url)
    },
    []
  )
}
