import { useCallback, useEffect, useState } from 'react'
import { BACKEND_BASE, DEBUG } from '../config'
import { sendClientLog } from '../logging'

export function useVideoHistory(refreshToken) {
  const [videos, setVideos] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const refresh = useCallback(async () => {
    setLoading(true)
    setError('')

    try {
      if (DEBUG) console.debug('🔍 fetching history /list-videos')
      const res = await fetch(`${BACKEND_BASE}/list-videos`)
      const data = await res.json()
      if (!res.ok) throw new Error(data?.detail || `History failed (${res.status})`)
      setVideos(Array.isArray(data?.videos) ? data.videos : [])
    } catch (e) {
      const message = e?.message || 'Failed to load history'
      setError(message)
      await sendClientLog({
        level: 'error',
        message: 'History fetch failed',
        timestamp: new Date().toISOString(),
        meta: { message },
      })
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh, refreshToken])

  return { videos, loading, error, refresh }
}
