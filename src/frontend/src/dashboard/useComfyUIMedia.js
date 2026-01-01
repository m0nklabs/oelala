import { useCallback, useEffect, useState } from 'react'
import { BACKEND_BASE, DEBUG } from '../config'
import { sendClientLog } from '../logging'

export function useComfyUIMedia(refreshToken, type = 'all') {
  const [media, setMedia] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [stats, setStats] = useState({ videos: 0, images: 0 })

  const refresh = useCallback(async () => {
    setLoading(true)
    setError('')

    try {
      if (DEBUG) console.debug('🔍 fetching ComfyUI media /list-comfyui-media?type=' + type)
      const res = await fetch(`${BACKEND_BASE}/list-comfyui-media?type=${type}`)
      const data = await res.json()
      if (!res.ok) throw new Error(data?.detail || `Media fetch failed (${res.status})`)
      setMedia(Array.isArray(data?.media) ? data.media : [])
      setStats({ videos: data?.videos || 0, images: data?.images || 0 })
    } catch (e) {
      const message = e?.message || 'Failed to load media'
      setError(message)
      await sendClientLog({
        level: 'error',
        message: 'ComfyUI media fetch failed',
        timestamp: new Date().toISOString(),
        meta: { message },
      })
    } finally {
      setLoading(false)
    }
  }, [type])

  useEffect(() => {
    void refresh()
  }, [refresh, refreshToken])

  return { media, loading, error, stats, refresh }
}
