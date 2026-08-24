// Frontend configuration
// Detect if running on production domain or local
const isProduction = window.location.hostname === 'oelala.xyz'

// Set the backend base URL based on environment
const BACKEND_BASE = isProduction
  ? 'https://api.oelala.xyz'
  : 'http://localhost:7998'

// Storage service URL (MinIO — serves presigned URLs directly)
const STORAGE_BASE = isProduction
  ? 'https://storage-main.oelala.xyz'
  : 'http://localhost:9000'

// Global debug flag for UI logging
const DEBUG = import.meta.env?.DEV ?? false

// External service URLs
const EXTERNAL_SERVICES = {
  COMFYUI: isProduction ? 'https://comfy.oelala.xyz/' : 'http://localhost:8188/',
  TARS_AI: 'http://localhost:8001/',
  NADSCAB: 'http://localhost:7000/'
}

/**
 * Utility to get full media URL - handles both signed URLs and relative paths
 * Presigned URLs from MinIO come as full URLs (http://...)
 * Legacy backend paths are relative (/files/video.mp4)
 * @param {string} url - URL or relative path
 * @param {string} [signedUrl] - Optional signed URL (preferred if available)
 * @returns {string} Full URL ready for use
 */
const getMediaUrl = (url, signedUrl = null) => {
  // Prefer signed URL if available
  const finalUrl = signedUrl || url
  if (!finalUrl) return ''
  // If it's already a full URL, use as-is
  if (finalUrl.startsWith('http://') || finalUrl.startsWith('https://')) {
    return finalUrl
  }

  // Otherwise prepend backend base
  let fullUrl = `${BACKEND_BASE}${finalUrl}`

  // Try to append token from localStorage for protected media endpoints accessed via img/video src
  try {
    const sbKey = Object.keys(localStorage).find(k => k.startsWith('sb-') && k.endsWith('-auth-token'))
    if (sbKey) {
      const data = JSON.parse(localStorage.getItem(sbKey))
      const token = data?.access_token
      if (token) {
        fullUrl += (fullUrl.includes('?') ? '&' : '?') + `token=${token}`
      }
    }
  } catch (e) {
    // Ignore localStorage access errors
  }

  return fullUrl
}

export { BACKEND_BASE, STORAGE_BASE, EXTERNAL_SERVICES, getMediaUrl }
export default BACKEND_BASE
export { DEBUG }
