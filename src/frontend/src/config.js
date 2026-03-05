// Frontend configuration
// Detect if running on production domain or local
const isProduction = window.location.hostname === 'oelala.xyz'

// Set the backend base URL based on environment
const BACKEND_BASE = isProduction
  ? 'https://api.oelala.xyz'
  : 'http://192.168.1.2:7998'

// Storage service URL (oelala-storage Go service)
const STORAGE_BASE = isProduction
  ? 'https://storage.oelala.xyz'
  : 'http://192.168.1.2:7990'

// Global debug flag for UI logging
const DEBUG = import.meta.env?.DEV ?? false

// External service URLs
const EXTERNAL_SERVICES = {
  COMFYUI: isProduction ? 'https://comfy.oelala.xyz/' : 'http://192.168.1.2:8188/',
  TARS_AI: 'http://192.168.1.35:8001/',
  NADSCAB: 'http://192.168.1.2:7000/'
}

/**
 * Utility to get full media URL - handles both signed URLs and relative paths
 * Signed URLs from oelala-storage come as full URLs (http://...)
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
  return `${BACKEND_BASE}${finalUrl}`
}

export { BACKEND_BASE, STORAGE_BASE, EXTERNAL_SERVICES, getMediaUrl }
export default BACKEND_BASE
export { DEBUG }
