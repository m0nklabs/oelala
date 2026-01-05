// Frontend configuration
// Detect if running on production domain or local
const isProduction = window.location.hostname === 'oelala.xyz'

// Set the backend base URL based on environment
const BACKEND_BASE = isProduction
  ? 'https://api.oelala.xyz'
  : 'http://192.168.1.2:7998'

// Storage service URL (oelala-storage Go service)
const STORAGE_BASE = isProduction
  ? 'https://storage.oelala.xyz'  // TODO: Add to tunnel when needed
  : 'http://192.168.1.2:7990'

// Global debug flag for UI logging
const DEBUG = import.meta.env?.DEV ?? false

// External service URLs
const EXTERNAL_SERVICES = {
  COMFYUI: isProduction ? 'https://comfy.oelala.xyz/' : 'http://192.168.1.2:8188/',
  TARS_AI: 'http://192.168.1.35:8001/',
  NADSCAB: 'http://192.168.1.2:7000/'
}

export { BACKEND_BASE, STORAGE_BASE, EXTERNAL_SERVICES }
export default BACKEND_BASE
export { DEBUG }
