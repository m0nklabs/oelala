// Frontend configuration
// Set the backend base URL explicitly to the LAN address used by Oelala
const BACKEND_BASE = 'http://192.168.1.2:7998'

// Global debug flag for UI logging
const DEBUG = import.meta.env?.DEV ?? false

// External service URLs
const EXTERNAL_SERVICES = {
  COMFYUI: 'http://192.168.1.2:8188/',
  TARS_AI: 'http://192.168.1.35:8001/',
  NADSCAB: 'http://192.168.1.2:7000/'
}

export { BACKEND_BASE, EXTERNAL_SERVICES }
export default BACKEND_BASE
export { DEBUG }
