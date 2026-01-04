/**
 * API helpers for Oelala frontend
 * Includes authenticated requests using Supabase JWT
 */

import { BACKEND_BASE, DEBUG } from './config'
import { supabase } from './lib/supabase'

/**
 * Get the current session's access token
 */
async function getAccessToken() {
  try {
    if (!supabase) {
      console.log('🔐 API: supabase client not available')
      return null
    }
    const { data: { session }, error } = await supabase.auth.getSession()
    if (error) {
      console.error('🔐 API: getSession error:', error)
      return null
    }
    if (session?.access_token) {
      console.log('🔐 API: Got access token for user:', session.user?.email)
    } else {
      console.log('🔐 API: No active session')
    }
    return session?.access_token || null
  } catch (e) {
    console.error('🔐 API: getAccessToken exception:', e)
    return null
  }
}

/**
 * Make an authenticated API request
 * @param {string} endpoint - API endpoint (e.g., '/user/media')
 * @param {RequestInit} options - Fetch options
 * @returns {Promise<Response>}
 */
export async function apiFetch(endpoint, options = {}) {
  const token = await getAccessToken()
  
  const headers = {
    ...options.headers,
  }
  
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  
  // Don't set Content-Type for FormData (browser sets it with boundary)
  if (!(options.body instanceof FormData) && !headers['Content-Type']) {
    if (options.body) {
      headers['Content-Type'] = 'application/json'
    }
  }
  
  const url = endpoint.startsWith('http') ? endpoint : `${BACKEND_BASE}${endpoint}`
  
  if (DEBUG) {
    console.log(`🔐 API: ${options.method || 'GET'} ${endpoint}`, token ? '(authenticated)' : '(anonymous)')
  }
  
  return fetch(url, {
    ...options,
    headers,
    credentials: 'same-origin',
  })
}

// ============================================================================
// Legacy helpers (kept for backwards compatibility)
// ============================================================================

// Lightweight API helper with graceful JSON parsing fallback
export async function postForm(url, formData, headers = {}) {
  const token = await getAccessToken()
  const authHeaders = token ? { ...headers, 'Authorization': `Bearer ${token}` } : headers
  
  const res = await fetch(url, {
    method: 'POST',
    body: formData,
    headers: authHeaders,
    credentials: 'same-origin',
  })

  const text = await res.text()
  try {
    const data = text ? JSON.parse(text) : null
    return { ok: res.ok, status: res.status, data }
  } catch (e) {
    // Fallback: return raw text when JSON parsing fails
    return { ok: res.ok, status: res.status, data: text }
  }
}

export async function getJson(url) {
  const token = await getAccessToken()
  const headers = token ? { 'Authorization': `Bearer ${token}` } : {}
  
  const res = await fetch(url, { method: 'GET', headers, credentials: 'same-origin' })
  const text = await res.text()
  try {
    const data = text ? JSON.parse(text) : null
    return { ok: res.ok, status: res.status, data }
  } catch (e) {
    return { ok: res.ok, status: res.status, data: text }
  }
}

export async function postJson(url, body = {}) {
  const token = await getAccessToken()
  const headers = { 'Content-Type': 'application/json' }
  if (token) headers['Authorization'] = `Bearer ${token}`
  
  const res = await fetch(url, {
    method: 'POST',
    body: JSON.stringify(body),
    headers,
    credentials: 'same-origin',
  })
  const text = await res.text()
  try {
    const data = text ? JSON.parse(text) : null
    return { ok: res.ok, status: res.status, data }
  } catch (e) {
    return { ok: res.ok, status: res.status, data: text }
  }
}

// ============================================================================
// New authenticated API helpers
// ============================================================================

/**
 * GET request with auth
 */
export async function apiGet(endpoint) {
  const response = await apiFetch(endpoint, { method: 'GET' })
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`)
  }
  return response.json()
}

/**
 * POST request with auth
 */
export async function apiPost(endpoint, data) {
  const body = data instanceof FormData ? data : JSON.stringify(data)
  const response = await apiFetch(endpoint, {
    method: 'POST',
    body,
  })
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`)
  }
  return response.json()
}

/**
 * DELETE request with auth
 */
export async function apiDelete(endpoint) {
  const response = await apiFetch(endpoint, { method: 'DELETE' })
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`)
  }
  return response.json()
}

// ============================================================================
// User Media API helpers
// ============================================================================

/**
 * List user's media files from storage
 * @param {string} type - 'all', 'video', 'image', 'audio'
 */
export async function listUserMedia(type = 'all') {
  return apiGet(`/user/media?type=${type}`)
}

/**
 * Build authenticated URL for user media
 * Note: For direct access, use the proxy endpoint which checks auth
 */
export function getUserMediaUrl(mediaType, filename) {
  return `${BACKEND_BASE}/user/media/${mediaType}/${encodeURIComponent(filename)}`
}

/**
 * Upload media to user's storage
 * @param {string} mediaType - 'images', 'videos', 'audio'
 * @param {File} file - File to upload
 * @param {function} onProgress - Progress callback (0-100)
 */
export async function uploadUserMedia(mediaType, file, onProgress = null) {
  const token = await getAccessToken()
  
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    const formData = new FormData()
    formData.append('file', file)
    
    xhr.open('POST', `${BACKEND_BASE}/user/media/${mediaType}`)
    
    if (token) {
      xhr.setRequestHeader('Authorization', `Bearer ${token}`)
    }
    
    if (onProgress) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          onProgress(Math.round((e.loaded / e.total) * 100))
        }
      }
    }
    
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText))
        } catch {
          resolve({ success: true })
        }
      } else {
        reject(new Error(`Upload failed: ${xhr.status}`))
      }
    }
    
    xhr.onerror = () => reject(new Error('Upload failed'))
    xhr.send(formData)
  })
}

/**
 * Delete user's media file
 */
export async function deleteUserMedia(mediaType, filename) {
  return apiDelete(`/user/media/${mediaType}/${encodeURIComponent(filename)}`)
}

/**
 * Get user profile
 */
export async function getUserProfile() {
  return apiGet('/user/profile')
}
