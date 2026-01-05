/**
 * Media utility functions
 * Shared helpers for media type detection and handling
 */

/**
 * Determine media type from filename extension
 * @param {string} filename - The filename to analyze
 * @returns {string} - Media type: 'video', 'image', or 'audio'
 */
export function getMediaType(filename) {
  if (!filename) return 'image'

  const ext = filename.toLowerCase().split('.').pop()

  // Handle edge case where filename has no extension or ends with a dot
  if (!ext || ext === filename.toLowerCase()) {
    return 'image'
  }

  if (['mp4', 'webm', 'mov', 'avi', 'mkv', 'flv'].includes(ext)) {
    return 'video'
  }

  if (['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'svg'].includes(ext)) {
    return 'image'
  }

  if (['mp3', 'wav', 'ogg', 'flac', 'aac', 'm4a'].includes(ext)) {
    return 'audio'
  }

  // Default to image for unknown types
  return 'image'
}

/**
 * Get file extension from filename
 * @param {string} filename - The filename
 * @returns {string} - File extension in lowercase
 */
export function getFileExtension(filename) {
  if (!filename) return ''

  const lowerFilename = filename.toLowerCase()
  const ext = lowerFilename.split('.').pop()

  // Handle edge case where filename has no extension or ends with a dot
  if (!ext || ext === lowerFilename) {
    return ''
  }

  return ext
}

/**
 * Check if file is a video
 * @param {string} filename - The filename to check
 * @returns {boolean}
 */
export function isVideo(filename) {
  return getMediaType(filename) === 'video'
}

/**
 * Check if file is an image
 * @param {string} filename - The filename to check
 * @returns {boolean}
 */
export function isImage(filename) {
  return getMediaType(filename) === 'image'
}

/**
 * Check if file is audio
 * @param {string} filename - The filename to check
 * @returns {boolean}
 */
export function isAudio(filename) {
  return getMediaType(filename) === 'audio'
}
