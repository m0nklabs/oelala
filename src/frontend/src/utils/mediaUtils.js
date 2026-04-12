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

/**
 * Extract the first frame from a video blob URL using <video> + <canvas>.
 * Fetches the video via apiFetch, creates an object URL, draws frame 0 to canvas.
 * @param {Function} apiFetchFn - The apiFetch function for authenticated fetches
 * @param {string} fetchUrl - URL to fetch the video from (relative or absolute)
 * @param {string} filename - Original video filename (used to derive .png name)
 * @returns {Promise<{file: File, previewUrl: string}>} PNG File + preview object URL
 */
export async function extractVideoFirstFrame(apiFetchFn, fetchUrl, filename) {
  const response = await apiFetchFn(fetchUrl)
  const videoBlob = await response.blob()
  const videoObjectUrl = URL.createObjectURL(videoBlob)

  const file = await new Promise((resolve, reject) => {
    const video = document.createElement('video')
    video.muted = true
    video.preload = 'auto'
    video.playsInline = true

    const cleanup = () => URL.revokeObjectURL(videoObjectUrl)

    video.onloadeddata = () => {
      try {
        const canvas = document.createElement('canvas')
        canvas.width = video.videoWidth || 640
        canvas.height = video.videoHeight || 480
        const ctx = canvas.getContext('2d')
        ctx.drawImage(video, 0, 0)
        canvas.toBlob((blob) => {
          cleanup()
          if (blob) {
            const pngName = (filename || 'frame.mp4').replace(/\.(mp4|webm|mov)$/i, '.png')
            resolve(new File([blob], pngName, { type: 'image/png' }))
          } else {
            reject(new Error('Canvas toBlob returned null'))
          }
        }, 'image/png')
      } catch (e) {
        cleanup()
        reject(e)
      }
    }

    video.onerror = () => {
      cleanup()
      reject(new Error('Failed to load video for frame extraction'))
    }

    video.src = videoObjectUrl
  })

  const previewUrl = URL.createObjectURL(file)
  return { file, previewUrl }
}
