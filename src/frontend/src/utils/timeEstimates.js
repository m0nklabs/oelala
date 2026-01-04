/**
 * Estimate generation time based on parameters.
 * 
 * These are rough estimates based on typical GPU performance.
 * Actual times vary based on GPU load and specific settings.
 */

// Base time estimates (in seconds) for 480p, 6s video @ 16fps
const BASE_TIME_SECONDS = 90 // ~1.5 min for standard I2V

// Multipliers for different settings
const RESOLUTION_MULTIPLIER = {
  '360p': 0.6,
  '480p': 1.0,
  '540p': 1.3,
  '720p': 2.0,
  '1080p': 4.0,
}

const DURATION_PER_SECOND = 15 // Additional seconds per video second

/**
 * Estimate generation time for Image-to-Video
 * @param {object} params - Generation parameters
 * @returns {object} - { seconds, formatted, range }
 */
export function estimateI2VTime({ 
  resolution = '480p', 
  duration = 6,
  steps = 6,
}) {
  // Base calculation
  let seconds = BASE_TIME_SECONDS
  
  // Adjust for resolution
  const resMult = RESOLUTION_MULTIPLIER[resolution] || 1.0
  seconds *= resMult
  
  // Adjust for duration (more frames = more time)
  seconds += (duration - 6) * DURATION_PER_SECOND
  
  // Adjust for steps (DisTorch uses fewer steps)
  if (steps > 6) {
    seconds *= (steps / 6)
  }
  
  // Add variance for range
  const min = Math.round(seconds * 0.8)
  const max = Math.round(seconds * 1.3)
  
  return {
    seconds: Math.round(seconds),
    min,
    max,
    formatted: formatTime(seconds),
    range: `${formatTime(min)} - ${formatTime(max)}`,
  }
}

/**
 * Estimate generation time for Text-to-Video
 * (Slightly longer due to T2I step)
 */
export function estimateT2VTime({ 
  resolution = '480p', 
  numFrames = 41,
  steps = 6,
  t2iSteps = 20,
}) {
  const duration = numFrames / 16 // Approximate seconds
  
  // T2V includes T2I step
  const t2iTime = t2iSteps * 1.5 // ~1.5 seconds per step for T2I
  
  let estimate = estimateI2VTime({ resolution, duration, steps })
  estimate.seconds += t2iTime
  estimate.min += t2iTime * 0.8
  estimate.max += t2iTime * 1.2
  estimate.formatted = formatTime(estimate.seconds)
  estimate.range = `${formatTime(estimate.min)} - ${formatTime(estimate.max)}`
  
  return estimate
}

/**
 * Format seconds to human-readable time
 */
export function formatTime(seconds) {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`
  }
  const mins = Math.floor(seconds / 60)
  const secs = Math.round(seconds % 60)
  if (secs === 0) {
    return `${mins} min`
  }
  return `${mins}m ${secs}s`
}

/**
 * Calculate queue wait time
 * @param {number} queuePosition - Position in queue (0 = next)
 * @param {number} avgJobTime - Average time per job in seconds
 * @returns {object} - Wait time estimate
 */
export function estimateQueueWait(queuePosition, avgJobTime = 120) {
  const waitSeconds = queuePosition * avgJobTime
  return {
    seconds: waitSeconds,
    formatted: formatTime(waitSeconds),
    position: queuePosition + 1,
  }
}
