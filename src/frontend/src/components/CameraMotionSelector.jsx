import React from 'react'

// Camera motion presets for Wan2.2 video generation
export const CAMERA_MOTIONS = [
  { value: '', label: 'None', desc: 'No camera motion', prefix: '' },
  { value: 'static', label: '📷 Static', desc: 'Camera stays still', prefix: 'static camera shot, ' },
  { value: 'pan_left', label: '⬅️ Pan Left', desc: 'Camera pans left', prefix: 'camera slowly panning left, ' },
  { value: 'pan_right', label: '➡️ Pan Right', desc: 'Camera pans right', prefix: 'camera slowly panning right, ' },
  { value: 'tilt_up', label: '⬆️ Tilt Up', desc: 'Camera tilts up', prefix: 'camera slowly tilting up, ' },
  { value: 'tilt_down', label: '⬇️ Tilt Down', desc: 'Camera tilts down', prefix: 'camera slowly tilting down, ' },
  { value: 'zoom_in', label: '🔍 Zoom In', desc: 'Camera zooms in', prefix: 'camera slowly zooming in, ' },
  { value: 'zoom_out', label: '🔭 Zoom Out', desc: 'Camera zooms out', prefix: 'camera slowly zooming out, ' },
  { value: 'dolly_in', label: '🎬 Dolly In', desc: 'Camera moves forward', prefix: 'camera dollying forward, ' },
  { value: 'dolly_out', label: '🎬 Dolly Out', desc: 'Camera moves back', prefix: 'camera dollying backward, ' },
  { value: 'orbit_left', label: '🔄 Orbit Left', desc: 'Camera orbits left', prefix: 'camera orbiting left around subject, ' },
  { value: 'orbit_right', label: '🔄 Orbit Right', desc: 'Camera orbits right', prefix: 'camera orbiting right around subject, ' },
  { value: 'handheld', label: '📹 Handheld', desc: 'Slight shake', prefix: 'shaky handheld camera, ' },
  { value: 'tracking', label: '🏃 Tracking', desc: 'Follows subject', prefix: 'camera tracking shot following subject, ' },
  { value: 'crane_up', label: '🏗️ Crane Up', desc: 'Camera rises up', prefix: 'crane shot rising up, ' },
  { value: 'crane_down', label: '🏗️ Crane Down', desc: 'Camera lowers', prefix: 'crane shot lowering down, ' },
]

/**
 * Get the prompt prefix for a camera motion value
 * @param {string} motionValue - The camera motion value
 * @returns {string} The prefix to prepend to the prompt
 */
export function getCameraMotionPrefix(motionValue) {
  const motion = CAMERA_MOTIONS.find(m => m.value === motionValue)
  return motion?.prefix || ''
}

/**
 * Camera Motion Selector Component
 * Reusable component for selecting camera motion presets in video generation tools
 */
export default function CameraMotionSelector({ value, onChange, style = {} }) {
  const selectedMotion = CAMERA_MOTIONS.find(m => m.value === value)

  return (
    <div style={{ marginBottom: '12px', ...style }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
        <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Camera Motion</span>
        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
          {value ? selectedMotion?.desc : 'Optional'}
        </span>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
        {CAMERA_MOTIONS.map(motion => (
          <button
            key={motion.value}
            onClick={() => onChange(motion.value === value ? '' : motion.value)}
            type="button"
            style={{
              padding: '6px 10px',
              borderRadius: '6px',
              border: value === motion.value ? '1px solid var(--accent-color)' : '1px solid var(--border-color)',
              background: value === motion.value ? 'rgba(59, 130, 246, 0.2)' : 'rgba(255,255,255,0.05)',
              color: value === motion.value ? 'var(--accent-color)' : 'var(--text-secondary)',
              fontSize: '0.8rem',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
            title={motion.desc}
          >
            {motion.label}
          </button>
        ))}
      </div>
    </div>
  )
}
