import React from 'react'
import { ChevronDown } from 'lucide-react'

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
 * Camera Motion Selector Component - Dropdown version
 * Compact dropdown for selecting camera motion presets
 */
export default function CameraMotionSelector({ value, onChange, style = {} }) {
  const selectedMotion = CAMERA_MOTIONS.find(m => m.value === value)

  return (
    <div style={{ ...style }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
        <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Camera Motion</span>
      </div>
      <div style={{ position: 'relative' }}>
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          style={{
            width: '100%',
            padding: '10px 36px 10px 12px',
            backgroundColor: 'var(--bg-secondary, #1a1a1a)',
            border: '1px solid var(--border-color)',
            borderRadius: '8px',
            color: 'var(--text-primary, #fff)',
            fontSize: '0.9rem',
            appearance: 'none',
            cursor: 'pointer',
          }}
        >
          {CAMERA_MOTIONS.map(motion => (
            <option key={motion.value} value={motion.value}>
              {motion.label} {motion.desc && motion.value ? `- ${motion.desc}` : ''}
            </option>
          ))}
        </select>
        <ChevronDown
          size={18}
          style={{
            position: 'absolute',
            right: '12px',
            top: '50%',
            transform: 'translateY(-50%)',
            pointerEvents: 'none',
            color: 'var(--text-muted)'
          }}
        />
      </div>
      {/* Show prefix preview when motion is selected */}
      {value && selectedMotion?.prefix && (
        <div style={{
          marginTop: '6px',
          padding: '6px 10px',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          borderRadius: '6px',
          fontSize: '0.75rem',
          color: 'var(--accent-color)',
          fontStyle: 'italic',
        }}>
          ➕ Prefix: "{selectedMotion.prefix}"
        </div>
      )}
    </div>
  )
}
