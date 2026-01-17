import React from 'react'
import { ChevronDown } from 'lucide-react'

// Camera position/angle presets for T2I image generation
export const CAMERA_POSITIONS = [
  { value: '', label: 'None', desc: 'No camera angle specified', prefix: '' },
  // Angle shots
  { value: 'eye_level', label: '👁️ Eye Level', desc: 'Standard eye level', prefix: 'eye level shot, ' },
  { value: 'low_angle', label: '⬆️ Low Angle', desc: 'Looking up at subject', prefix: 'low angle shot, looking up, ' },
  { value: 'high_angle', label: '⬇️ High Angle', desc: 'Looking down at subject', prefix: 'high angle shot, looking down, ' },
  { value: 'dutch_angle', label: '📐 Dutch Angle', desc: 'Tilted frame', prefix: 'dutch angle, tilted frame, ' },
  { value: 'birds_eye', label: '🦅 Bird\'s Eye', desc: 'Directly from above', prefix: 'bird\'s eye view, top down shot, ' },
  { value: 'worms_eye', label: '🐛 Worm\'s Eye', desc: 'Directly from below', prefix: 'worm\'s eye view, extreme low angle, ' },
  // Distance shots
  { value: 'extreme_close', label: '🔬 Extreme Close-up', desc: 'Detail shot', prefix: 'extreme close-up, macro shot, ' },
  { value: 'close_up', label: '😊 Close-up', desc: 'Face/detail focus', prefix: 'close-up shot, ' },
  { value: 'medium_close', label: '👤 Medium Close-up', desc: 'Head and shoulders', prefix: 'medium close-up, head and shoulders, ' },
  { value: 'medium', label: '🧍 Medium Shot', desc: 'Waist up', prefix: 'medium shot, waist up, ' },
  { value: 'medium_full', label: '🚶 Medium Full', desc: 'Knees up', prefix: 'medium full shot, cowboy shot, ' },
  { value: 'full', label: '🧑‍🤝‍🧑 Full Shot', desc: 'Full body', prefix: 'full body shot, ' },
  { value: 'wide', label: '🏞️ Wide Shot', desc: 'Subject and environment', prefix: 'wide shot, establishing shot, ' },
  { value: 'extreme_wide', label: '🌄 Extreme Wide', desc: 'Vast landscape', prefix: 'extreme wide shot, panoramic view, ' },
  // Special shots
  { value: 'over_shoulder', label: '🤝 Over Shoulder', desc: 'Behind subject looking forward', prefix: 'over the shoulder shot, ' },
  { value: 'pov', label: '👀 POV', desc: 'First person view', prefix: 'POV shot, first person view, ' },
  { value: 'profile', label: '👤 Profile', desc: 'Side view', prefix: 'profile shot, side view, ' },
  { value: 'three_quarter', label: '🎭 Three-Quarter', desc: '45 degree angle', prefix: 'three-quarter view, 45 degree angle, ' },
  { value: 'from_behind', label: '🔙 From Behind', desc: 'Back view', prefix: 'shot from behind, back view, ' },
]

/**
 * Get the prompt prefix for a camera position value
 * @param {string} positionValue - The camera position value
 * @returns {string} The prefix to prepend to the prompt
 */
export function getCameraPositionPrefix(positionValue) {
  const position = CAMERA_POSITIONS.find(p => p.value === positionValue)
  return position?.prefix || ''
}

/**
 * Camera Position Selector Component - Dropdown version
 * Compact dropdown for selecting camera position/angle presets
 */
export default function CameraPositionSelector({ value, onChange, style = {} }) {
  return (
    <div style={{ ...style }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
        <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Camera Position</span>
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
          {CAMERA_POSITIONS.map(pos => (
            <option key={pos.value} value={pos.value}>
              {pos.label} {pos.desc && pos.value ? `- ${pos.desc}` : ''}
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
    </div>
  )
}
