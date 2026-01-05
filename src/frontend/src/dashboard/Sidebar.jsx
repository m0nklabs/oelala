import React from 'react'
import { NAV_GROUPS } from './nav'
import { ChevronLeft, ChevronRight } from 'lucide-react'

// Emoji icons for reliable cross-browser display
const ICONS = {
  // Video Tools
  'image-to-video': '🎬',
  'text-to-video': '📝',
  'text-to-image-to-video': '✨',
  'video-to-video': '🔄',
  'speech-to-video': '🎤',
  // Image Tools
  'text-to-image': '🖼️',
  'image-to-image': '🎨',
  'upscaler': '🔍',
  'reframe': '📐',
  'face-swap': '🎭',
  // Prompt Tools
  'prompt-generator': '💡',
  'image-to-text': '📷',
  'video-to-text': '🎥',
  // Audio Tools
  'audio-generation': '🔊',
  'voice-cloning': '🗣️',
  'lip-sync': '👄',
  // Advanced
  'pipeline': '⚙️',
  'lora-training': '🧠',
  // My Media
  'my-media-all': '📁',
  'my-media-videos': '🎞️',
  'my-media-images': '🖼️',
  'my-media-audio': '🎵',
  'my-media-prompts': '📝',
}

export default function Sidebar({ activeToolId, onSelectTool, collapsed, onToggleCollapsed }) {
  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-header">
        <div className="sidebar-logo">Oelala</div>
      </div>

      <nav className="sidebar-nav">
        {NAV_GROUPS.map((group) => (
          <div key={group.id} className="sidebar-group">
            <div className="sidebar-group-title">{group.title}</div>

            {group.items.map((item) => {
              const isActive = activeToolId === item.id
              const icon = ICONS[item.id] || '🔧'

              return (
                <button
                  key={item.id}
                  className={`nav-item${isActive ? ' active' : ''}`}
                  onClick={() => onSelectTool(item.id)}
                  type="button"
                >
                  <span className="nav-icon" style={{ fontSize: '16px' }}>
                    {icon}
                  </span>
                  <span className="nav-label">{item.label}</span>
                  {item.status === 'new' && <span className="nav-badge">new</span>}
                </button>
              )
            })}
          </div>
        ))}
      </nav>

      <div className="sidebar-footer">
        <button
            onClick={onToggleCollapsed}
            className="nav-item collapse-btn"
        >
            <span className="nav-icon" style={{ fontSize: '16px' }}>
              {collapsed ? '▶️' : '◀️'}
            </span>
            <span className="nav-label">Collapse</span>
        </button>
      </div>
    </aside>
  )
}
