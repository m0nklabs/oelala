import React from 'react'
import { NAV_GROUPS } from './nav'
import { ChevronLeft, ChevronRight, PanelLeftClose, PanelLeft } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'

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
  // Admin
  'admin-panel': '👑',
}

export default function Sidebar({ activeToolId, onSelectTool, collapsed, onToggleCollapsed }) {
  const { isAdmin } = useAuth()

  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-header">
        <div className="sidebar-logo">Oelala</div>
        <button
          onClick={onToggleCollapsed}
          className="sidebar-collapse-btn"
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <PanelLeft size={18} /> : <PanelLeftClose size={18} />}
        </button>
      </div>

      <nav className="sidebar-nav">
        {NAV_GROUPS.map((group) => {
          // Skip admin-only groups if user is not admin
          if (group.adminOnly && !isAdmin) {
            return null
          }

          return (
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
          )
        })}
      </nav>
    </aside>
  )
}
