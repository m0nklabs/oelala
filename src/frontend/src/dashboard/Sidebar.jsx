import React from 'react'
import { NAV_GROUPS } from './nav'
import { 
  Type, Image as ImageIcon, Video, Film, 
  Maximize2, RefreshCw, User, ChevronLeft, ChevronRight,
  Layers, Wand2, Workflow, Clapperboard, FolderOpen, Play, ImagePlus
} from 'lucide-react'

const ICONS = {
  'text-to-video': Video,
  'image-to-video': Film,
  'text-to-image-to-video': Clapperboard,
  'pipeline': Workflow,
  'video-to-video': Layers,
  'text-to-image': Type,
  'image-to-image': ImageIcon,
  'reframe': Maximize2,
  'face-swap': User,
  'upscaler': Wand2,
  'lora-training': RefreshCw,
  'my-media-all': FolderOpen,
  'my-media-videos': Play,
  'my-media-images': ImagePlus,
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
              const Icon = ICONS[item.id] || Wand2
              
              return (
                <button
                  key={item.id}
                  className={`nav-item${isActive ? ' active' : ''}`}
                  onClick={() => onSelectTool(item.id)}
                  type="button"
                >
                  <span className="nav-icon">
                    <Icon size={18} />
                  </span>
                  <span className="nav-label">{item.label}</span>
                  {item.status === 'missing-backend' && <span className="nav-badge">v2</span>}
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
            <span className="nav-icon">
              {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
            </span>
            <span className="nav-label">Collapse</span>
        </button>
      </div>
    </aside>
  )
}
