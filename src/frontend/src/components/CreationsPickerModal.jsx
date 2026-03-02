import React, { lazy, Suspense } from 'react'
import { Loader2, X } from 'lucide-react'

const MyMediaTool = lazy(() => import('../dashboard/tools/MyMediaTool'))

/**
 * Inline expandable panel for picking media from "My Creations".
 * Replaces the old full-screen overlay with a clean inline panel
 * that renders in-place within the tool's layout.
 *
 * Props:
 *  - show       {boolean}  Whether to render the panel
 *  - onClose    {function} Called when user closes the panel
 *  - onSelect   {function} Called with the selected media item
 *  - filter     {'image'|'video'|'all'} Media type filter (default: 'all')
 *  - title      {string}   Header text (default: 'Select from My Creations')
 */
export default function CreationsPickerModal({ show, onClose, onSelect, filter = 'all', title = 'Select from My Creations' }) {
  if (!show) return null

  return (
    <div className="creations-picker-inline">
      {/* Header */}
      <div className="creations-picker-header">
        <h3>{title}</h3>
        <button onClick={onClose} className="creations-picker-close" title="Close">
          <X size={16} />
        </button>
      </div>
      {/* Media grid */}
      <div className="creations-picker-content">
        <Suspense fallback={
          <div style={{ textAlign: 'center', padding: '40px' }}>
            <Loader2 size={24} className="animate-spin" style={{ margin: '0 auto' }} />
            <p style={{ marginTop: 8, color: 'var(--text-muted)' }}>Loading media…</p>
          </div>
        }>
          <MyMediaTool
            filter={filter}
            selectionMode={true}
            onSelectItem={(item) => {
              onSelect(item)
              onClose()
            }}
          />
        </Suspense>
      </div>

      <style>{`
        .creations-picker-inline {
          border: 1px solid var(--border-color, #333);
          border-radius: 12px;
          background: var(--bg-secondary, #1a1a2e);
          margin-top: 8px;
          overflow: hidden;
          animation: creations-picker-slide-in 0.2s ease-out;
        }
        @keyframes creations-picker-slide-in {
          from { opacity: 0; max-height: 0; }
          to { opacity: 1; max-height: 2000px; }
        }
        .creations-picker-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 10px 16px;
          border-bottom: 1px solid var(--border-color, #333);
          background: var(--bg-tertiary, #16162a);
        }
        .creations-picker-header h3 {
          margin: 0;
          font-size: 0.9rem;
          font-weight: 600;
          color: var(--text-primary, #e0e0e0);
        }
        .creations-picker-close {
          background: none;
          border: none;
          color: var(--text-muted, #888);
          cursor: pointer;
          padding: 4px;
          border-radius: 6px;
          display: flex;
          align-items: center;
          transition: all 0.15s;
        }
        .creations-picker-close:hover {
          background: var(--bg-hover, rgba(255,255,255,0.1));
          color: var(--text-primary, #e0e0e0);
        }
        .creations-picker-content {
          max-height: 60vh;
          overflow-y: auto;
          padding: 8px;
        }
      `}</style>
    </div>
  )
}
