import React, { lazy, Suspense } from 'react'
import { Loader2 } from 'lucide-react'

const MyMediaTool = lazy(() => import('../dashboard/tools/MyMediaTool'))

/**
 * Reusable modal overlay for picking media from "My Creations".
 *
 * Props:
 *  - show       {boolean}  Whether to render the modal
 *  - onClose    {function} Called when user closes the modal
 *  - onSelect   {function} Called with the selected media item
 *  - filter     {'image'|'video'|'all'} Media type filter (default: 'all')
 *  - title      {string}   Header text (default: 'Select from My Creations')
 */
export default function CreationsPickerModal({ show, onClose, onSelect, filter = 'all', title = 'Select from My Creations' }) {
  if (!show) return null

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0,0,0,0.8)',
      zIndex: 1000,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px',
    }}>
      <div style={{
        backgroundColor: 'var(--bg-primary)',
        borderRadius: '12px',
        width: '100%',
        maxWidth: '1200px',
        maxHeight: '80vh',
        overflow: 'auto',
        position: 'relative',
      }}>
        {/* Sticky header */}
        <div style={{
          position: 'sticky',
          top: 0,
          padding: '16px',
          backgroundColor: 'var(--bg-primary)',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          zIndex: 1,
        }}>
          <h3 style={{ margin: 0 }}>{title}</h3>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              fontSize: '1.5rem',
              lineHeight: 1,
            }}
          >
            ×
          </button>
        </div>
        {/* Media grid */}
        <div style={{ padding: '16px' }}>
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
      </div>
    </div>
  )
}
