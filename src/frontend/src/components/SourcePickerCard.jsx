import { useRef } from 'react'

/**
 * The source-media card used by every generation tool.
 *
 * Image-to-Video already grouped its source options in one card with tabs
 * (Upload File / From URL / From My Creations / My Uploads). The other tools
 * instead crammed a "From My Creations" button plus an expanding panel into
 * the same frame as their settings, which left no room to browse. This
 * component carries the tabbed card so every tool can look and behave the
 * same way.
 *
 * @param {object} props
 * @param {React.ReactNode} [props.title] - card header; a node is allowed so callers
 *   can put an icon next to the label (e.g. "Source Image" with an image icon)
 * @param {Array<{id: string, label: string, icon?: Function, badge?: string|number}>} props.tabs
 * @param {string} props.active - id of the active tab
 * @param {(id: string) => void} props.onChange - called with the clicked tab id
 * @param {React.ReactNode} props.children - content for the active tab
 * @param {React.ReactNode} [props.headerAction] - optional control on the header row
 */
export default function SourcePickerCard({
  title = 'Source',
  tabs = [],
  active,
  onChange,
  children,
  className = '',
  headerAction = null,
}) {
  const tabRefs = useRef({})

  // Roving tabindex needs arrow-key navigation, otherwise the inactive tabs are
  // unreachable by keyboard (the ARIA tabs pattern).
  const handleKeyDown = (event, index) => {
    if (!tabs.length) return
    let nextIndex = null
    if (event.key === 'ArrowRight') nextIndex = (index + 1) % tabs.length
    else if (event.key === 'ArrowLeft') nextIndex = (index - 1 + tabs.length) % tabs.length
    else if (event.key === 'Home') nextIndex = 0
    else if (event.key === 'End') nextIndex = tabs.length - 1
    if (nextIndex === null) return

    event.preventDefault()
    const next = tabs[nextIndex]
    onChange(next.id)
    tabRefs.current[next.id]?.focus()
  }

  return (
    <div className={`grok-card ${className}`.trim()}>
      <div className="grok-card-header">
        <div className="grok-card-title">{title}</div>
        {headerAction}
      </div>

      <div className="grok-tabs" role="tablist">
        {tabs.map((tab, index) => {
          const Icon = tab.icon
          const isActive = active === tab.id
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              id={`source-tab-${tab.id}`}
              aria-selected={isActive}
              aria-controls={`source-tabpanel-${tab.id}`}
              tabIndex={isActive ? 0 : -1}
              className={`grok-tab ${isActive ? 'active' : ''}`}
              onClick={() => onChange(tab.id)}
              onKeyDown={(event) => handleKeyDown(event, index)}
              ref={(element) => { tabRefs.current[tab.id] = element }}
            >
              {Icon ? <Icon size={14} /> : null}
              {Icon ? ' ' : null}
              {tab.label}
              {tab.badge ? (
                <span style={{ fontSize: '10px', opacity: 0.7, marginLeft: '4px' }}>
                  ({tab.badge})
                </span>
              ) : null}
            </button>
          )
        })}
      </div>

      <div
        className="grok-tabpanel"
        role="tabpanel"
        id={`source-tabpanel-${active}`}
        aria-labelledby={`source-tab-${active}`}
      >
        {children}
      </div>
    </div>
  )
}
