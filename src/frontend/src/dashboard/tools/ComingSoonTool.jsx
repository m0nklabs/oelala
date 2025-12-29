import React from 'react'

export default function ComingSoonTool({ title }) {
  return (
    <div className="tool-coming-soon">
      <div className="tool-title">{title}</div>
      <div className="muted">Missing backend endpoint (planned for v2).</div>
    </div>
  )
}
