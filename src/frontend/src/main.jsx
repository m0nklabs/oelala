import React from 'react'
import ReactDOM from 'react-dom/client'
import * as Sentry from '@sentry/react'
import App from './App.jsx'
import './index.css'

// ── Sentry initialization ──────────────────────────────────────────
const SENTRY_DSN = import.meta.env.VITE_SENTRY_DSN || ''
if (SENTRY_DSN) {
  Sentry.init({
    dsn: SENTRY_DSN,
    environment: import.meta.env.DEV ? 'development' : 'production',
    integrations: [
      Sentry.browserTracingIntegration(),
      Sentry.replayIntegration({ maskAllText: false, blockAllMedia: false }),
    ],
    tracesSampleRate: parseFloat(import.meta.env.VITE_SENTRY_TRACES_SAMPLE_RATE || '0.2'),
    replaysSessionSampleRate: 0.05,
    replaysOnErrorSampleRate: 0.5,
    // Don't send health-check polling noise
    beforeSendTransaction(event) {
      if (event.transaction === '/health') return null
      return event
    },
  })
  console.log('✅ Sentry initialized')
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
