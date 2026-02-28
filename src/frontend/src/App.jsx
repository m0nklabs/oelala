import React from 'react'
import * as Sentry from '@sentry/react'
import Dashboard from './dashboard/Dashboard'
import { NSFWProvider } from './contexts/NSFWContext'
import { AuthProvider, useAuth } from './contexts/AuthContext'
import { CreditsProvider } from './contexts/CreditsContext'
import './App.css'
import './components/ProgressBar.css'

/** Global error fallback shown when React tree crashes */
function ErrorFallback({ error, resetError }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', height: '100vh', padding: '2rem',
      background: '#0a0a0a', color: '#e0e0e0', textAlign: 'center',
    }}>
      <h1 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Something went wrong</h1>
      <p style={{ color: '#888', marginBottom: '1.5rem', maxWidth: '500px' }}>
        {error?.message || 'An unexpected error occurred.'}
      </p>
      <button
        onClick={resetError}
        style={{
          padding: '0.75rem 1.5rem', background: '#6366f1', color: '#fff',
          border: 'none', borderRadius: '8px', cursor: 'pointer', fontSize: '1rem',
        }}
      >
        Try again
      </button>
    </div>
  )
}

function AppContent() {
  const { loading } = useAuth()

  // Loading state (only while checking auth)
  if (loading) {
    return (
      <div className="app-loading">
        <div className="app-loading-spinner"></div>
      </div>
    )
  }

  // Show dashboard for everyone (logged in or not)
  // Login is optional and can be done via UserMenu
  // Credits are only available for logged-in users
  return (
    <CreditsProvider>
      <NSFWProvider>
        <Dashboard />
      </NSFWProvider>
    </CreditsProvider>
  )
}

function App() {
  return (
    <Sentry.ErrorBoundary fallback={({ error, resetError }) => (
      <ErrorFallback error={error} resetError={resetError} />
    )}>
      <AuthProvider>
        <AppContent />
      </AuthProvider>
    </Sentry.ErrorBoundary>
  )
}

export default App
