import React from 'react'
import Dashboard from './dashboard/Dashboard'
import LoginPage from './pages/LoginPage'
import { NSFWProvider } from './contexts/NSFWContext'
import { AuthProvider, useAuth } from './contexts/AuthContext'
import { CreditsProvider } from './contexts/CreditsContext'
import { isAuthEnabled } from './lib/supabase'
import './App.css'
import './components/ProgressBar.css'

function AppContent() {
  const { user, loading } = useAuth()

  // If auth is not configured, show dashboard directly (dev mode)
  if (!isAuthEnabled()) {
    return (
      <CreditsProvider>
        <NSFWProvider>
          <Dashboard />
        </NSFWProvider>
      </CreditsProvider>
    )
  }

  // Loading state
  if (loading) {
    return (
      <div className="app-loading">
        <div className="app-loading-spinner"></div>
      </div>
    )
  }

  // Not logged in - show login page
  if (!user) {
    return <LoginPage />
  }

  // Logged in - show dashboard with credits
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
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  )
}

export default App
