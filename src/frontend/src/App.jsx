import React from 'react'
import Dashboard from './dashboard/Dashboard'
import { NSFWProvider } from './contexts/NSFWContext'
import { AuthProvider, useAuth } from './contexts/AuthContext'
import { CreditsProvider } from './contexts/CreditsContext'
import './App.css'
import './components/ProgressBar.css'

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
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  )
}

export default App
