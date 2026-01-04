import React from 'react'
import Dashboard from './dashboard/Dashboard'
import { NSFWProvider } from './contexts/NSFWContext'
import './App.css'
import './components/ProgressBar.css'

function App() {
  return (
    <NSFWProvider>
      <Dashboard />
    </NSFWProvider>
  )
}

export default App
