/**
 * Credits Context & Hook
 * Manages user credit balance and purchase flow
 */

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { useAuth } from './AuthContext'
import { apiFetch } from '../api'
import { DEBUG } from '../config'

const CreditsContext = createContext(null)

/**
 * Credits Provider - wraps app to provide credit state
 */
export function CreditsProvider({ children }) {
  const { user } = useAuth()
  const [balance, setBalance] = useState(0)
  const [lifetimePurchased, setLifetimePurchased] = useState(0)
  const [lifetimeUsed, setLifetimeUsed] = useState(0)
  const [packages, setPackages] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Fetch balance when user changes
  const fetchBalance = useCallback(async () => {
    if (!user) {
      setBalance(0)
      setLifetimePurchased(0)
      setLifetimeUsed(0)
      return
    }

    setLoading(true)
    setError(null)

    try {
      const res = await apiFetch('/api/credits')
      if (res.ok) {
        const data = await res.json()
        setBalance(data.balance || 0)
        setLifetimePurchased(data.lifetime_purchased || 0)
        setLifetimeUsed(data.lifetime_used || 0)
        if (DEBUG) console.log('💰 Credits balance:', data.balance)
      } else {
        console.error('Failed to fetch credits:', res.status)
        // Don't error out - might just be no credits record yet
        setBalance(0)
      }
    } catch (e) {
      console.error('Credits fetch error:', e)
      setError('Failed to load credits')
    } finally {
      setLoading(false)
    }
  }, [user])

  // Fetch packages (public, no auth needed)
  const fetchPackages = useCallback(async () => {
    try {
      const res = await apiFetch('/api/credits/packages')
      if (res.ok) {
        const data = await res.json()
        setPackages(data)
        if (DEBUG) console.log('💰 Credit packages:', data.length)
      }
    } catch (e) {
      console.error('Packages fetch error:', e)
    }
  }, [])

  // Fetch balance when user changes (use user.id as stable reference)
  useEffect(() => {
    fetchBalance()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id]) // Only refetch when user.id actually changes

  // Fetch packages once on mount (they don't depend on user)
  useEffect(() => {
    fetchPackages()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // Only fetch once on mount

  /**
   * Estimate credits for a generation
   */
  const estimateCost = useCallback(async (generationType, params = {}) => {
    try {
      const res = await apiFetch('/api/credits/estimate', {
        method: 'POST',
        body: JSON.stringify({
          generation_type: generationType,
          width: params.width || 1024,
          height: params.height || 1024,
          duration_seconds: params.duration_seconds || null,
          steps: params.steps || 20,
        }),
      })
      if (res.ok) {
        return await res.json()
      }
    } catch (e) {
      console.error('Estimate error:', e)
    }
    return null
  }, [])

  /**
   * Initiate credit purchase - returns Stripe checkout URL
   */
  const purchaseCredits = useCallback(async (packageId) => {
    if (!user) {
      setError('Please sign in to purchase credits')
      return null
    }

    try {
      const res = await apiFetch('/api/credits/purchase', {
        method: 'POST',
        body: JSON.stringify({
          package_id: packageId,
        }),
      })

      if (res.ok) {
        const data = await res.json()
        return data.checkout_url
      } else {
        const err = await res.json()
        setError(err.detail || 'Purchase failed')
        return null
      }
    } catch (e) {
      console.error('Purchase error:', e)
      setError('Purchase failed')
      return null
    }
  }, [user])

  /**
   * Check if user has enough credits
   */
  const hasCredits = useCallback((amount) => {
    return balance >= amount
  }, [balance])

  /**
   * Update balance after successful generation (optimistic)
   */
  const deductCredits = useCallback((amount) => {
    setBalance(prev => Math.max(0, prev - amount))
    setLifetimeUsed(prev => prev + amount)
  }, [])

  /**
   * Refund credits (if generation fails)
   */
  const refundCredits = useCallback((amount) => {
    setBalance(prev => prev + amount)
    setLifetimeUsed(prev => Math.max(0, prev - amount))
  }, [])

  const value = {
    // State
    balance,
    lifetimePurchased,
    lifetimeUsed,
    packages,
    loading,
    error,

    // Actions
    fetchBalance,
    estimateCost,
    purchaseCredits,
    hasCredits,
    deductCredits,
    refundCredits,
    clearError: () => setError(null),
  }

  return (
    <CreditsContext.Provider value={value}>
      {children}
    </CreditsContext.Provider>
  )
}

/**
 * useCredits hook - access credits state and actions
 */
export function useCredits() {
  const context = useContext(CreditsContext)
  if (!context) {
    throw new Error('useCredits must be used within a CreditsProvider')
  }
  return context
}
