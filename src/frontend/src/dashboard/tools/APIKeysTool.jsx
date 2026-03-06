import React, { useState, useEffect, useCallback } from 'react'
import { useAuth } from '../../contexts/AuthContext'
import { BACKEND_BASE } from '../../config'
import { apiFetch } from '../../api'
import {
  Key, Plus, Copy, Trash2, Eye, EyeOff,
  Check, AlertCircle, Clock, Activity, Shield
} from 'lucide-react'

/**
 * API Keys Management Tool
 * Allows users to create, view, and revoke their API keys
 */
export default function APIKeysTool() {
  const { session, user } = useAuth()
  const [loading, setLoading] = useState(true)
  const [keys, setKeys] = useState([])
  const [error, setError] = useState(null)

  // Create key modal
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [newKeyName, setNewKeyName] = useState('')
  const [newKeyExpires, setNewKeyExpires] = useState('90') // days
  const [createLoading, setCreateLoading] = useState(false)
  const [createError, setCreateError] = useState('')

  // Newly created key (shown only once)
  const [newlyCreatedKey, setNewlyCreatedKey] = useState(null)
  const [keyCopied, setKeyCopied] = useState(false)

  // Delete confirmation
  const [deleteKeyId, setDeleteKeyId] = useState(null)
  const [deleteLoading, setDeleteLoading] = useState(false)

  // Fetch API keys
  const fetchKeys = useCallback(async () => {
    if (!session) return

    setLoading(true)
    setError(null)

    try {
      const response = await apiFetch('/api/keys')

      if (!response.ok) {
        throw new Error('Failed to fetch API keys')
      }

      const data = await response.json()
      setKeys(data)
    } catch (err) {
      console.error('Failed to fetch API keys:', err)
      setError('Failed to load API keys. Please try again.')
    } finally {
      setLoading(false)
    }
  }, [session])

  useEffect(() => {
    fetchKeys()
  }, [fetchKeys])

  // Create new API key
  const handleCreateKey = async () => {
    if (!newKeyName.trim()) {
      setCreateError('Please enter a name for the key')
      return
    }

    setCreateLoading(true)
    setCreateError('')

    try {
      const body = {
        name: newKeyName.trim(),
      }

      if (newKeyExpires && newKeyExpires !== 'never') {
        body.expires_days = parseInt(newKeyExpires, 10)
      }

      const response = await apiFetch('/api/keys', {
        method: 'POST',
        body: JSON.stringify(body),
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Failed to create API key')
      }

      const data = await response.json()
      setNewlyCreatedKey(data)
      setShowCreateModal(false)
      setNewKeyName('')
      setNewKeyExpires('90')
      fetchKeys()
    } catch (err) {
      console.error('Failed to create API key:', err)
      setCreateError(err.message || 'Failed to create API key')
    } finally {
      setCreateLoading(false)
    }
  }

  // Copy key to clipboard
  const handleCopyKey = async (key) => {
    try {
      await navigator.clipboard.writeText(key)
      setKeyCopied(true)
      setTimeout(() => setKeyCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  // Delete API key
  const handleDeleteKey = async () => {
    if (!deleteKeyId) return

    setDeleteLoading(true)

    try {
      const response = await apiFetch(`/api/keys/${deleteKeyId}`, {
        method: 'DELETE',
      })

      if (!response.ok) {
        throw new Error('Failed to delete API key')
      }

      setDeleteKeyId(null)
      fetchKeys()
    } catch (err) {
      console.error('Failed to delete API key:', err)
    } finally {
      setDeleteLoading(false)
    }
  }

  // Toggle key active status
  const handleToggleKey = async (keyId, currentStatus) => {
    try {
      const response = await apiFetch(`/api/keys/${keyId}`, {
        method: 'PATCH',
        body: JSON.stringify({ is_active: !currentStatus }),
      })

      if (!response.ok) {
        throw new Error('Failed to update API key')
      }

      fetchKeys()
    } catch (err) {
      console.error('Failed to toggle API key:', err)
    }
  }

  // Format date
  const formatDate = (dateString) => {
    if (!dateString) return 'Never'
    const date = new Date(dateString)
    return date.toLocaleDateString('nl-NL', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    })
  }

  // Check if key is expired
  const isExpired = (expiresAt) => {
    if (!expiresAt) return false
    return new Date(expiresAt) < new Date()
  }

  if (!session) {
    return (
      <div className="p-6 text-center">
        <Key className="w-12 h-12 mx-auto mb-4 text-zinc-500" />
        <p className="text-zinc-400">Please log in to manage API keys</p>
      </div>
    )
  }

  return (
    <div className="p-4 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Key className="w-5 h-5" />
            API Keys
          </h2>
          <p className="text-sm text-zinc-400 mt-1">
            Manage your API keys for programmatic access
          </p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
        >
          <Plus className="w-4 h-4" />
          Create Key
        </button>
      </div>

      {/* Newly created key banner */}
      {newlyCreatedKey && (
        <div className="bg-green-900/30 border border-green-600/50 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <Check className="w-5 h-5 text-green-500 mt-0.5" />
            <div className="flex-1">
              <h3 className="font-medium text-green-400">API Key Created!</h3>
              <p className="text-sm text-zinc-400 mt-1">
                Copy your key now. You won't be able to see it again!
              </p>
              <div className="mt-3 flex items-center gap-2">
                <code className="flex-1 bg-zinc-900 px-3 py-2 rounded font-mono text-sm text-zinc-300 break-all">
                  {newlyCreatedKey.api_key}
                </code>
                <button
                  onClick={() => handleCopyKey(newlyCreatedKey.api_key)}
                  className="p-2 bg-zinc-700 hover:bg-zinc-600 rounded transition-colors"
                  title="Copy to clipboard"
                >
                  {keyCopied ? (
                    <Check className="w-4 h-4 text-green-500" />
                  ) : (
                    <Copy className="w-4 h-4 text-zinc-300" />
                  )}
                </button>
              </div>
              <button
                onClick={() => setNewlyCreatedKey(null)}
                className="mt-3 text-sm text-zinc-400 hover:text-zinc-300"
              >
                I've saved my key
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="bg-red-900/30 border border-red-600/50 rounded-lg p-4 flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-red-500" />
          <span className="text-red-400">{error}</span>
        </div>
      )}

      {/* Loading state */}
      {loading && (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full" />
        </div>
      )}

      {/* Keys list */}
      {!loading && keys.length === 0 && (
        <div className="text-center py-12 bg-zinc-800/50 rounded-lg">
          <Key className="w-12 h-12 mx-auto mb-4 text-zinc-500" />
          <p className="text-zinc-400">No API keys yet</p>
          <p className="text-sm text-zinc-500 mt-1">
            Create your first API key to get started
          </p>
        </div>
      )}

      {!loading && keys.length > 0 && (
        <div className="space-y-3">
          {keys.map((key) => (
            <div
              key={key.id}
              className={`bg-zinc-800/50 rounded-lg p-4 border ${
                !key.is_active || isExpired(key.expires_at)
                  ? 'border-zinc-700 opacity-60'
                  : 'border-zinc-700'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <h3 className="font-medium text-white">{key.name}</h3>
                    {!key.is_active && (
                      <span className="px-2 py-0.5 bg-zinc-700 text-zinc-400 text-xs rounded">
                        Disabled
                      </span>
                    )}
                    {isExpired(key.expires_at) && (
                      <span className="px-2 py-0.5 bg-red-900/50 text-red-400 text-xs rounded">
                        Expired
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-4 mt-2 text-sm text-zinc-400">
                    <span className="font-mono">{key.key_prefix}...</span>
                    <span className="flex items-center gap-1">
                      <Activity className="w-3 h-3" />
                      {key.usage_count} uses
                    </span>
                    {key.last_used_at && (
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        Last used: {formatDate(key.last_used_at)}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-4 mt-1 text-xs text-zinc-500">
                    <span>Created: {formatDate(key.created_at)}</span>
                    <span>
                      Expires:{' '}
                      {key.expires_at ? formatDate(key.expires_at) : 'Never'}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleToggleKey(key.id, key.is_active)}
                    className={`p-2 rounded transition-colors ${
                      key.is_active
                        ? 'bg-green-900/30 hover:bg-green-900/50 text-green-500'
                        : 'bg-zinc-700 hover:bg-zinc-600 text-zinc-400'
                    }`}
                    title={key.is_active ? 'Disable key' : 'Enable key'}
                  >
                    {key.is_active ? (
                      <Eye className="w-4 h-4" />
                    ) : (
                      <EyeOff className="w-4 h-4" />
                    )}
                  </button>
                  <button
                    onClick={() => setDeleteKeyId(key.id)}
                    className="p-2 bg-red-900/30 hover:bg-red-900/50 text-red-500 rounded transition-colors"
                    title="Delete key"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Usage instructions */}
      <div className="bg-zinc-800/30 rounded-lg p-4 border border-zinc-700">
        <h3 className="font-medium text-white flex items-center gap-2 mb-3">
          <Shield className="w-4 h-4" />
          How to use API keys
        </h3>
        <div className="text-sm text-zinc-400 space-y-2">
          <p>Include your API key in the <code className="bg-zinc-800 px-1.5 py-0.5 rounded text-zinc-300">X-API-Key</code> header:</p>
          <pre className="bg-zinc-900 p-3 rounded overflow-x-auto text-xs">
{`curl -X POST https://oelala.xyz/api/v1/generate/video \\
  -H "X-API-Key: YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "A beautiful sunset", "mode": "fast"}'`}
          </pre>
        </div>
      </div>

      {/* Create key modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-zinc-800 rounded-lg p-6 w-full max-w-md mx-4">
            <h3 className="text-lg font-medium text-white mb-4">
              Create API Key
            </h3>

            {createError && (
              <div className="mb-4 p-3 bg-red-900/30 border border-red-600/50 rounded text-sm text-red-400">
                {createError}
              </div>
            )}

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-zinc-300 mb-1">
                  Key Name
                </label>
                <input
                  type="text"
                  value={newKeyName}
                  onChange={(e) => setNewKeyName(e.target.value)}
                  placeholder="My Production App"
                  className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-white placeholder-zinc-500 focus:outline-none focus:border-blue-500"
                  autoFocus
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-zinc-300 mb-1">
                  Expiration
                </label>
                <select
                  value={newKeyExpires}
                  onChange={(e) => setNewKeyExpires(e.target.value)}
                  className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-blue-500"
                >
                  <option value="30">30 days</option>
                  <option value="90">90 days</option>
                  <option value="180">180 days</option>
                  <option value="365">1 year</option>
                  <option value="never">Never expires</option>
                </select>
              </div>
            </div>

            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => {
                  setShowCreateModal(false)
                  setCreateError('')
                }}
                className="px-4 py-2 bg-zinc-700 hover:bg-zinc-600 text-white rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateKey}
                disabled={createLoading}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 disabled:opacity-50 text-white rounded-lg transition-colors flex items-center gap-2"
              >
                {createLoading ? (
                  <>
                    <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full" />
                    Creating...
                  </>
                ) : (
                  <>
                    <Plus className="w-4 h-4" />
                    Create Key
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete confirmation modal */}
      {deleteKeyId && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-zinc-800 rounded-lg p-6 w-full max-w-md mx-4">
            <h3 className="text-lg font-medium text-white mb-2">
              Delete API Key?
            </h3>
            <p className="text-zinc-400 mb-6">
              This action cannot be undone. Any applications using this key will
              stop working.
            </p>

            <div className="flex justify-end gap-3">
              <button
                onClick={() => setDeleteKeyId(null)}
                className="px-4 py-2 bg-zinc-700 hover:bg-zinc-600 text-white rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleDeleteKey}
                disabled={deleteLoading}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-red-800 disabled:opacity-50 text-white rounded-lg transition-colors flex items-center gap-2"
              >
                {deleteLoading ? (
                  <>
                    <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full" />
                    Deleting...
                  </>
                ) : (
                  <>
                    <Trash2 className="w-4 h-4" />
                    Delete Key
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
