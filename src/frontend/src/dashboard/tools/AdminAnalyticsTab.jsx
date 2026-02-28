import React, { useState, useEffect, useMemo } from 'react'
import { BarChart3, Users, Coins, TrendingUp, Clock, Activity, RefreshCw, Zap, HardDrive } from 'lucide-react'
import { BACKEND_BASE, DEBUG } from '../../config'
import { useAuth } from '../../contexts/AuthContext'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from 'recharts'

const COLORS = ['#8b5cf6', '#6366f1', '#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#ec4899']

function StatCard({ icon: Icon, label, value, sub, color = 'purple' }) {
  const colorMap = {
    purple: 'bg-purple-600/20 text-purple-400',
    blue: 'bg-blue-600/20 text-blue-400',
    green: 'bg-green-600/20 text-green-400',
    yellow: 'bg-yellow-600/20 text-yellow-400',
    red: 'bg-red-600/20 text-red-400',
  }
  return (
    <div className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-4">
      <div className="flex items-center gap-2 mb-2">
        <div className={`p-1.5 rounded ${colorMap[color]}`}>
          <Icon className="w-4 h-4" />
        </div>
        <span className="text-xs text-zinc-400">{label}</span>
      </div>
      <div className="text-2xl font-bold text-white">{value ?? '—'}</div>
      {sub && <div className="text-xs text-zinc-500 mt-1">{sub}</div>}
    </div>
  )
}

export default function AdminAnalyticsTab() {
  const { session } = useAuth()
  const [stats, setStats] = useState(null)
  const [genStats, setGenStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const headers = { Authorization: `Bearer ${session?.access_token}` }

      const [statsResp, genResp] = await Promise.all([
        fetch(`${BACKEND_BASE}/api/admin/stats`, { headers }),
        fetch(`${BACKEND_BASE}/api/generation-stats?limit=10000`),
      ])

      if (statsResp.ok) {
        setStats(await statsResp.json())
      }
      if (genResp.ok) {
        setGenStats(await genResp.json())
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (session?.access_token) fetchData()
  }, [session?.access_token])

  // Tier data for pie chart
  const tierData = useMemo(() => {
    if (!stats?.tier_counts) return []
    return Object.entries(stats.tier_counts)
      .filter(([, count]) => count > 0)
      .map(([tier, count]) => ({ name: tier, value: count }))
  }, [stats])

  // Generation records aggregated by day
  const genByDay = useMemo(() => {
    if (!genStats?.records?.length) return []
    const days = {}
    genStats.records.forEach(r => {
      const day = r.timestamp?.split('T')[0] || 'unknown'
      if (!days[day]) days[day] = { date: day, total: 0, success: 0, failed: 0 }
      days[day].total++
      if (r.success) days[day].success++
      else days[day].failed++
    })
    return Object.values(days).sort((a, b) => a.date.localeCompare(b.date)).slice(-30)
  }, [genStats])

  // Generation by type
  const genByType = useMemo(() => {
    if (!genStats?.records?.length) return []
    const types = {}
    genStats.records.forEach(r => {
      const t = r.job_type || 'unknown'
      types[t] = (types[t] || 0) + 1
    })
    return Object.entries(types)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value)
  }, [genStats])

  // Generation by resolution
  const genByResolution = useMemo(() => {
    if (!genStats?.records?.length) return []
    const res = {}
    genStats.records.forEach(r => {
      const key = r.resolution || 'unknown'
      res[key] = (res[key] || 0) + 1
    })
    return Object.entries(res)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 8)
  }, [genStats])

  // Duration histogram
  const durationBuckets = useMemo(() => {
    if (!genStats?.records?.length) return []
    const buckets = [
      { name: '<1m', min: 0, max: 60, count: 0 },
      { name: '1-5m', min: 60, max: 300, count: 0 },
      { name: '5-10m', min: 300, max: 600, count: 0 },
      { name: '10-20m', min: 600, max: 1200, count: 0 },
      { name: '20-30m', min: 1200, max: 1800, count: 0 },
      { name: '>30m', min: 1800, max: Infinity, count: 0 },
    ]
    genStats.records.forEach(r => {
      if (!r.duration_seconds) return
      const b = buckets.find(b => r.duration_seconds >= b.min && r.duration_seconds < b.max)
      if (b) b.count++
    })
    return buckets.map(({ name, count }) => ({ name, count }))
  }, [genStats])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <RefreshCw className="w-6 h-6 text-purple-400 animate-spin" />
      </div>
    )
  }

  const summary = genStats?.summary || {}
  const creditUtilization = stats ? Math.round((stats.total_credits_used / Math.max(stats.total_credits_issued, 1)) * 100) : 0

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-purple-400" />
          <h2 className="text-lg font-semibold text-white">Analytics</h2>
        </div>
        <button
          onClick={fetchData}
          className="p-1.5 text-zinc-400 hover:text-white hover:bg-zinc-700 rounded transition-colors"
          title="Refresh"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {error && (
        <div className="p-2 bg-red-900/30 border border-red-800 rounded text-sm text-red-300">{error}</div>
      )}

      {/* KPI Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard icon={Users} label="Total Users" value={stats?.total_users ?? '—'} color="purple" />
        <StatCard icon={Coins} label="Credits Issued" value={stats?.total_credits_issued?.toLocaleString() ?? '—'} color="blue" />
        <StatCard icon={TrendingUp} label="Credits Used" value={stats?.total_credits_used?.toLocaleString() ?? '—'} sub={`${creditUtilization}% utilization`} color="green" />
        <StatCard icon={Zap} label="Generations" value={summary.total ?? 0} sub={`${(summary.success_rate ?? 0).toFixed(1)}% success rate`} color="yellow" />
      </div>

      {/* Row 2: Tier Distribution + Credit Economy */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Tier Distribution */}
        <div className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-4">
          <h3 className="text-sm font-medium text-zinc-300 mb-3">User Tiers</h3>
          {tierData.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={tierData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={80}
                  paddingAngle={3}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  {tierData.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#27272a', border: '1px solid #3f3f46', borderRadius: '8px' }}
                  labelStyle={{ color: '#fff' }}
                />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-center py-8 text-zinc-500 text-sm">No tier data</div>
          )}
        </div>

        {/* Credit Economy Summary */}
        <div className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-4">
          <h3 className="text-sm font-medium text-zinc-300 mb-3">Credit Economy</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-zinc-400">Total Issued</span>
              <span className="text-sm font-mono text-white">{stats?.total_credits_issued?.toLocaleString() ?? '—'}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-zinc-400">Total Used</span>
              <span className="text-sm font-mono text-white">{stats?.total_credits_used?.toLocaleString() ?? '—'}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-zinc-400">Outstanding</span>
              <span className="text-sm font-mono text-emerald-400">
                {stats ? (stats.total_credits_issued - stats.total_credits_used).toLocaleString() : '—'}
              </span>
            </div>
            <div className="w-full bg-zinc-700 rounded-full h-2 mt-2">
              <div
                className="bg-gradient-to-r from-purple-500 to-blue-500 h-2 rounded-full transition-all"
                style={{ width: `${Math.min(creditUtilization, 100)}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-zinc-500">
              <span>0%</span>
              <span>{creditUtilization}% utilized</span>
              <span>100%</span>
            </div>
            <div className="pt-2 border-t border-zinc-700 space-y-1">
              <div className="flex justify-between text-sm">
                <span className="text-zinc-400">Admins</span>
                <span className="text-white">{stats?.total_admins ?? '—'}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-zinc-400">VIPs</span>
                <span className="text-white">{stats?.total_vips ?? '—'}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Generation Analytics */}
      {(genStats?.records?.length > 0) && (
        <>
          {/* Generations over time */}
          {genByDay.length > 1 && (
            <div className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-4">
              <h3 className="text-sm font-medium text-zinc-300 mb-3">Generations Over Time</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={genByDay}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
                  <XAxis dataKey="date" tick={{ fill: '#71717a', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#71717a', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#27272a', border: '1px solid #3f3f46', borderRadius: '8px' }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Legend />
                  <Bar dataKey="success" name="Success" fill="#10b981" stackId="gen" />
                  <Bar dataKey="failed" name="Failed" fill="#ef4444" stackId="gen" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Generation by Type + Resolution */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {genByType.length > 0 && (
              <div className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-4">
                <h3 className="text-sm font-medium text-zinc-300 mb-3">By Job Type</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={genByType}
                      cx="50%"
                      cy="50%"
                      outerRadius={70}
                      paddingAngle={2}
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${value}`}
                    >
                      {genByType.map((_, i) => (
                        <Cell key={i} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{ backgroundColor: '#27272a', border: '1px solid #3f3f46', borderRadius: '8px' }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}

            {genByResolution.length > 0 && (
              <div className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-4">
                <h3 className="text-sm font-medium text-zinc-300 mb-3">By Resolution</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={genByResolution} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
                    <XAxis type="number" tick={{ fill: '#71717a', fontSize: 11 }} />
                    <YAxis dataKey="name" type="category" tick={{ fill: '#71717a', fontSize: 11 }} width={90} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#27272a', border: '1px solid #3f3f46', borderRadius: '8px' }}
                    />
                    <Bar dataKey="value" name="Count" fill="#8b5cf6" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* Duration Distribution + Performance */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {durationBuckets.some(b => b.count > 0) && (
              <div className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-4">
                <h3 className="text-sm font-medium text-zinc-300 mb-3">Generation Duration</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={durationBuckets}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
                    <XAxis dataKey="name" tick={{ fill: '#71717a', fontSize: 11 }} />
                    <YAxis tick={{ fill: '#71717a', fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#27272a', border: '1px solid #3f3f46', borderRadius: '8px' }}
                    />
                    <Bar dataKey="count" name="Jobs" fill="#06b6d4" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Performance summary */}
            <div className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-4">
              <h3 className="text-sm font-medium text-zinc-300 mb-3">Performance</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm text-zinc-400">Total Generations</span>
                  <span className="text-sm font-mono text-white">{summary.total}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-zinc-400">Successful</span>
                  <span className="text-sm font-mono text-emerald-400">{summary.successful}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-zinc-400">Failed</span>
                  <span className="text-sm font-mono text-red-400">{summary.failed}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-zinc-400">Avg Duration</span>
                  <span className="text-sm font-mono text-white">
                    {summary.avg_duration ? `${Math.round(summary.avg_duration)}s` : '—'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-zinc-400">Min / Max</span>
                  <span className="text-sm font-mono text-white">
                    {summary.min_duration
                      ? `${Math.round(summary.min_duration)}s / ${Math.round(summary.max_duration)}s`
                      : '—'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Empty state for generation analytics */}
      {(!genStats?.records?.length) && (
        <div className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-8 text-center">
          <Activity className="w-8 h-8 text-zinc-600 mx-auto mb-2" />
          <p className="text-sm text-zinc-400">No generation data available yet</p>
          <p className="text-xs text-zinc-500 mt-1">Charts will appear once videos/images are generated</p>
        </div>
      )}
    </div>
  )
}
