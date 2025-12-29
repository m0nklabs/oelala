import BACKEND_BASE from './config'

export async function sendClientLog(entry) {
  try {
    await fetch(`${BACKEND_BASE}/client-log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(entry),
    })
  } catch (e) {
    // best-effort; don't throw to avoid cascading UI errors
    // fallback: write to console
    console.error('Failed to send client log', e)
  }
}
