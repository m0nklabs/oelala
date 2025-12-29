// Lightweight API helper with graceful JSON parsing fallback
export async function postForm(url, formData, headers = {}) {
  const res = await fetch(url, {
    method: 'POST',
    body: formData,
    headers,
    credentials: 'same-origin',
  })

  const text = await res.text()
  try {
    const data = text ? JSON.parse(text) : null
    return { ok: res.ok, status: res.status, data }
  } catch (e) {
    // Fallback: return raw text when JSON parsing fails
    return { ok: res.ok, status: res.status, data: text }
  }
}

export async function getJson(url) {
  const res = await fetch(url, { method: 'GET', credentials: 'same-origin' })
  const text = await res.text()
  try {
    const data = text ? JSON.parse(text) : null
    return { ok: res.ok, status: res.status, data }
  } catch (e) {
    return { ok: res.ok, status: res.status, data: text }
  }
}
