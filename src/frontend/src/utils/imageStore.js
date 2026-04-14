/**
 * IndexedDB-based image store for session persistence.
 * sessionStorage has a ~5MB limit which is too small for data URLs of large images.
 * IndexedDB supports hundreds of MBs, so images survive page navigation reliably.
 */

const DB_NAME = 'oelala-image-store'
const DB_VERSION = 1
const STORE_NAME = 'images'

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME)
      }
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

/**
 * Store a blob/data URL under a key.
 * @param {string} key - Storage key (e.g. 'i2t_preview')
 * @param {Blob|string} value - Blob object or data URL string
 */
export async function storeImage(key, value) {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    tx.objectStore(STORE_NAME).put(value, key)
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}

/**
 * Retrieve a stored image by key.
 * @param {string} key
 * @returns {Promise<Blob|string|null>}
 */
export async function getImage(key) {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly')
    const req = tx.objectStore(STORE_NAME).get(key)
    req.onsuccess = () => resolve(req.result ?? null)
    req.onerror = () => reject(req.error)
  })
}

/**
 * Remove a stored image by key.
 * @param {string} key
 */
export async function removeImage(key) {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    tx.objectStore(STORE_NAME).delete(key)
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}
