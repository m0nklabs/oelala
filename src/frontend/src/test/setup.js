import '@testing-library/jest-dom/vitest'

// Blob URLs are a jsdom gap with a realm trap behind it. jsdom 31 implements no
// blob-URL API at all, and under vitest the globals resolve to Node's URL while
// `Blob` stays jsdom's — so URL.createObjectURL(new Blob(...)) throws
//   TypeError: The "obj" argument must be an instance of Blob
// for an object that *is* a Blob, just from the other realm (measured 2026-11-11).
//
// The hooks under test only need the lifecycle, not real blobs, so install a
// deterministic stub unconditionally. That keeps the suite independent of which
// URL implementation the environment happens to resolve.
let sequence = 0
URL.createObjectURL = () => `blob:test/${++sequence}`
URL.revokeObjectURL = () => {}
