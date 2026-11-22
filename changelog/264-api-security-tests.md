### Added

- Unit tests for the security-critical `src/frontend/src/api.js` (9 tests): the cross-origin guard
  rejects non-allow-listed endpoints, the Supabase bearer token is attached only to backend/browser
  origins (never to the storage origins) and is absent on unauthenticated calls, Content-Type is set
  for JSON but not FormData, the `_cors` bust param lands on cached static media paths, and the 402
  insufficient-credits envelope is normalised and dispatched for both `postForm` and `postJson`.
  Frontend suite is now 24 tests across 4 files. Mutation-proven: flipping the cross-origin
  assertion turns the suite red.
