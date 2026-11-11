### Added

- The frontend can run unit tests: `vitest` + `jsdom` + `@testing-library/react` as
  devDependencies, `npm test` wired to `vitest run`, and a `test` block in `vite.config.js`. CI
  already runs `npm test -- --run`, which was a no-op without a test script — from this change on it
  executes the suite (still masked by `|| true`; dropping that is a one-line workflow edit).
- `hooks/useObjectUrls.js` with `useObjectUrl` and `useTrackedObjectUrls`, plus 9 tests covering the
  three lifecycle bugs that took three review rounds on #244: revoking the replaced preview, revoking
  the dropped reference preview, and revoking what is still live on unmount — including URLs added
  *after* mount, which is the captured-empty-Set bug.
- `dashboard/__tests__/nav.test.js` guarding the nav: every tool id has exactly one entry, no
  duplicates, and a nav entry cannot name an id that does not exist.

### Changed

- `VideoToVideoTool` uses the two hooks instead of its own inline object-URL bookkeeping (29 lines
  gone). No behaviour change: the hooks are the same logic, now covered by tests.
- Dropped four unreferenced tool ids from `nav.js` (`MY_MEDIA_VIDEOS`, `MY_MEDIA_IMAGES`,
  `MY_MEDIA_AUDIO`, `MY_MEDIA_PROMPTS`). `MY_MEDIA_ALL` replaced them; the only remaining mention is
  the historical plan in `docs/UI_V2_PLAN.md`. The new nav test is what found them and now prevents
  their return.

### Notes

- Every mutation tried against the hook was caught: reading a captured Set instead of `ref.current`,
  never revoking on unmount, revoking the new URL instead of the old one, and no revoke at all.
- `eslint src` unchanged at 0 errors / 47 warnings; `npm run build` succeeds.
