### Added

- Static linting for the frontend (`src/frontend/`): ESLint 9 flat config in `src/frontend/eslint.config.js` with `@eslint/js` recommended, `eslint-plugin-react` (JSX-aware, automatic runtime), `eslint-plugin-react-hooks` and browser/node globals. `no-undef` is an **error** — `vite build` does not check identifiers, so ReferenceErrors like the shipped `resolved is not defined` are now caught statically. `no-unused-vars` is also an error (`^_`-prefixed args/vars are ignored). Dev deps: `eslint`, `@eslint/js`, `eslint-plugin-react`, `eslint-plugin-react-hooks`, `globals`. Run with `npm run lint` in `src/frontend/`.
- Deliberately not enabled (yet): formatting/stylistic rules are not configured at all; `react/prop-types`, `react/display-name` and `react/react-in-jsx-scope` are off; `react-hooks/exhaustive-deps`, `react/no-unescaped-entities`, `react/jsx-no-useless-fragment` and `react/no-unknown-property` start as warnings because of pre-existing findings. Empty catch blocks (`catch {}`) are accepted via `no-empty` `allowEmptyCatch`.
- Lint is intentionally **not** wired into `npm run build`: systemd unit `oelala-frontend` runs the build in `ExecStartPre`, so a lint failure there would break deploys. Wiring lint into CI is a separate follow-up step.

### Fixed

- Two runtime ReferenceErrors surfaced by the new `no-undef` check: `wrapWithSuspense` called outside the scope where it is defined in `src/dashboard/Dashboard.jsx` (crashed the user-profile overlay) and `setNegPrompt` (non-existent setter) in `src/dashboard/tools/ImageToVideoTool.jsx` (crashed the "extract prompt from image" button).
