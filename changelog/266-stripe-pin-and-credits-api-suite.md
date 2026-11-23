### Added

- Behavioral suite for the credits API (20 tests): balance/welcome-credits fallback, packages
  fallback-on-error, estimate breakdown flags, history cap, purchase (503 without key, 404 unknown
  package, checkout-session construction with the `{CHECKOUT_SESSION_ID}` placeholder, Stripe-error
  500), and the webhook money path (signature-verified events credit the user; unpaid events stay
  pending; missing user/credits never credit; add-credits failure returns 500 so Stripe retries).
  Plus `check_credits`' 402 envelope — the exact shape the frontend's insufficient-credits flow
  consumes.

### Fixed

- `stripe` was declared in **no manifest at all** — the purchase endpoint and webhook imported it
  lazily inside the functions, so every install (production GPU venv included) shipped without it and
  both endpoints 500'd with ModuleNotFoundError at call time. Found because the behavioral tests ran
  the endpoints; pinned `stripe==17.1.0` in both requirements files.
