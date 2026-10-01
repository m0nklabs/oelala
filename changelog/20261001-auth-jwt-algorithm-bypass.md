### Security
- **Fixed an authentication bypass on every protected endpoint.** `auth.py`
  verified JWTs against Supabase's JWKS with `algorithms=["RS256"]`, but Supabase
  issues **ES256** (EC P-256) tokens. Every legitimate token therefore failed with
  `InvalidAlgorithmError`, which the code caught and silently fell through to a
  last-resort branch that decoded the token **without any signature check**. That
  branch required only a `sub` claim, so anyone able to reach the API could forge a
  token for any user id and act as that user — reading their data and spending their
  credits. The `RS256` restriction turned the emergency fallback into the de-facto
  authentication path, and the "Cloudflare Tunnel provides transport security"
  justification did not hold: a tunnel provides TLS and origin hiding, not issuer
  authentication, so a forged token passes through it unchanged.
  - The JWKS path now accepts `ES256`, `ES384`, `RS256` and `RS384`.
  - The unverified fallback is gated behind `AUTH_ALLOW_UNVERIFIED_JWT` (default
    on, so the change could not lock anyone out before the ES256 path was confirmed
    in production) and now logs a warning naming the risk.
  - The active path is logged (`JWT verified via JWKS`), so the verification route
    is observable instead of silent.

  Verified end to end against a throwaway Supabase user created and deleted for the
  test: a genuinely signed ES256 token authenticates (HTTP 200) and is logged as
  `verified via JWKS`, while an HS256 token forged with a random key is rejected
  (HTTP 401) with `AUTH_ALLOW_UNVERIFIED_JWT=0`. The operator's live sessions were
  confirmed to keep working through the same JWKS path.
