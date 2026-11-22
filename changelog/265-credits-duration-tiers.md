### Fixed

- `calculate_credits` duration tiers were dead code: `>5 s → 1.5x` came before `>10 s → 2.0x`, so the
  elif never fired and every video above 5 s paid a flat 1.5x. Reordered so the intended tiers apply —
  a 12 s I2V video now prices 16 credits instead of 12. Found via coverage analysis (the branch was
  uncoverable by construction). Pricing tests added for the full tier table plus three error-path
  tests closing `api_key_auth.py` to 100%.
