### Fixed

- Prompt generation silently fell back to the raw input: the Guardian call had
  a 120 s timeout while reasoning models without a max_tokens cap need minutes
  (measured 145-344 s for the H3 skill on the 9B route). The timeout is now
  600 s, and transient Guardian 502/504 responses are retried like 503s.
- Prompt Generator now shows a warning when a template fallback was used
  (`llm_used: false`) instead of presenting the barely-enhanced output as a
  normal result.
