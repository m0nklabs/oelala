### Fixed

- Scoped `/comfyui/queue` and `/comfyui/job/{prompt_id}` to the authenticated job owner so queue completions and signed media URLs no longer leak into other sessions.
- Switched the dashboard queue indicator to authenticated API requests and clear its local queue state when no scoped queue data is available.
