### Fixed

- Lowered the Cloud Max text-to-video default preset to a safer 720p 5s profile and capped frontend duration ranges per resolution so serverless jobs do not queue known timeout-heavy settings by default.
- Added backend validation for oversized Cloud Max text-to-video requests so unsupported pixel-frame budgets fail fast instead of timing out after submission.
