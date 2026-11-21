### Changed

- `docs/OELALA_LOOP.md` §0 spells out the throwaway-venv test recipe exactly: `requirements-web.txt`
  plus `pytest pytest-asyncio pytest-cov ruff==0.15.0`. A plugins-only install collects nothing
  (27 collection errors, proven today); the full recipe gives 705 passed / 8 skipped. Recorded after
  a rebuilt-from-memory env failed exactly that way.
