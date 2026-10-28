### Fixed

- `GET /api/media/unified?source=public` and the `/api/gallery/<id>/file` download path
  read `published_media` through a `supabase_utils` module that does not exist in the
  repository. The surrounding `except Exception` swallowed the `ImportError` into a
  debug log, so the public gallery silently returned nothing while the endpoint answered
  200. Both call sites now use `gallery_api.get_supabase_client()` — the same helper the
  rest of `app.py` already uses for that table.

### Added

- `tests/test_media_unified_public_source.py` pins the behaviour (the `published_media`
  query must actually run, not just return 200) and guards the module name, since a
  missing import was the whole defect.
- `tests/conftest.py` stubs `face_train_service.recover_stuck_jobs()` before any test
  module imports `app`, so the suite no longer marks live training jobs as failed just by
  being collected (exists per #220, which tracks the central fix).
