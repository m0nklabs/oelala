### Fixed

- The Image-to-Video tool no longer assumes a restored source frame lives under `uploads/`: it
  probes the media segments (`uploads`, `images`, `videos`) and uses the first that exists, so a
  generated output reused as a source resolves instead of 404ing.
- `GET /user/media/{type}/{filename}` logs a missing object at WARNING instead of ERROR. A
  NoSuchKey for a filename the client guessed is a normal 404; real storage failures keep ERROR so
  they stay visible. Detected via minio's `S3Error.code` (with a string fallback).
