### Fixed
- Hardened backend media and metadata routes against path traversal, SSRF, command-line injection, and stack trace exposure findings from GitHub code scanning.
- Sanitized image-to-video frontend filenames before converting fetched media blobs into `File` objects.