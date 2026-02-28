### Added

- `GET /share/{media_id}` endpoint returning HTML with Open Graph + Twitter Card meta tags and JS redirect to `/?openItem={id}`
- Dashboard reads `?openItem` URL param on mount → switches active tool to Gallery
- Gallery reads `?openItem` URL param on mount → auto-fetches item and opens detail modal
- MediaDetailModal share button now copies `/share/{id}` URL; uses native `navigator.share` on supported browsers
- `SITE_URL` env var controls canonical base URL in share meta tags (default: `http://localhost:5174`)
