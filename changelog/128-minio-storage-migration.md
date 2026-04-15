### Added
- Dual-node CDN distribution: public media served via round-robin across `storage.oelala.xyz` + `storage2.oelala.xyz`
- Backblaze B2 offsite backup: all 4 buckets mirrored every 15 min via `scripts/minio-backup-mirror.sh`

### Changed
- **Storage Migration: oelala-storage → MinIO** — Replaced custom Go storage with MinIO S3-compatible object storage. Backend uses `minio` Python SDK directly. 1.1 GiB / 1,059 objects migrated.

### Removed
- oelala-storage Go service dependency
- B2 as primary storage (now backup-only)
