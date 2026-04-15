#!/usr/bin/env bash
# MinIO backup mirror — syncs all oelala buckets to:
#   1. Node 2 (192.168.1.62) — active serving replica
#   2. Backblaze B2 (oelala-media-eu) — cold offsite backup
# Runs via cron on ai-kvm2: */15 * * * * /home/flip/oelala/scripts/minio-backup-mirror.sh
set -euo pipefail

LOG="/home/flip/oelala/logs/minio-mirror.log"
MC="/usr/local/bin/mc"
SRC="oelala"       # ai-kvm2 localhost:9000
DST="node2"        # 192.168.1.62:9000
B2="b2"            # Backblaze B2 (s3.eu-central-003.backblazeb2.com)
B2_BUCKET="oelala-media-eu"

BUCKETS="oelala-generated oelala-comfyui oelala-avatars oelala-users"

echo "[$(date -Is)] Mirror started" >> "$LOG"

# Mirror to node 2 (active replica)
for bucket in $BUCKETS; do
    $MC mirror --preserve --quiet "$SRC/$bucket/" "$DST/$bucket/" >> "$LOG" 2>&1 || true
done

# Mirror to B2 (offsite backup) — flattened into sub-prefixes
for bucket in $BUCKETS; do
    $MC mirror --preserve --quiet "$SRC/$bucket/" "$B2/$B2_BUCKET/$bucket/" >> "$LOG" 2>&1 || true
done

echo "[$(date -Is)] Mirror completed" >> "$LOG"
