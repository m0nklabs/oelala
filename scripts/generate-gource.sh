#!/bin/bash
# Generate Gource visualization for Oelala repository
# Run: ./scripts/generate-gource.sh

set -e

REPO_DIR="/home/flip/oelala"
OUTPUT_DIR="${REPO_DIR}/media"
OUTPUT_FILE="${OUTPUT_DIR}/oelala-gource-latest.mp4"
TEMP_PPM="/tmp/gource-oelala.ppm"

echo "🎬 Generating Gource visualization for Oelala..."

cd "$REPO_DIR"

# Generate gource visualization and pipe to ffmpeg
# Using xvfb-run for headless rendering
xvfb-run -a gource \
    --title "Oelala - AI Media Creation Platform" \
    --key \
    --highlight-users \
    --hide mouse,filenames \
    --seconds-per-day 0.5 \
    --auto-skip-seconds 1 \
    --file-idle-time 0 \
    --max-files 0 \
    --background-colour 0D1117 \
    --font-colour FFFFFF \
    --dir-colour 58A6FF \
    --highlight-colour FF7B72 \
    --date-format "%Y-%m-%d" \
    --viewport 1920x1080 \
    --stop-at-end \
    --output-ppm-stream - \
    . \
    2>/dev/null | \
ffmpeg -y -r 60 -f image2pipe -vcodec ppm -i - \
    -vcodec libx264 -preset medium -pix_fmt yuv420p -crf 18 \
    "$OUTPUT_FILE" \
    2>/dev/null

echo "✅ Generated: $OUTPUT_FILE"
echo "📊 Size: $(du -h "$OUTPUT_FILE" | cut -f1)"
