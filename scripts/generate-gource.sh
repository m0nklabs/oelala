#!/bin/bash
# Generate Gource visualization for Oelala repository
# Run: ./scripts/generate-gource.sh

set -e

REPO_DIR="/home/flip/oelala"
OUTPUT_DIR="${REPO_DIR}/media"
OUTPUT_MP4="${OUTPUT_DIR}/oelala-gource-latest.mp4"
OUTPUT_GIF="${OUTPUT_DIR}/oelala-gource-preview.gif"

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
    "$OUTPUT_MP4" \
    2>/dev/null

echo "✅ Generated: $OUTPUT_MP4"
echo "📊 Size: $(du -h "$OUTPUT_MP4" | cut -f1)"

# Generate GIF preview for GitHub README (full video at 2x slowmo)
echo "🎞️ Generating GIF preview..."
ffmpeg -y -i "$OUTPUT_MP4" \
    -vf "setpts=2*PTS,fps=8,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer" \
    -loop 0 "$OUTPUT_GIF" \
    2>/dev/null

echo "✅ Generated: $OUTPUT_GIF"
echo "📊 Size: $(du -h "$OUTPUT_GIF" | cut -f1)"
