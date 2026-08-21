#!/usr/bin/env bash
# Sync AGENTS.md (source of truth) to tool-native copies.
# Use this INSTEAD of symlinks on Windows / OSS-shared repos.
# (oelala ships Windows scripts/bat files, so sync-copies keep AGENTS.md
#  the single source of truth without relying on symlink support.)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Remove symlinks if present (avoid copying over them = drift)
for f in CLAUDE.md .goosehints; do
  if [ -L "$f" ]; then rm "$f"; fi
done

cp AGENTS.md CLAUDE.md
cp AGENTS.md .goosehints
echo "Synced AGENTS.md → CLAUDE.md + .goosehints"
