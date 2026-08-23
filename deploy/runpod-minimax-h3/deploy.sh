#!/usr/bin/env bash
# ============================================================
# RunPod MiniMax-H3 Worker Deploy Script
# ============================================================
# Builds the Docker image, tags with a dated version, pushes to
# GHCR, and updates the RunPod template — all in one shot.
#
# This is the MiniMax-H3 video+audio variant (t2v / i2v).
# Target: 80 GB+ GPUs (A100, H100, B200, GB200)
#
# Usage:
#   ./deploy.sh              # Build, push, update template
#   ./deploy.sh --skip-build # Push existing image, update template
#   ./deploy.sh --dry-run    # Show what would happen, don't execute
#
# Before the FIRST deploy, create the template + endpoint on RunPod
# (see docs/RUNPOD_NEW_ENDPOINT_CHECKLIST.md) and set:
#   RUNPOD_MINIMAX_H3_TEMPLATE_ID  (or edit TEMPLATE_ID below)
#   RUNPOD_MINIMAX_H3_ENDPOINT_ID  (or edit ENDPOINT_ID below)
#
# Requires:
#   - docker (logged into ghcr.io)
#   - python3 + httpx (pip install httpx)
#   - RUNPOD_API_KEY in ../../.env or environment
# ============================================================

set -euo pipefail

# ---- Config ----
REGISTRY="ghcr.io/m0nklabs/oelala-minimax-h3-worker"
TEMPLATE_ID="${RUNPOD_MINIMAX_H3_TEMPLATE_ID:-fpfo4gmnrw}"
ENDPOINT_ID="${RUNPOD_MINIMAX_H3_ENDPOINT_ID:-5xuvnvyww4ujnc}"
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DEPLOY_DIR/../.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

# ---- Parse args ----
SKIP_BUILD=false
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=true ;;
        --dry-run)    DRY_RUN=true ;;
        -h|--help)
            echo "Usage: $0 [--skip-build] [--dry-run] [-h|--help]"
            echo ""
            echo "  --skip-build   Skip Docker build, just push + update template"
            echo "  --dry-run      Show what would happen without executing"
            echo "  -h, --help     Show this help"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# ---- Load RunPod API key ----
if [[ -z "${RUNPOD_API_KEY:-}" ]] && [[ -f "$ENV_FILE" ]]; then
    RUNPOD_API_KEY=$(grep -E '^RUNPOD_API_KEY=' "$ENV_FILE" | cut -d= -f2- | tr -d '"' | tr -d "'")
fi
if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
    echo "❌ RUNPOD_API_KEY not found in environment or $ENV_FILE"
    exit 1
fi

# ---- Validate template/endpoint IDs ----
if [[ "$TEMPLATE_ID" == "PLACEHOLDER" ]] && ! $DRY_RUN; then
    echo "❌ TEMPLATE_ID is PLACEHOLDER. Set RUNPOD_MINIMAX_H3_TEMPLATE_ID or update this script."
    echo "   Create a new template on RunPod first (see docs/RUNPOD_NEW_ENDPOINT_CHECKLIST.md),"
    echo "   then set the ID."
    exit 1
fi

# ---- Generate version tag ----
VERSION_TAG="$(date +%Y%m%d-%H%M%S)"
FULL_IMAGE="${REGISTRY}:${VERSION_TAG}"
LATEST_IMAGE="${REGISTRY}:latest"

echo "============================================"
echo "🚀 RunPod MiniMax-H3 Worker Deploy"
echo "============================================"
echo "  Version:   ${VERSION_TAG}"
echo "  Image:     ${FULL_IMAGE}"
echo "  Template:  ${TEMPLATE_ID}"
echo "  Endpoint:  ${ENDPOINT_ID}"
echo "  Skip build: ${SKIP_BUILD}"
echo "  Dry run:    ${DRY_RUN}"
echo "============================================"

if $DRY_RUN; then
    echo ""
    echo "🔍 DRY RUN — no changes will be made"
    echo ""
    echo "Would execute:"
    if ! $SKIP_BUILD; then
        echo "  1. docker build -t ${LATEST_IMAGE} -f ${DEPLOY_DIR}/Dockerfile ${DEPLOY_DIR}"
    fi
    echo "  2. docker tag ${LATEST_IMAGE} ${FULL_IMAGE}"
    echo "  3. docker push ${FULL_IMAGE}"
    echo "  4. docker push ${LATEST_IMAGE}"
    echo "  5. Update RunPod template ${TEMPLATE_ID} → ${FULL_IMAGE}"
    echo ""
    echo "✅ Dry run complete — nothing was changed."
    exit 0
fi

# ---- Step 1: Build ----
if ! $SKIP_BUILD; then
    echo ""
    echo "📦 Step 1/5: Building Docker image..."
    docker build -t "${LATEST_IMAGE}" -f "${DEPLOY_DIR}/Dockerfile" "${DEPLOY_DIR}"
    echo "✅ Build complete"
else
    echo ""
    echo "⏭️  Step 1/5: Skipping build (--skip-build)"
fi

# ---- Step 2: Tag ----
echo ""
echo "🏷️  Step 2/5: Tagging as ${VERSION_TAG}..."
docker tag "${LATEST_IMAGE}" "${FULL_IMAGE}"
echo "✅ Tagged"

# ---- Step 3: Push versioned tag ----
echo ""
echo "⬆️  Step 3/5: Pushing ${FULL_IMAGE}..."
docker push "${FULL_IMAGE}"
echo "✅ Pushed versioned tag"

# ---- Step 4: Push :latest (for reference) ----
echo ""
echo "⬆️  Step 4/5: Pushing ${LATEST_IMAGE}..."
docker push "${LATEST_IMAGE}"
echo "✅ Pushed :latest"

# ---- Step 5: Update RunPod template ----
echo ""
echo "🔄 Step 5/6: Updating RunPod template ${TEMPLATE_ID} → ${FULL_IMAGE}..."

HF_LORA_TOKEN=$(grep -E '^HF_LORA_TOKEN=' "$ENV_FILE" | cut -d= -f2- | tr -d '"' | tr -d "'") || true
if [[ -z "${HF_LORA_TOKEN:-}" ]]; then
    HF_ENV_STR=""
else
    HF_ENV_STR="{ key: \"HF_TOKEN\", value: \"${HF_LORA_TOKEN}\" },"
fi

TEMPLATE_RESULT=$(python3 -c "
import httpx, json, sys

resp = httpx.post(
    'https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}',
    json={
        'query': '''
            mutation {
                saveTemplate(input: {
                    id: \"${TEMPLATE_ID}\"
                    name: \"oelala-minimax-h3-worker\"
                    imageName: \"${FULL_IMAGE}\"
                    containerDiskInGb: 100
                    volumeInGb: 0
                    dockerArgs: \"\"
                    env: [
                        ${HF_ENV_STR}
                        { key: \"COMFYUI_PATH\", value: \"/comfyui\" }
                    ]
                    containerRegistryAuthId: \"cmmbssf3a00anky07egrtupgt\"
                }) {
                    id
                    imageName
                    name
                }
            }
        '''
    },
    timeout=30
)

data = resp.json()
if 'errors' in data:
    print(f'ERROR: {json.dumps(data[\"errors\"])}', file=sys.stderr)
    sys.exit(1)

saved = data.get('data', {}).get('saveTemplate', {})
print(f'Template updated: {saved.get(\"id\")} → {saved.get(\"imageName\")}')
")

if [[ $? -ne 0 ]]; then
    echo "❌ Failed to update RunPod template!"
    echo "$TEMPLATE_RESULT"
    exit 1
fi

echo "✅ ${TEMPLATE_RESULT}"

# ---- Step 6: Apply endpoint defaults ----
echo ""
echo "🎛️  Step 6/6: Applying RunPod endpoint defaults..."
python3 "$PROJECT_ROOT/scripts/apply_runpod_endpoint_defaults.py" --profile minimax_h3
echo "✅ Endpoint defaults applied"

# ---- Done ----
echo ""
echo "============================================"
echo "🎉 Deploy complete!"
echo "============================================"
echo "  Image:    ${FULL_IMAGE}"
echo "  Template: ${TEMPLATE_ID} updated"
echo ""
echo "  Next cold start will pull the new image."
if [[ "$ENDPOINT_ID" != "PLACEHOLDER" ]]; then
    echo "  Monitor: https://www.runpod.io/console/serverless/${ENDPOINT_ID}/workers"
fi
echo "============================================"
