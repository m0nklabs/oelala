#!/usr/bin/env bash

set -euo pipefail

echo "This script is intentionally disabled."
echo "RunPod Network Volume storage is reserved for LoRAs and private/custom assets only."
echo "General Hugging Face models must not be preloaded onto the volume."
echo "Use deploy/runpod/upload_private_assets.py for selected LoRA/private asset uploads."
exit 1
