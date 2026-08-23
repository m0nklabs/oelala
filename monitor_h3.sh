#!/bin/bash
# Wacht tot minimaal het int8-model EN de text-encoder compleet zijn (resume-veilig).
set -u
for i in $(seq 1 30); do
  sleep 240
  SIZES=$(ssh -o BatchMode=yes -o ConnectTimeout=8 teams-host "powershell -NoProfile -Command \"Get-Item 'C:\PROGRAMME\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors','C:\PROGRAMME\ComfyUI_windows_portable\ComfyUI\models\text_encoders\qwen3vl_32b_minimax_h3_int8_convrot.safetensors' -ErrorAction SilentlyContinue | ForEach-Object { Write-Output ('{0}={1:N2}GB' -f \$_.Name, (\$_.Length/1GB)) }\"" 2>/dev/null)
  echo "[$(date +%H:%M:%S)] ${SIZES}"
  MODEL=$(echo "$SIZES" | awk -F= '/minimax_h3_fl2va_pruned_int8_convrot/{print int($2+0)}')
  TE=$(echo "$SIZES" | awk -F= '/qwen3vl_32b_minimax_h3_int8_convrot/{print int($2+0)}')
  if [ "${MODEL:-0}" -ge 19 ] && [ "${TE:-0}" -ge 10 ]; then
    echo "KLAAR: ${SIZES}"
    exit 0
  fi
done
echo "einde monitor (30 rondes, niet klaar)"
exit 1
