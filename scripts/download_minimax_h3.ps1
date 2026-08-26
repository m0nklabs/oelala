$ErrorActionPreference = 'Continue'
$M = 'C:\PROGRAMME\ComfyUI_windows_portable\ComfyUI\models'
$BASE = 'https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main'
$LOG = 'C:\PROGRAMME\ComfyUI_windows_portable\h3_download.ps1.log'

function Log($msg) { Add-Content -Path $LOG -Value ("[{0:s}] {1}" -f (Get-Date), $msg) -ErrorAction SilentlyContinue }

$items = @(
  @{ sub = 'diffusion_models'; file = 'minimax_h3_fl2va_pruned_int8_convrot.safetensors'; min = 500MB },
  @{ sub = 'text_encoders';    file = 'qwen3vl_32b_minimax_h3_int8_convrot.safetensors'; min = 500MB },
  @{ sub = 'vae';              file = 'minimax_h3_video_vae_fp16.safetensors';            min = 1MB },
  @{ sub = 'vae';              file = 'minimax_h3_audio_vae_fp32.safetensors';            min = 1MB },
  @{ sub = 'loras';            file = 'minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors'; min = 1MB }
)

foreach ($it in $items) {
  $out = Join-Path $M (Join-Path $it.sub $it.file)
  $url = "$BASE/$($it.sub)/$($it.file)?download=1"
  for ($i = 1; $i -le 10; $i++) {
    Log ("POGING {0}: {1}" -f $i, $it.file)
    & curl.exe -L --retry 3 --retry-delay 5 -sS -o $out $url
    $ec = $LASTEXITCODE
    $sz = if (Test-Path $out) { (Get-Item $out).Length } else { 0 }
    Log "curl exit=$ec size=$sz"
    if ($ec -eq 0 -and $sz -gt $it.min) { Log "OK: $($it.file) = $sz bytes"; break }
    Remove-Item $out -Force -ErrorAction SilentlyContinue
    Start-Sleep 15
  }
}
Log "H3 download klaar (of max 10 pogingen per bestand)"
